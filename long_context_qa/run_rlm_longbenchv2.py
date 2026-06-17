import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Any

# If the `rlm` package is not pip-installed, point RLM_PATH at its source checkout.
_RLM_PATH = os.environ.get("RLM_PATH")
if _RLM_PATH:
    sys.path.insert(0, _RLM_PATH)

from rlm import RLM
from rlm.logger import RLMLogger


# Override any of these via CLI flags or environment variables.
DEFAULT_DATA_PATH = None  # required: path to LongBench-v2 data.json (or set LONGBENCH_V2_DATA_PATH)
DEFAULT_MODEL_NAME = None  # required: served model name / path (or set RLM_MODEL_NAME)
DEFAULT_VLLM_URL = "http://127.0.0.1:8000/v1"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_OUTPUT_DIR = "./outputs/longbench_v2_rlm"


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the RLM evaluation on the full LongBench-v2 benchmark."
    )
    parser.add_argument("--data-path", default=os.environ.get("LONGBENCH_V2_DATA_PATH", DEFAULT_DATA_PATH))
    parser.add_argument("--model-name", default=os.environ.get("RLM_MODEL_NAME", DEFAULT_MODEL_NAME))
    parser.add_argument("--vllm-url", default=os.environ.get("VLLM_URL", DEFAULT_VLLM_URL))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", DEFAULT_API_KEY))
    parser.add_argument("--output-dir", default=os.environ.get("RLM_EVAL_DIR", DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-iterations", type=int, default=int(os.environ.get("RLM_MAX_ITERATIONS", "15")))
    parser.add_argument("--max-depth", type=int, default=int(os.environ.get("RLM_MAX_DEPTH", "2")))
    parser.add_argument("--max-concurrent-subcalls", type=int,
                        default=int(os.environ.get("RLM_MAX_CONCURRENT_SUBCALLS", "4")))
    parser.add_argument("--compaction", action=argparse.BooleanOptionalAction,
                        default=env_bool("RLM_COMPACTION", True))
    parser.add_argument("--compaction-threshold-pct", type=float,
                        default=float(os.environ.get("RLM_COMPACTION_THRESHOLD_PCT", "0.75")))
    parser.add_argument("--timeout-base", type=float, default=float(os.environ.get("RLM_TIMEOUT_BASE", "1800")),
                        help="Timeout (s) for contexts up to 1M chars.")
    parser.add_argument("--timeout-mid", type=float, default=float(os.environ.get("RLM_TIMEOUT_MID", "3600")),
                        help="Timeout (s) for contexts between 1M and 3M chars.")
    parser.add_argument("--timeout-high", type=float, default=float(os.environ.get("RLM_TIMEOUT_HIGH", "7200")),
                        help="Timeout (s) for contexts above 3M chars.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction,
                        default=env_bool("RLM_VERBOSE", True))
    parser.add_argument("--domain", action="append", help="Optional domain filter. May be passed more than once.")
    parser.add_argument("--sub-domain", action="append", help="Optional sub-domain filter. May be passed more than once.")
    parser.add_argument("--task-id", action="append", help="Optional task id filter. May be passed more than once.")
    parser.add_argument("--limit", type=int, help="Optional cap after filters and sorting.")
    parser.add_argument("--retry-errors", action="store_true", help="Rerun existing result files whose error field is set.")
    parser.add_argument("--overwrite", action="store_true", help="Rerun tasks even if a result file already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected task counts without calling the RLM.")
    return parser.parse_args()


def log(message: str = "") -> None:
    print(message, flush=True)


def root_prompt(item: dict[str, Any]) -> str:
    return (
        f"{item['question']}\n\n"
        f"(A) {item['choice_A']}\n"
        f"(B) {item['choice_B']}\n"
        f"(C) {item['choice_C']}\n"
        f"(D) {item['choice_D']}\n\n"
        'Format your final answer as: "The correct answer is (X)" where X is A, B, C, or D. Do not explain.'
    )


def extract_prediction(text: str) -> str | None:
    patterns = [
        r"correct answer is\s*\(([A-D])\)",
        r"correct answer\s*[:\-]\s*\(?([A-D])\)?",
        r"final answer\s*[:\-]\s*\(?([A-D])\)?",
        r"answer is\s*\(?([A-D])\)?",
        r"^\s*\(?([A-D])\)?\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def compute_timeout(context_chars: int, args: argparse.Namespace) -> float:
    if context_chars > 3_000_000:
        return args.timeout_high
    if context_chars > 1_000_000:
        return args.timeout_mid
    return args.timeout_base


def selected_tasks(data: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    tasks = data
    if args.domain:
        domains = set(args.domain)
        tasks = [item for item in tasks if item["domain"] in domains]
    if args.sub_domain:
        sub_domains = set(args.sub_domain)
        tasks = [item for item in tasks if item["sub_domain"] in sub_domains]
    if args.task_id:
        task_ids = set(args.task_id)
        tasks = [item for item in tasks if item["_id"] in task_ids]

    tasks = sorted(tasks, key=lambda item: (item["domain"], item["sub_domain"], len(item["context"]), item["_id"]))
    if args.limit is not None:
        tasks = tasks[:args.limit]
    return tasks


def result_should_skip(result_path: str, args: argparse.Namespace) -> tuple[bool, dict[str, Any] | None]:
    if args.overwrite or not os.path.exists(result_path):
        return False, None
    existing = json.load(open(result_path))
    if args.retry_errors and existing.get("error"):
        return False, existing
    return True, existing


def make_stats(results: list[dict[str, Any]], total_selected: int) -> dict[str, Any]:
    done = [r for r in results if not r.get("error")]
    errors = [r for r in results if r.get("error")]
    n_correct = sum(1 for r in done if r.get("correct"))

    def group_stats(key_fn):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in done:
            grouped[key_fn(result)].append(result)
        out = {}
        for key, items in sorted(grouped.items()):
            correct = sum(1 for item in items if item.get("correct"))
            out[key] = {
                "correct": correct,
                "done": len(items),
                "accuracy": correct / len(items) if items else None,
            }
        return out

    return {
        "framework": "rlm",
        "method": "rlm_recursive",
        "total_selected": total_selected,
        "results_recorded": len(results),
        "done": len(done),
        "errors": len(errors),
        "correct": n_correct,
        "accuracy": n_correct / len(done) if done else None,
        "avg_elapsed_seconds": (
            sum(r.get("elapsed_seconds", 0) for r in done) / len(done) if done else None
        ),
        "by_domain": group_stats(lambda r: r["domain"]),
        "by_sub_domain": group_stats(lambda r: f"{r['domain']} / {r['sub_domain']}"),
        "by_difficulty": group_stats(lambda r: r["difficulty"]),
        "by_length": group_stats(lambda r: r["length_tag"]),
    }


def write_outputs(output_dir: str, results: list[dict[str, Any]], total_selected: int) -> None:
    summary_path = os.path.join(output_dir, "summary.json")
    stats_path = os.path.join(output_dir, "summary_stats.json")
    json.dump(results, open(summary_path, "w"), indent=2, ensure_ascii=False)
    json.dump(make_stats(results, total_selected), open(stats_path, "w"), indent=2, ensure_ascii=False)


def print_selection(tasks: list[dict[str, Any]]) -> None:
    log(f"Selected tasks: {len(tasks)}")
    by_domain = Counter(item["domain"] for item in tasks)
    for domain, count in sorted(by_domain.items()):
        log(f"  {count:3d}  {domain}")
    log("Selected sub-domains:")
    by_sub_domain = Counter((item["domain"], item["sub_domain"]) for item in tasks)
    for (domain, sub_domain), count in sorted(by_sub_domain.items()):
        log(f"  {count:3d}  {domain} / {sub_domain}")


def run_task(args: argparse.Namespace, item: dict[str, Any]) -> dict[str, Any]:
    context = item["context"]
    context_chars = len(context)
    timeout = compute_timeout(context_chars, args)

    log(f"  domain     : {item['domain']} / {item['sub_domain']}")
    log(f"  difficulty : {item['difficulty']} | length: {item['length']}")
    log(f"  context    : {context_chars:,} chars (~{context_chars // 4:,} tokens est.)")
    log(f"  timeout    : {timeout:.0f}s")
    log(f"  question   : {item['question'][:160].replace(chr(10), ' ')}...")
    log(f"  answer key : {item['answer']}")

    logger = RLMLogger(
        log_dir=os.path.join(args.output_dir, "trajectories"),
        file_name=f"rlm_{item['_id']}",
    )

    rlm_instance = RLM(
        backend="vllm",
        backend_kwargs={
            "base_url": args.vllm_url,
            "model_name": args.model_name,
            "api_key": args.api_key,
        },
        environment="local",
        max_iterations=args.max_iterations,
        max_timeout=timeout,
        max_depth=args.max_depth,
        max_concurrent_subcalls=args.max_concurrent_subcalls,
        compaction=args.compaction,
        compaction_threshold_pct=args.compaction_threshold_pct,
        logger=logger,
        verbose=args.verbose,
    )

    t0 = time.time()
    error_msg = None
    response_text = None
    try:
        result = rlm_instance.completion(prompt=context, root_prompt=root_prompt(item))
        response_text = result.response
    except Exception as exc:
        error_msg = str(exc)
        log(f"  ERROR: {error_msg[:500]}")

    elapsed = time.time() - t0
    predicted = extract_prediction(response_text or "")
    correct = (predicted == item["answer"].upper()) if predicted else False
    log(f"  predicted={predicted}  expected={item['answer']}  correct={correct}  elapsed={elapsed:.0f}s")

    return {
        "framework": "rlm",
        "method": "rlm_recursive",
        "task_id": item["_id"],
        "domain": item["domain"],
        "sub_domain": item["sub_domain"],
        "difficulty": item["difficulty"],
        "length_tag": item["length"],
        "context_chars": context_chars,
        "max_iterations": args.max_iterations,
        "max_depth": args.max_depth,
        "max_concurrent_subcalls": args.max_concurrent_subcalls,
        "compaction": args.compaction,
        "compaction_threshold_pct": args.compaction_threshold_pct,
        "max_timeout": timeout,
        "question": item["question"],
        "expected_answer": item["answer"],
        "predicted_answer": predicted,
        "response": response_text,
        "correct": correct,
        "error": error_msg,
        "elapsed_seconds": round(elapsed, 2),
        "log_file": logger.log_file_path,
    }


def main() -> None:
    args = parse_args()
    if not args.data_path:
        raise SystemExit(
            "--data-path is required (or set LONGBENCH_V2_DATA_PATH). Download with: "
            "huggingface-cli download THUDM/LongBench-v2 --repo-type dataset"
        )
    if not args.model_name:
        raise SystemExit("--model-name is required (or set RLM_MODEL_NAME).")

    data = json.load(open(args.data_path))
    tasks = selected_tasks(data, args)
    print_selection(tasks)

    if args.dry_run:
        return

    os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "trajectories"), exist_ok=True)

    log("=" * 80)
    log("RLM LongBench-v2 (full benchmark)")
    log(f"model              : {args.model_name}")
    log(f"vLLM URL           : {args.vllm_url}")
    log(f"output dir         : {args.output_dir}")
    log(f"max iterations     : {args.max_iterations}")
    log(f"max depth          : {args.max_depth}")
    log(f"max concurrent subs: {args.max_concurrent_subcalls}")
    log(f"compaction         : {args.compaction} (threshold {args.compaction_threshold_pct})")
    log(f"timeout tiers      : <=1M={args.timeout_base:.0f}s, <=3M={args.timeout_mid:.0f}s, >3M={args.timeout_high:.0f}s")
    log("=" * 80)

    results: list[dict[str, Any]] = []

    for index, item in enumerate(tasks):
        task_id = item["_id"]
        result_path = os.path.join(args.output_dir, "results", f"{task_id}.json")
        should_skip, existing = result_should_skip(result_path, args)
        if should_skip and existing is not None:
            results.append(existing)
            log(f"[{index + 1:3d}/{len(tasks)}] SKIP {task_id}  "
                f"correct={existing.get('correct')}  error={bool(existing.get('error'))}")
            continue

        log("")
        log(f"[{index + 1:3d}/{len(tasks)}] {task_id}")
        task_result = run_task(args, item)
        json.dump(task_result, open(result_path, "w"), indent=2, ensure_ascii=False)
        results.append(task_result)
        write_outputs(args.output_dir, results, len(tasks))

        done = [r for r in results if not r.get("error")]
        if done:
            n_correct = sum(1 for r in done if r.get("correct"))
            log(f"  Running acc: {n_correct}/{len(done)} = {n_correct / len(done):.1%}")

    log("")
    log("=" * 80)
    log("FINAL SUMMARY - RLM LongBench-v2 (full benchmark)")
    log("=" * 80)
    stats = make_stats(results, len(tasks))
    log(f"Total selected : {stats['total_selected']}")
    log(f"Recorded       : {stats['results_recorded']}")
    log(f"Done           : {stats['done']} | Errors: {stats['errors']}")
    if stats["done"]:
        log(f"Accuracy       : {stats['correct']}/{stats['done']} = {stats['accuracy']:.1%}")
    for difficulty, group in sorted(stats["by_difficulty"].items()):
        if group["accuracy"] is not None:
            log(f"  {difficulty:<10}: {group['correct']}/{group['done']} = {group['accuracy']:.1%}")
    log(f"Summary        : {os.path.join(args.output_dir, 'summary.json')}")
    log(f"Stats          : {os.path.join(args.output_dir, 'summary_stats.json')}")
    log("=" * 80)


if __name__ == "__main__":
    main()

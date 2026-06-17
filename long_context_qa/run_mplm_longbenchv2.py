import argparse
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from openai import OpenAI


# Override any of these via CLI flags or environment variables.
DEFAULT_DATA_PATH = None  # required: path to LongBench-v2 data.json (or set LONGBENCH_V2_DATA_PATH)
DEFAULT_MODEL_NAME = None  # required: served model name / path (or set MPLM_MODEL_NAME)
DEFAULT_VLLM_URL = "http://127.0.0.1:8000/v1"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_OUTPUT_DIR = "./outputs/longbench_v2_mplm"

Messages = list[dict[str, str]]


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the multi-round MPLM LongBench-v2 evaluation."
    )
    parser.add_argument("--data-path", default=os.environ.get("LONGBENCH_V2_DATA_PATH", DEFAULT_DATA_PATH))
    parser.add_argument("--model-name", default=os.environ.get("MPLM_MODEL_NAME", DEFAULT_MODEL_NAME))
    parser.add_argument("--vllm-url", default=os.environ.get("VLLM_URL", DEFAULT_VLLM_URL))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", DEFAULT_API_KEY))
    parser.add_argument("--output-dir", default=os.environ.get("MULTIROUND_EVAL_DIR", DEFAULT_OUTPUT_DIR))
    parser.add_argument("--query-rounds", type=int, default=int(os.environ.get("QUERY_ROUNDS", "1")),
                        help="Number of master->agent query rounds before the final answer.")
    parser.add_argument("--tokens-per-chunk", type=int, default=int(os.environ.get("TOKENS_PER_CHUNK", "10000")))
    parser.add_argument("--chars-per-token", type=int, default=int(os.environ.get("CHARS_PER_TOKEN", "4")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SUMMARY_WORKERS", "4")))
    parser.add_argument("--summary-total-budget", type=int, default=int(os.environ.get("SUMMARY_TOTAL_BUDGET", "96000")))
    parser.add_argument("--min-summary-tokens", type=int, default=int(os.environ.get("MIN_SUMMARY_TOKENS", "192")))
    parser.add_argument("--max-summary-tokens", type=int, default=int(os.environ.get("MAX_SUMMARY_TOKENS", "2048")))
    parser.add_argument("--agent-answer-max-tokens", type=int, default=int(os.environ.get("AGENT_ANSWER_MAX_TOKENS", "1024")))
    parser.add_argument("--master-query-max-tokens", type=int, default=int(os.environ.get("MASTER_QUERY_MAX_TOKENS", "2048")))
    parser.add_argument("--master-max-tokens", type=int, default=int(os.environ.get("MASTER_MAX_TOKENS", "256")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0.6")))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("TOP_P", "0.95")))
    parser.add_argument("--request-timeout", type=float, default=float(os.environ.get("REQUEST_TIMEOUT", "1800")))
    parser.add_argument("--summary-enable-thinking", action=argparse.BooleanOptionalAction,
                        default=env_bool("SUMMARY_ENABLE_THINKING", False))
    parser.add_argument("--master-enable-thinking", action=argparse.BooleanOptionalAction,
                        default=env_bool("MASTER_ENABLE_THINKING", False))
    parser.add_argument("--domain", action="append", help="Optional domain filter. May be passed more than once.")
    parser.add_argument("--sub-domain", action="append", help="Optional sub-domain filter. May be passed more than once.")
    parser.add_argument("--task-id", action="append", help="Optional task id filter. May be passed more than once.")
    parser.add_argument("--limit", type=int, help="Optional cap after filters and sorting.")
    parser.add_argument("--retry-errors", action="store_true", help="Rerun existing result files whose error field is set.")
    parser.add_argument("--overwrite", action="store_true", help="Rerun tasks even if a result file already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected task counts without calling vLLM.")
    return parser.parse_args()


def log(message: str = "") -> None:
    print(message, flush=True)


def chat(
    client: OpenAI,
    model_name: str,
    messages: Messages,
    max_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    return response.choices[0].message.content or ""


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def summary_max_tokens(n_chunks: int, args: argparse.Namespace) -> int:
    per_chunk_budget = max(1, args.summary_total_budget // max(1, n_chunks))
    return min(args.max_summary_tokens, max(args.min_summary_tokens, per_chunk_budget))


def chunk_context(context: str, chars_per_chunk: int) -> list[str]:
    n_chunks = max(1, math.ceil(len(context) / chars_per_chunk))
    chunk_size = math.ceil(len(context) / n_chunks)
    return [context[i * chunk_size:(i + 1) * chunk_size] for i in range(n_chunks)]


def run_child_summarize(
    client: OpenAI,
    args: argparse.Namespace,
    rank: int,
    n_chunks: int,
    chunk_text: str,
    max_tokens: int,
) -> tuple[str, Messages]:
    messages: Messages = [
        {
            "role": "system",
            "content": (
                f"You are reader agent {rank} of {n_chunks - 1} (0-indexed). "
                "You will be given one portion of a long benchmark context. "
                "The question is intentionally hidden from you. "
                "Summarize facts, entities, numbers, dates, causal links, code APIs, "
                "table fields, dialogue states, and other details that may help answer "
                f"a later multiple-choice question. Keep the summary under about {max_tokens} tokens. "
                "Output only the summary."
            ),
        },
        {"role": "user", "content": chunk_text},
    ]
    summary = strip_thinking(chat(
        client=client,
        model_name=args.model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=args.summary_enable_thinking,
    ))
    messages.append({"role": "assistant", "content": summary})
    return summary, messages


def run_child_query(
    client: OpenAI,
    args: argparse.Namespace,
    messages: Messages,
    question: str,
) -> tuple[str, Messages]:
    new_messages = messages + [{"role": "user", "content": question}]
    answer = strip_thinking(chat(
        client=client,
        model_name=args.model_name,
        messages=new_messages,
        max_tokens=args.agent_answer_max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=args.summary_enable_thinking,
    ))
    return answer, new_messages + [{"role": "assistant", "content": answer}]


_SEND_RE = re.compile(r"<send>(.*?)</send>", re.DOTALL)
_AGENT_RE = re.compile(r"\[(\d+)\]:\s*(.*?)(?=\n\s*\[\d+\]:|$)", re.DOTALL)


def parse_send(text: str) -> dict[int, str]:
    match = _SEND_RE.search(text)
    if not match:
        return {}
    return {
        int(m.group(1)): m.group(2).strip()
        for m in _AGENT_RE.finditer(match.group(1))
    }


def build_master_system(n_chunks: int, query_rounds: int) -> str:
    agent_ids = ", ".join(str(i) for i in range(n_chunks))
    return (
        "You are the master answerer for a LongBench-v2 multiple-choice task. "
        f"The context was split into {n_chunks} chunks read by reader agents [{agent_ids}] (0-indexed). "
        "You have received a summary from each agent. Each agent still holds its full chunk "
        "and can answer detailed questions about it. "
        f"You have {query_rounds} query round(s) before you must answer. "
        "In a query round, respond with ONLY a <send> block:\n"
        "  <send>\n"
        "  [id]: your question\n"
        "  [id]: your question\n"
        "  </send>\n"
        "You will be told explicitly when it is time to give your final answer. "
        "Until then you MUST respond with a <send> block, never a final answer. "
        'When asked for the final answer, output exactly one line: "The correct answer is (X)" '
        "where X is A, B, C, or D."
    )


def run_master_turn(
    client: OpenAI,
    args: argparse.Namespace,
    messages: Messages,
    user_content: str,
    max_tokens: int,
) -> tuple[str, Messages]:
    new_messages = messages + [{"role": "user", "content": user_content}]
    response = strip_thinking(chat(
        client=client,
        model_name=args.model_name,
        messages=new_messages,
        max_tokens=max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=args.master_enable_thinking,
    ))
    return response, new_messages + [{"role": "assistant", "content": response}]


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


def make_stats(results: list[dict[str, Any]], total_selected: int, query_rounds: int) -> dict[str, Any]:
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
        "framework": "mplm_multiround",
        "method": f"chunk_summarize_then_{query_rounds}_query_rounds_then_answer",
        "query_rounds": query_rounds,
        "total_selected": total_selected,
        "results_recorded": len(results),
        "done": len(done),
        "errors": len(errors),
        "correct": n_correct,
        "accuracy": n_correct / len(done) if done else None,
        "avg_query_rounds_taken": (
            sum(r.get("query_rounds_taken", 0) for r in done) / len(done) if done else None
        ),
        "by_domain": group_stats(lambda r: r["domain"]),
        "by_sub_domain": group_stats(lambda r: f"{r['domain']} / {r['sub_domain']}"),
        "by_difficulty": group_stats(lambda r: r["difficulty"]),
        "by_length": group_stats(lambda r: r["length_tag"]),
    }


def write_outputs(output_dir: str, results: list[dict[str, Any]], total_selected: int, query_rounds: int) -> None:
    summary_path = os.path.join(output_dir, "summary.json")
    stats_path = os.path.join(output_dir, "summary_stats.json")
    json.dump(results, open(summary_path, "w"), indent=2, ensure_ascii=False)
    json.dump(make_stats(results, total_selected, query_rounds), open(stats_path, "w"), indent=2, ensure_ascii=False)


def print_selection(tasks: list[dict[str, Any]]) -> None:
    log(f"Selected tasks: {len(tasks)}")
    by_domain = Counter(item["domain"] for item in tasks)
    for domain, count in sorted(by_domain.items()):
        log(f"  {count:3d}  {domain}")
    log("Selected sub-domains:")
    by_sub_domain = Counter((item["domain"], item["sub_domain"]) for item in tasks)
    for (domain, sub_domain), count in sorted(by_sub_domain.items()):
        log(f"  {count:3d}  {domain} / {sub_domain}")


def run_task(
    client: OpenAI,
    args: argparse.Namespace,
    item: dict[str, Any],
    chars_per_chunk: int,
) -> dict[str, Any]:
    context = item["context"]
    chunks = chunk_context(context, chars_per_chunk)
    n_chunks = len(chunks)
    per_summary_tokens = summary_max_tokens(n_chunks, args)

    log(f"  domain     : {item['domain']} / {item['sub_domain']}")
    log(f"  difficulty : {item['difficulty']} | length: {item['length']}")
    log(f"  context    : {len(context):,} chars (~{len(context) // args.chars_per_token:,} tokens est.)")
    log(f"  chunks     : {n_chunks} | summary max tokens/chunk: {per_summary_tokens}")
    log(f"  question   : {item['question'][:160].replace(chr(10), ' ')}...")
    log(f"  answer key : {item['answer']}")

    t0 = time.time()
    error_msg = None
    response_text = None
    summaries: list[str] = [""] * n_chunks
    agent_states: dict[int, Messages] = {}
    master_messages: Messages = []
    query_log: list[dict[str, Any]] = []
    query_rounds_taken = 0
    early_answer = False

    try:
        log(f"  [phase 1] summarising {n_chunks} chunks with {min(args.workers, n_chunks)} workers...")
        with ThreadPoolExecutor(max_workers=min(args.workers, n_chunks)) as pool:
            futures = {
                pool.submit(run_child_summarize, client, args, rank, n_chunks, chunks[rank], per_summary_tokens): rank
                for rank in range(n_chunks)
            }
            for future in as_completed(futures):
                rank = futures[future]
                summary, messages = future.result()
                summaries[rank] = summary
                agent_states[rank] = messages
                preview = summary[:100].replace("\n", " ")
                log(f"    agent[{rank}] summary: {preview}...")

        summary_block = "\n\n".join(
            f"=== Agent {rank} Summary ===\n{summaries[rank]}" for rank in range(n_chunks)
        )
        master_messages = [
            {"role": "system", "content": build_master_system(n_chunks, args.query_rounds)},
        ]
        master_user = (
            f"{summary_block}\n\n"
            "The question is:\n\n"
            f"{root_prompt(item)}\n\n"
            f"This is query round 1 of {args.query_rounds}. "
            "Respond with ONLY a <send> block to query specific agents. Do NOT give a final answer yet."
        )

        master_response = ""
        for query_round in range(args.query_rounds):
            master_response, master_messages = run_master_turn(
                client, args, master_messages, master_user, args.master_query_max_tokens,
            )
            log(f"  [query round {query_round + 1}] master: {master_response[:160].replace(chr(10), ' ')}...")

            queries = parse_send(master_response)
            valid_queries = {rank: q for rank, q in queries.items() if rank in agent_states}
            if not valid_queries:
                log(f"  [query round {query_round + 1}] master produced no valid <send> block; forcing final answer.")
                early_answer = True
                break

            query_rounds_taken = query_round + 1
            log(f"  [query round {query_round + 1}] querying agents: {sorted(valid_queries)}")
            agent_responses: dict[int, str] = {}
            with ThreadPoolExecutor(max_workers=min(args.workers, len(valid_queries))) as pool:
                futures = {
                    pool.submit(run_child_query, client, args, agent_states[rank], q): rank
                    for rank, q in valid_queries.items()
                }
                for future in as_completed(futures):
                    rank = futures[future]
                    answer, updated_messages = future.result()
                    agent_responses[rank] = answer
                    agent_states[rank] = updated_messages
                    preview = answer[:100].replace("\n", " ")
                    log(f"    agent[{rank}] reply: {preview}...")

            query_log.append({
                "round": query_round + 1,
                "queries": {str(rank): q for rank, q in valid_queries.items()},
                "responses": {str(rank): agent_responses[rank] for rank in sorted(agent_responses)},
            })

            response_block = "\n\n".join(
                f"=== Agent {rank} Response ===\n{agent_responses[rank]}"
                for rank in sorted(agent_responses)
            )
            if query_round + 1 < args.query_rounds:
                master_user = (
                    f"{response_block}\n\n"
                    f"This is query round {query_round + 2} of {args.query_rounds}. "
                    "Respond with ONLY a <send> block to query specific agents. Do NOT give a final answer yet."
                )
            else:
                master_user = (
                    f"{response_block}\n\n"
                    "All query rounds are complete. Now give your final answer. "
                    "Do not output a <send> block. Do not explain. "
                    'Your entire response must be exactly one line: "The correct answer is (X)".'
                )

        if early_answer:
            master_user = (
                "You did not include a valid <send> block. "
                "Now give your final answer directly. Do not output a <send> block. Do not explain. "
                'Your entire response must be exactly one line: "The correct answer is (X)".'
            )
        log("  [master] answering...")
        response_text, master_messages = run_master_turn(
            client, args, master_messages, master_user, args.master_max_tokens,
        )
        log(f"  [master] {response_text[:160].replace(chr(10), ' ')}...")

    except Exception as exc:
        error_msg = str(exc)
        log(f"  ERROR: {error_msg[:500]}")

    elapsed = time.time() - t0
    predicted = extract_prediction(response_text or "")
    correct = (predicted == item["answer"].upper()) if predicted else False
    log(f"  predicted={predicted}  expected={item['answer']}  correct={correct}  "
        f"query_rounds={query_rounds_taken}  elapsed={elapsed:.0f}s")

    return {
        "framework": "mplm_multiround",
        "method": f"chunk_summarize_then_{args.query_rounds}_query_rounds_then_answer",
        "task_id": item["_id"],
        "domain": item["domain"],
        "sub_domain": item["sub_domain"],
        "difficulty": item["difficulty"],
        "length_tag": item["length"],
        "context_chars": len(context),
        "n_chunks": n_chunks,
        "tokens_per_chunk": args.tokens_per_chunk,
        "summary_max_tokens": per_summary_tokens,
        "summary_total_budget": args.summary_total_budget,
        "summary_enable_thinking": args.summary_enable_thinking,
        "master_enable_thinking": args.master_enable_thinking,
        "query_rounds_allowed": args.query_rounds,
        "query_rounds_taken": query_rounds_taken,
        "total_qa_rounds": query_rounds_taken + 1,
        "question": item["question"],
        "expected_answer": item["answer"],
        "predicted_answer": predicted,
        "response": response_text,
        "correct": correct,
        "error": error_msg,
        "elapsed_seconds": round(elapsed, 2),
        "summaries": summaries,
        "query_log": query_log,
        "master_messages": master_messages,
    }


def main() -> None:
    args = parse_args()
    if not args.data_path:
        raise SystemExit(
            "--data-path is required (or set LONGBENCH_V2_DATA_PATH). Download with: "
            "huggingface-cli download THUDM/LongBench-v2 --repo-type dataset"
        )
    if not args.model_name:
        raise SystemExit("--model-name is required (or set MPLM_MODEL_NAME).")
    chars_per_chunk = args.tokens_per_chunk * args.chars_per_token

    data = json.load(open(args.data_path))
    tasks = selected_tasks(data, args)
    print_selection(tasks)

    if args.dry_run:
        return

    os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)
    client = OpenAI(base_url=args.vllm_url, api_key=args.api_key, timeout=args.request_timeout)

    log("=" * 80)
    log("MPLM multi-round LongBench-v2")
    log(f"model              : {args.model_name}")
    log(f"vLLM URL           : {args.vllm_url}")
    log(f"output dir         : {args.output_dir}")
    log(f"query rounds       : {args.query_rounds} (total QA rounds = {args.query_rounds + 1})")
    log(f"chunk size         : {args.tokens_per_chunk:,} tokens ({chars_per_chunk:,} chars)")
    log(f"summary workers    : {args.workers}")
    log(f"summary budget     : total={args.summary_total_budget:,}, "
        f"min={args.min_summary_tokens}, max={args.max_summary_tokens}")
    log(f"agent answer tokens: {args.agent_answer_max_tokens}")
    log(f"master max tokens  : query={args.master_query_max_tokens}, final={args.master_max_tokens}")
    log(f"thinking           : summaries={args.summary_enable_thinking}, master={args.master_enable_thinking}")
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
        task_result = run_task(client, args, item, chars_per_chunk)
        json.dump(task_result, open(result_path, "w"), indent=2, ensure_ascii=False)
        results.append(task_result)
        write_outputs(args.output_dir, results, len(tasks), args.query_rounds)

        done = [r for r in results if not r.get("error")]
        if done:
            n_correct = sum(1 for r in done if r.get("correct"))
            log(f"  Running acc: {n_correct}/{len(done)} = {n_correct / len(done):.1%}")

    log("")
    log("=" * 80)
    log("FINAL SUMMARY - MPLM multi-round LongBench-v2")
    log("=" * 80)
    stats = make_stats(results, len(tasks), args.query_rounds)
    log(f"Query rounds   : {args.query_rounds}")
    log(f"Total selected : {stats['total_selected']}")
    log(f"Recorded       : {stats['results_recorded']}")
    log(f"Done           : {stats['done']} | Errors: {stats['errors']}")
    if stats["done"]:
        log(f"Accuracy       : {stats['correct']}/{stats['done']} = {stats['accuracy']:.1%}")
    log(f"Summary        : {os.path.join(args.output_dir, 'summary.json')}")
    log(f"Stats          : {os.path.join(args.output_dir, 'summary_stats.json')}")
    log("=" * 80)


if __name__ == "__main__":
    main()

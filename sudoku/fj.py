import json
import math
import argparse
import random
from typing import Dict, Set, Tuple, List, Any, Optional

# Types
Cell = Tuple[int, int]

MASTER_SYS = "You are a helpful assistant."
WORKER_SYS = "You are a helpful assistant."

MASTER_UPSAMPLE = 1
WORKER_DOWNSAMPLE = 1
REMOVE_WORKERS = False


# Formatting helpers
def format_updates_dict_str(upd: Dict[Cell, Set[int]]) -> str:
    """
    Format {(r,c): {a,b,...}} into a single-line string:
        { (r,c): {v1,v2,...}, (r2,c2): {v3} }
    Keys and values are sorted for determinism.
    """
    items = []
    for (r, c) in sorted(upd.keys()):
        vals = "{" + ",".join(str(v) for v in sorted(upd[(r, c)])) + "}"
        items.append(f"({r},{c}): {vals}")
    return "{ " + ", ".join(items) + " }"


def format_candidate_set(cands: Set[int]) -> str:
    """Format candidate set like {1,2,5} with sorted values."""
    return "{" + ", ".join(str(v) for v in sorted(cands)) + "}"


def _indent_text(s: str, prefix: str) -> str:
    """Indent a multi-line string with the given prefix."""
    return "\n".join(prefix + line for line in s.splitlines())


def _format_grid_inline_rows(grid: List[List[int]], base_indent: str) -> str:
    """
    Render a grid with each row on a single line (compact JSON), preserving outer indentation.
    """
    row_indent = base_indent + "  "
    rows = [row_indent + json.dumps(row, ensure_ascii=False) for row in grid]
    return "[\n" + ",\n".join(rows) + "\n" + base_indent + "]"


# Sudoku / clues helpers
def build_clues_from_grid_as_sets(grid: List[List[int]]) -> Dict[Cell, Set[int]]:
    """Extract all known (non-zero) values as clues, as {(r,c): {v}}."""
    N = len(grid)
    clues: Dict[Cell, Set[int]] = {}
    for r in range(N):
        for c in range(N):
            v = int(grid[r][c])
            if v != 0:
                clues[(r, c)] = {v}
    return clues


def build_clues_only_from_puzzle_as_sets(puzzle: List[List[int]]) -> Dict[Cell, Set[int]]:
    """Only given cells from the original puzzle, as {(r,c): {v}}."""
    N = len(puzzle)
    clues: Dict[Cell, Set[int]] = {}
    for r in range(N):
        for c in range(N):
            v = int(puzzle[r][c])
            if v != 0:
                clues[(r, c)] = {v}
    return clues


def cell_candidates_from_grid(grid: List[List[int]], r: int, c: int, N: int, b: int) -> Set[int]:
    """
    Remaining candidates for cell (r,c) given the current global grid.
    If already solved (grid[r][c] != 0), returns singleton {value}.
    """
    v0 = int(grid[r][c])
    if v0 != 0:
        return {v0}

    domain = set(range(1, N + 1))

    # Row exclusions
    for j in range(N):
        v = int(grid[r][j])
        if v != 0:
            domain.discard(v)

    # Column exclusions
    for i in range(N):
        v = int(grid[i][c])
        if v != 0:
            domain.discard(v)

    # Block exclusions
    br0 = (r // b) * b
    bc0 = (c // b) * b
    for i in range(br0, br0 + b):
        for j in range(bc0, bc0 + b):
            v = int(grid[i][j])
            if v != 0:
                domain.discard(v)

    return domain


# Tag helpers
def build_send_to_master(wid: int, value: Optional[int]) -> Tuple[str, str]:
    """
    Build:
      <send [0]>{ From wid: value_or_None }</send><stop>

    Returns (send_text, payload_text).
    payload_text is:
      { From wid: ... }
    """
    payload = f"{{ From {wid}: {value if value is not None else 'None'} }}"
    send_text = f"<send [0]>{payload}</send><stop>"
    return send_text, payload


def build_master_spawn(ids: List[int], clues_body: str) -> str:
    """
    Build master response for a fork:
      <spawn [ids]>clues_body</spawn>
      <recv [ids]>
    """
    ids_str = ",".join(str(i) for i in ids)
    return f"<spawn [{ids_str}]>{clues_body}</spawn>\n<recv [{ids_str}]>"


def build_master_recv_block(payloads: List[Tuple[int, str]]) -> str:
    """
    Close the previously opened <recv ...> by stitching payloads in wid ascending order:
      { From 1: ... }
      { From 2: ... }
      ...
      </recv>
    """
    payloads_sorted = sorted(payloads, key=lambda x: x[0])
    body = "\n".join(p for _, p in payloads_sorted)
    if body:
        return f"{body}\n</recv>"
    return "</recv>"


# Core per-puzzle process (FJ)
def process_one_puzzle(
    idx: int,
    puzzle: List[List[int]],
    solution_board: List[List[int]],
) -> Tuple[str, str, List[Dict[str, str]], Dict[str, List[Dict[str, str]]], bool, int, int]:
    """
    Fork/Join
    - Each round, master spawns workers for remaining unsolved cells.
    - Worker prompt: "Your id is: wid.\n{ (r,c): {v}, ... }"
    - Worker response: MPI-like narrative + <send ...><stop>
    - Master prompt accumulates: prompt + response + recv blocks.
    """
    N = len(puzzle)
    b = int(math.isqrt(N))
    assert b * b == N, "original must be N x N with N a perfect square."

    master_system_prompt = MASTER_SYS
    worker_system_prompt = WORKER_SYS

    # Global grid state
    grid = [row[:] for row in puzzle]

    # Worker traces
    worker_steps: Dict[str, List[Dict[str, str]]] = {str(wid): [] for wid in range(1, N * N + 1)}

    # Master trace
    master_steps: List[Dict[str, str]] = []

    clues_only_map = build_clues_only_from_puzzle_as_sets(puzzle)
    clues_only_body = format_updates_dict_str(clues_only_map)
    master_history_parts: List[str] = []
    master_history_parts.append("Solve this Sudoku:\n" + clues_only_body)

    max_rounds = N * N * 4

    for round_idx in range(max_rounds + 1):
        # Spawn only unsolved cells
        spawn_ids: List[int] = []
        for r in range(N):
            for c in range(N):
                if int(grid[r][c]) == 0:
                    spawn_ids.append(r * N + c + 1)

        if not spawn_ids:
            break

        clues_all_map = build_clues_from_grid_as_sets(grid)
        clues_body = format_updates_dict_str(clues_all_map)

        master_prompt = "\n".join(p for p in master_history_parts if p).strip()
        master_resp = build_master_spawn(spawn_ids, clues_body)
        master_steps.append({"prompt": master_prompt, "response": master_resp})

        # Fork: stateless workers
        payloads_for_master: List[Tuple[int, str]] = []
        newly_solved: List[Tuple[int, int, int]] = []

        for wid in spawn_ids:
            r = (wid - 1) // N
            c = (wid - 1) % N
            wid_str = str(wid)

            prompt = f"Your id is: {wid}.\n{clues_body}"

            cands = cell_candidates_from_grid(grid, r, c, N, b)

            value_to_send: Optional[int] = None
            if len(cands) == 1:
                v = next(iter(cands))
                value_to_send = int(v)
                if int(grid[r][c]) == 0:
                    newly_solved.append((r, c, int(v)))

            send_text, payload = build_send_to_master(wid, value_to_send)

            all_possibilities = set(range(1, N + 1))
            filled_possibilities = all_possibilities - cands

            response = f"My id is {wid} which corresponds to cell ({r},{c}).\n"
            response += f"The neighboring cells already have the following values: {format_candidate_set(filled_possibilities)}.\n"
            response += f"The remaining candidates are: {format_candidate_set(cands)}.\n" + send_text

            worker_steps[wid_str].append({"prompt": prompt, "response": response})
            payloads_for_master.append((wid, payload))

        # Join
        recv_block = build_master_recv_block(payloads_for_master)

        master_history_parts.append(master_resp)
        master_history_parts.append(recv_block)

        # Apply updates
        for (r, c, v) in newly_solved:
            grid[r][c] = v

        # Termination checks
        all_solved = all(int(grid[r][c]) != 0 for r in range(N) for c in range(N))
        if all_solved:
            break
        if not newly_solved:
            break

    # Final "<stop>\n" + str(grid) + "\n</stop>"
    final_prompt = "\n".join(p for p in master_history_parts if p).strip()
    final_resp = "<stop>\n" + str(grid) + "\n</stop>"
    master_steps.append({"prompt": final_prompt, "response": final_resp})

    # Verify
    success = all(int(grid[r][c]) != 0 for r in range(N) for c in range(N))
    solved_ok = success and (grid == solution_board)

    if REMOVE_WORKERS:
        for wid in worker_steps:
            worker_steps[wid] = {}

    keys = list(worker_steps.keys())
    random.shuffle(keys)

    if WORKER_DOWNSAMPLE < 1:
        worker_downsample = 1
    else:
        worker_downsample = WORKER_DOWNSAMPLE

    to_delete = keys[len(worker_steps) // worker_downsample:]
    for k in to_delete:
        del worker_steps[k]

    if MASTER_UPSAMPLE < 1:
        master_upsample = 1
    else:
        master_upsample = MASTER_UPSAMPLE

    master_steps = master_steps * master_upsample
    total_master_steps = len(master_steps)
    total_worker_steps = sum(len(worker_steps[wid]) for wid in worker_steps)
    print(
        f"Master steps: {total_master_steps}, Worker steps: {total_worker_steps}",
        "Master upsample:", master_upsample,
        "Worker downsample:", worker_downsample
    )

    return (
        master_system_prompt,
        worker_system_prompt,
        master_steps,
        worker_steps,
        solved_ok,
        N,
        b,
    )


def dump_with_compact_grids(records: List[Dict[str, Any]], fp) -> None:
    fp.write("[\n")
    for idx, rec in enumerate(records):
        base = "  "
        inner = base + "  "
        fp.write(base + "{\n")
        fp.write(inner + f'"sudoku_id": {json.dumps(rec["sudoku_id"])},\n')
        fp.write(inner + f'"N": {json.dumps(rec["N"])},\n')
        fp.write(inner + f'"base_n": {json.dumps(rec["base_n"])},\n')
        fp.write(inner + f'"master_system_prompt": {json.dumps(rec["master_system_prompt"], ensure_ascii=False)},\n')
        fp.write(inner + f'"worker_system_prompt": {json.dumps(rec["worker_system_prompt"], ensure_ascii=False)},\n')

        fp.write(inner + '"original": ')
        fp.write(_format_grid_inline_rows(rec["original"], inner))
        fp.write(",\n")

        fp.write(inner + '"solution": ')
        fp.write(_format_grid_inline_rows(rec["solution"], inner))
        fp.write(",\n")

        master_json = json.dumps(rec["master"], ensure_ascii=False, indent=2)
        fp.write(inner + '"master": ' + _indent_text(master_json, inner).lstrip() + ",\n")

        workers_json = json.dumps(rec["workers"], ensure_ascii=False, indent=2)
        fp.write(inner + '"workers": ' + _indent_text(workers_json, inner).lstrip() + "\n")

        fp.write(base + "}")
        fp.write(",\n" if idx < len(records) - 1 else "\n")
    fp.write("]\n")


def generate_dataset(dataset: List[Dict[str, Any]], output_filename: str, max_puzzles: int = 5) -> None:
    all_records: List[Dict[str, Any]] = []
    processed = 0
    solved = 0

    for idx, item in enumerate(dataset):
        if processed >= max_puzzles:
            break

        puzzle = item["original"]
        solution = item["solution"]

        (master_sys,
         worker_sys,
         master_steps,
         worker_steps,
         ok,
         N,
         b) = process_one_puzzle(idx, puzzle, solution)

        print(f"\nProcessing puzzle {processed}/{max_puzzles} (index {idx}, N={N}, b={b})...")
        if ok:
            solved += 1
            processed += 1
            print("  -> SUCCESS.")
        else:
            print("  -> NOT FULLY SOLVED; Skipped.")
            continue

        record = {
            "sudoku_id": idx,
            "N": N,
            "base_n": b,
            "master_system_prompt": master_sys,
            "worker_system_prompt": worker_sys,
            "original": puzzle,
            "solution": solution,
            "master": master_steps,
            "workers": worker_steps,
        }
        all_records.append(record)

    with open(output_filename, "w", encoding="utf-8") as f:
        dump_with_compact_grids(all_records, f)

    print("\nGeneration complete!")
    print(f"Processed {processed} puzzles.")
    print(f"Succeeded {solved} puzzles -> {output_filename}")


def main():
    # IMPORTANT: declare globals before any reference in this function
    global MASTER_UPSAMPLE, WORKER_DOWNSAMPLE, REMOVE_WORKERS

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSON (list of {original, solution})"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output JSON path"
    )
    parser.add_argument(
        "--max-puzzles",
        type=int,
        default=1,
        help="Max puzzles to process"
    )
    parser.add_argument("--master-upsample", type=int, default=MASTER_UPSAMPLE)
    parser.add_argument("--worker-downsample", type=int, default=WORKER_DOWNSAMPLE)
    parser.add_argument("--remove-workers", action="store_true", default=REMOVE_WORKERS)
    args = parser.parse_args()

    MASTER_UPSAMPLE = args.master_upsample
    WORKER_DOWNSAMPLE = args.worker_downsample
    REMOVE_WORKERS = args.remove_workers

    with open(args.input, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    generate_dataset(dataset, output_filename=args.out, max_puzzles=args.max_puzzles)


if __name__ == "__main__":
    main()
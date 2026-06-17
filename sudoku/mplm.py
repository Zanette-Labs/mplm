import json
import math
import argparse
import random
from typing import Dict, Set, Tuple, List, Any

# Types
Cell = Tuple[int, int]
CandMap = Dict[Cell, Set[int]]

MASTER_SYS = "You are a helpful assistant."
WORKER_SYS = "You are a helpful assistant."

# Defaults (can be overridden by CLI)
MASTER_UPSAMPLE = 1
WORKER_DOWNSAMPLE = 1
REMOVE_WORKERS = False


# Formatting helpers
def format_updates_dict_str(upd: Dict[Cell, Set[int]]) -> str:
    """
    Format {(r,c): {a,b,...}} into a single-line string:
        { (r,c): {v1,v2}, (r2,c2): {v3} }
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


# Sudoku / worker helpers
def cell_neighbors(r: int, c: int, N: int, b: int) -> List[Cell]:
    """
    All neighbors of cell (r,c): same row, same column, same block.
    The cell itself is excluded.
    """
    neigh: Set[Cell] = set()

    # Row neighbors
    for j in range(N):
        if j != c:
            neigh.add((r, j))

    # Column neighbors
    for i in range(N):
        if i != r:
            neigh.add((i, c))

    # Block neighbors
    br0 = (r // b) * b
    bc0 = (c // b) * b
    for i in range(br0, br0 + b):
        for j in range(bc0, bc0 + b):
            if i == r and j == c:
                continue
            neigh.add((i, j))

    return sorted(neigh)


def init_candidates(puzzle: List[List[int]]) -> CandMap:
    """
    Initialize candidate sets for every cell:
      - Given cell -> singleton {value}
      - Empty cell (0) -> full domain {1..N}
    """
    N = len(puzzle)
    domain = set(range(1, N + 1))
    cand: CandMap = {}
    for r in range(N):
        for c in range(N):
            v = puzzle[r][c]
            if v != 0:
                cand[(r, c)] = {int(v)}
            else:
                cand[(r, c)] = set(domain)
    return cand


def build_grid_from_candidates(cand: CandMap, N: int) -> List[List[int]]:
    """Given a CandMap where all entries are singleton sets, build a 2D grid."""
    grid = [[0] * N for _ in range(N)]
    for (r, c), s in cand.items():
        if len(s) == 1:
            grid[r][c] = next(iter(s))
        else:
            grid[r][c] = 0
    return grid


def build_send_for_cell(
    wid: int,
    r: int,
    c: int,
    v: int,
    neighbors: List[Cell],
    N: int
) -> Tuple[str, str]:
    """
    Build the worker response string for a solved cell:

      <send [0,neighbor_ids]>{ ... }</send><stop>

    and return (response_text, payload_text).

    payload_text is:
      { From wid: v }
    """
    neighbor_ids = [nr * N + nc + 1 for (nr, nc) in neighbors]
    ids_set = set(neighbor_ids)
    ids_set.add(0)
    ids_str = ",".join(str(i) for i in sorted(ids_set))

    payload = f"{{ From {wid}: {v} }}"
    send_text = f"<send [{ids_str}]>{payload}</send><stop>"

    return send_text, payload


def build_recv_block_from_messages(
    msgs: List[Tuple[Cell, int]],
    N: int
) -> str:
    """
    Build a <recv>...</recv> block from a list of (src_cell, value) messages.
    Messages are stitched in ascending src_wid order.
    """
    if not msgs:
        return ""

    msgs_sorted = sorted(
        msgs,
        key=lambda mv: ((mv[0][0] * N + mv[0][1] + 1), mv[1])
    )

    lines: List[str] = []
    for (src_rc, v) in msgs_sorted:
        sr, sc = src_rc
        src_wid = sr * N + sc + 1
        lines.append(f"{{ From {src_wid}: {v} }}")

    inner = "\n".join(lines)
    return inner + "\n</recv>"


# Master helpers
def build_master_init_step(puzzle: List[List[int]]) -> Tuple[List[Dict[str, str]], Dict[Cell, Set[int]]]:
    """
    Build master step 0:
      prompt:  Solve this Sudoku:\n...
      response: <spawn [1..N^2]>{clues-only}</spawn>\n<recv [1..N^2]>

    Returns master_steps list (with one step) and clues-only map.
    """
    N = len(puzzle)
    clues: Dict[Cell, Set[int]] = {}
    for r in range(N):
        for c in range(N):
            v = puzzle[r][c]
            if v != 0:
                clues[(r, c)] = {int(v)}

    ids = ",".join(str(i) for i in range(1, N * N + 1))
    clues_body = format_updates_dict_str(clues)

    master_prompt0 = "Solve this Sudoku:\n" + clues_body
    master_resp0 = f"<spawn [{ids}]>{clues_body}</spawn>\n<recv [{ids}]>"

    master_steps: List[Dict[str, str]] = [
        {"prompt": master_prompt0, "response": master_resp0}
    ]
    return master_steps, clues


def build_master_final_step(
    master_steps: List[Dict[str, str]],
    final_grid: List[List[int]],
    master_recv_payloads: List[Tuple[int, str]]
) -> None:
    """Append a final master step. The stitched payloads are ordered by wid ascending."""
    if not master_steps:
        prev_prompt = ""
        prev_resp = ""
    else:
        last = master_steps[-1]
        prev_prompt = last.get("prompt", "")
        prev_resp = last.get("response", "")

    pieces = [prev_prompt, prev_resp]

    if master_recv_payloads:
        payloads_sorted = sorted(master_recv_payloads, key=lambda x: x[0])
        body = "\n".join(payload for (_, payload) in payloads_sorted)
        recv_block = f"{body}\n</recv>"
        pieces.append(recv_block)

    master_promptF = "\n".join(p for p in pieces if p).strip()
    master_respF = "<stop>" + "\n" + str(final_grid) + "\n" + "</stop>"
    master_steps.append({"prompt": master_promptF, "response": master_respF})


# Core per-puzzle process
def process_one_puzzle(
    idx: int,
    puzzle: List[List[int]],
    solution_board: List[List[int]],
    master_upsample: int = MASTER_UPSAMPLE,
    worker_downsample: int = WORKER_DOWNSAMPLE,
    remove_workers: bool = REMOVE_WORKERS,
) -> Tuple[str, str, List[Dict[str, str]], Dict[str, List[Dict[str, str]]], bool, int, int]:
    """
    Run the cell-worker process for a single Sudoku instance and build:
      master_system_prompt, worker_system_prompt,
      master_steps, worker_steps, solved_ok, N, base_n.
    """
    N = len(puzzle)
    b = int(math.isqrt(N))
    assert b * b == N, "Puzzle must be N x N with N a perfect square."

    master_system_prompt = MASTER_SYS
    worker_system_prompt = WORKER_SYS

    candidates: CandMap = init_candidates(puzzle)
    solved_cells: Dict[Cell, int] = {}
    stopped_cells: Set[Cell] = set()

    message_queues: Dict[Cell, List[Tuple[Cell, int]]] = {
        (r, c): [] for r in range(N) for c in range(N)
    }

    master_steps, clues_only_map = build_master_init_step(puzzle)
    clues_body = format_updates_dict_str(clues_only_map)

    worker_steps: Dict[str, List[Dict[str, str]]] = {}
    last_prompt: Dict[int, str] = {}
    last_response: Dict[int, str] = {}
    last_resp_was_recv: Dict[int, bool] = {}

    master_recv_payloads: List[Tuple[int, str]] = []

    newly_solved: Dict[Cell, int] = {}

    # Round 0
    for r in range(N):
        for c in range(N):
            cell = (r, c)
            wid = r * N + c + 1
            wid_str = str(wid)
            worker_steps.setdefault(wid_str, [])

            prompt0 = f"Your id is: {wid}.\n{clues_body}"

            v = puzzle[r][c]
            if v != 0:
                v = int(v)
                solved_cells[cell] = v
                newly_solved[cell] = v

                neigh = cell_neighbors(r, c, N, b)
                send0, payload = build_send_for_cell(wid, r, c, v, neigh, N)

                resp0 = f"My id is {wid} which corresponds to cell ({r},{c}). "
                resp0 += f"The cell is given with value {v}.\n"
                resp0 += f"The remaining candidates are: {{{v}}}.\n" + send0

                master_recv_payloads.append((wid, payload))
                stopped_cells.add(cell)
                last_resp_was_recv[wid] = False
            else:
                cur_cands = candidates[cell]
                resp0 = f"My id is {wid} which corresponds to cell ({r},{c}). "
                resp0 += "The cell is empty.\n"
                resp0 += f"The remaining candidates are: {format_candidate_set(cur_cands)}.\n<recv>"
                last_resp_was_recv[wid] = True

            worker_steps[wid_str].append({"prompt": prompt0, "response": resp0})
            last_prompt[wid] = prompt0
            last_response[wid] = resp0

    # Broadcast given cells
    for (r, c), v in newly_solved.items():
        cell = (r, c)
        neigh = cell_neighbors(r, c, N, b)
        for (nr, nc) in neigh:
            if (nr, nc) not in solved_cells:
                message_queues[(nr, nc)].append((cell, v))
    newly_solved = {}

    # Subsequent rounds
    max_rounds = N * N * 4
    round_idx = 1

    while round_idx <= max_rounds:
        newly_solved = {}
        progress = False
        cells_to_stop: Set[Cell] = set()

        for r in range(N):
            for c in range(N):
                cell = (r, c)
                if cell in stopped_cells:
                    continue

                inbox = message_queues[cell]
                if not inbox:
                    continue

                wid = r * N + c + 1
                wid_str = str(wid)

                prev_prompt = last_prompt.get(wid, "")
                prev_resp = last_response.get(wid, "")

                if prev_prompt:
                    prompt_parts = [prev_prompt, prev_resp]
                else:
                    prompt_parts = [prev_resp] if prev_resp else []

                prompt = "\n".join(p for p in prompt_parts if p).strip()

                if last_resp_was_recv.get(wid, False):
                    recv_block = build_recv_block_from_messages(inbox, N)
                    prompt = (prompt + "\n" + recv_block) if prompt else recv_block

                if not prompt:
                    prompt = f"Your id is: {wid}.\n{clues_body}"

                cur_cands = candidates[cell]
                before_len = len(cur_cands)

                for (src_rc, v) in inbox:
                    if v in cur_cands:
                        cur_cands.discard(v)
                message_queues[cell] = []
                if len(cur_cands) != before_len:
                    progress = True

                if len(cur_cands) == 1 and cell not in solved_cells:
                    v = next(iter(cur_cands))
                    solved_cells[cell] = v
                    newly_solved[cell] = v

                    neigh = cell_neighbors(r, c, N, b)
                    send_text, payload = build_send_for_cell(wid, r, c, v, neigh, N)

                    all_possibilities = set(range(1, N + 1))
                    filled_possibilities = all_possibilities - set([v])

                    response = (
                        f"The neighboring cells have sent the following values: "
                        f"{format_candidate_set(filled_possibilities)}.\n"
                        f"The remaining candidates are: {{{v}}}.\n" + send_text
                    )

                    master_recv_payloads.append((wid, payload))
                    last_resp_was_recv[wid] = False
                    cells_to_stop.add(cell)
                else:
                    all_possibilities = set(range(1, N + 1))
                    filled_possibilities = all_possibilities - cur_cands
                    response = (
                        f"The neighboring cells have sent the following values: "
                        f"{format_candidate_set(filled_possibilities)}.\n"
                        f"The remaining candidates are: {format_candidate_set(cur_cands)}.\n<recv>"
                    )
                    last_resp_was_recv[wid] = True

                worker_steps[wid_str].append({"prompt": prompt, "response": response})
                last_prompt[wid] = prompt
                last_response[wid] = response

        for cell in cells_to_stop:
            stopped_cells.add(cell)

        for (r, c), v in newly_solved.items():
            cell = (r, c)
            neigh = cell_neighbors(r, c, N, b)
            for (nr, nc) in neigh:
                if (nr, nc) not in solved_cells:
                    message_queues[(nr, nc)].append((cell, v))

        if len(solved_cells) == N * N:
            break
        if not newly_solved and not progress:
            break

        round_idx += 1

    final_grid = build_grid_from_candidates(candidates, N)
    success = all(len(candidates[(r, c)]) == 1 for r in range(N) for c in range(N))
    solved_ok = success and (final_grid == solution_board)

    build_master_final_step(master_steps, final_grid, master_recv_payloads)

    # 1) remove workers (optional)
    if remove_workers:
        for wid in list(worker_steps.keys()):
            worker_steps[wid] = {}

    # 2) downsample workers (keep 1/worker_downsample)
    if worker_downsample is None or worker_downsample < 1:
        worker_downsample = 1

    keys = list(worker_steps.keys())
    random.shuffle(keys)

    keep_n = len(worker_steps) // worker_downsample
    to_delete = keys[keep_n:]
    for k in to_delete:
        del worker_steps[k]

    # 3) upsample master
    if master_upsample is None or master_upsample < 1:
        master_upsample = 1
    master_steps = master_steps * master_upsample

    total_master_steps = len(master_steps)
    total_worker_steps = sum(len(worker_steps[wid]) for wid in worker_steps)
    print(
        f"Master steps: {total_master_steps}, Worker steps: {total_worker_steps}",
        "Master upsample:", master_upsample,
        "Worker downsample:", worker_downsample,
        "Remove workers:", remove_workers,
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
    """Print the dataset list with compact grids and indented master/worker traces."""
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


def generate_dataset(
    dataset: List[Dict[str, Any]],
    output_filename: str,
    max_puzzles: int = 5,
    master_upsample: int = MASTER_UPSAMPLE,
    worker_downsample: int = WORKER_DOWNSAMPLE,
    remove_workers: bool = REMOVE_WORKERS,
) -> None:
    """Run multiple puzzles and dump a dataset file with master/worker traces."""
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
         b) = process_one_puzzle(
            idx, puzzle, solution,
            master_upsample=master_upsample,
            worker_downsample=worker_downsample,
            remove_workers=remove_workers,
        )

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSON array of puzzles, each with 'original' and 'solution' grids.")
    parser.add_argument("--out", type=str, required=True, help="Output JSON path for the generated traces.")
    parser.add_argument("--max-puzzles", type=int, default=1)

    parser.add_argument("--master-upsample", type=int, default=MASTER_UPSAMPLE)
    parser.add_argument("--worker-downsample", type=int, default=WORKER_DOWNSAMPLE)
    parser.add_argument("--remove-workers", action="store_true", default=REMOVE_WORKERS)

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    generate_dataset(
        dataset,
        output_filename=args.out,
        max_puzzles=args.max_puzzles,
        master_upsample=args.master_upsample,
        worker_downsample=args.worker_downsample,
        remove_workers=args.remove_workers,
    )


if __name__ == "__main__":
    main()
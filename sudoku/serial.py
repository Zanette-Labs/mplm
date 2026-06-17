import json
import math
import argparse
from typing import Dict, Set, Tuple, List, Any, Optional
from collections import defaultdict

Cell = Tuple[int, int]
CandMap = Dict[Cell, Set[int]]

MASTER_SYS = "You are a helpful assistant."
WORKER_SYS = "You are a helpful assistant."

def format_updates_dict_str(upd: Dict[Cell, Set[int]]) -> str:
    items = []
    for (r, c) in sorted(upd.keys()):
        vals = "{" + ",".join(str(v) for v in sorted(upd[(r, c)])) + "}"
        items.append(f"({r},{c}): {vals}")
    return "{ " + ", ".join(items) + " }"

def _indent_text(s: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in s.splitlines())

def _format_grid_inline_rows(grid: List[List[int]], base_indent: str) -> str:
    row_indent = base_indent + "  "
    rows = [row_indent + json.dumps(row, ensure_ascii=False) for row in grid]
    return "[\n" + ",\n".join(rows) + "\n" + base_indent + "]"

def dump_with_compact_grids(records: List[Dict[str, Any]], fp) -> None:
    fp.write("[\n")
    for idx, rec in enumerate(records):
        base = "  "; inner = base + "  "
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

def build_clues_only_from_puzzle_as_sets(puzzle: List[List[int]]) -> Dict[Cell, Set[int]]:
    N = len(puzzle)
    clues: Dict[Cell, Set[int]] = {}
    for r in range(N):
        for c in range(N):
            v = int(puzzle[r][c])
            if v != 0:
                clues[(r, c)] = {v}
    return clues

def bootstrap_candidates(puzzle: List[List[int]]) -> CandMap:
    N = len(puzzle)
    domain = set(range(1, N + 1))
    cand: CandMap = {}
    for r in range(N):
        for c in range(N):
            v = puzzle[r][c]
            cand[(r, c)] = {v} if v != 0 else set(domain)
    return cand

def is_solved(cset: Set[int]) -> bool:
    return len(cset) == 1

def to_board(cands: CandMap, N: int) -> List[List[int]]:
    grid = [[0] * N for _ in range(N)]
    for (r, c), s in cands.items():
        grid[r][c] = next(iter(s)) if len(s) == 1 else 0
    return grid

def _format_set(vals: Set[int]) -> str:
    return "{" + ",".join(str(v) for v in sorted(vals)) + "}"

def format_candmap_str(mem: CandMap) -> str:
    items = []
    for (r, c) in sorted(mem.keys()):
        vals = _format_set(mem[(r, c)])
        items.append(f"({r},{c}): {vals}")
    return "{ " + ", ".join(items) + " }"

def basic_update_global(mem: CandMap) -> Tuple[CandMap, Dict[Cell, Set[int]], bool]:
    N = 0
    for (r, _c) in mem.keys():
        N = max(N, r + 1)
    b = int(N ** 0.5)

    updates: CandMap = {}
    neighbor_vals_for_updates: Dict[Cell, Set[int]] = {}
    any_change = False

    row_singletons: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    col_singletons: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    blk_singletons: Dict[Tuple[int, int], List[Tuple[Cell, int]]] = defaultdict(list)

    for (r, c), s in mem.items():
        if is_solved(s):
            v = next(iter(s))
            row_singletons[r].append((c, v))
            col_singletons[c].append((r, v))
            br, bc = r // b, c // b
            blk_singletons[(br, bc)].append(((r, c), v))

    for (r, c), cur in mem.items():
        if is_solved(cur):
            continue

        allowed = set(cur)
        neigh_vals: Set[int] = set()

        for (pc, v) in row_singletons[r]:
            if pc == c:
                continue
            neigh_vals.add(v)
            if v in allowed:
                allowed.discard(v)

        for (pr, v) in col_singletons[c]:
            if pr == r:
                continue
            neigh_vals.add(v)
            if v in allowed:
                allowed.discard(v)

        br, bc = r // b, c // b
        for (src_rc, v) in blk_singletons[(br, bc)]:
            if src_rc == (r, c):
                continue
            neigh_vals.add(v)
            if v in allowed:
                allowed.discard(v)

        if allowed and allowed != cur:
            updates[(r, c)] = allowed
            neighbor_vals_for_updates[(r, c)] = neigh_vals
            any_change = True

    return updates, neighbor_vals_for_updates, any_change

def solve_with_serial_cot(puzzle: List[List[int]]) -> Tuple[List[str], List[List[int]]]:
    N = len(puzzle)
    mem = bootstrap_candidates(puzzle)
    cot_lines: List[str] = []

    round_idx = 1
    while True:
        updates, neighbor_vals_for_updates, changed = basic_update_global(mem)

        cot_lines.append(f"\nRound {round_idx}:")

        for (r, c) in sorted(updates.keys()):
            neigh_vals = neighbor_vals_for_updates.get((r, c), set())
            cot_lines.append(
                f"For cell ({r},{c}), the neighboring cells have the values {_format_set(neigh_vals)}.\n"
                f"The remaining candidates are: {_format_set(updates[(r, c)])}"
            )

        for rc, newset in updates.items():
            mem[rc] = set(newset)

        cot_lines.append(f"The current remaining candidates of the sudoku is :{format_candmap_str(mem)}")

        if not changed:
            break
        if all(is_solved(s) for s in mem.values()):
            break

        round_idx += 1

    final_board = to_board(mem, N)
    return cot_lines, final_board

def generate_serial_dataset_from_existing(
    input_path: str,
    output_filename: str,
    max_puzzles: Optional[int] = None,
    strict_identity: bool = True,
):
    with open(input_path, "r", encoding="utf-8") as f:
        in_records: List[Dict[str, Any]] = json.load(f)
    if not isinstance(in_records, list):
        raise ValueError("Input must be a JSON array of records.")

    out_records: List[Dict[str, Any]] = []
    processed = 0

    for i, rec in enumerate(in_records):
        if (max_puzzles is not None) and (processed >= max_puzzles):
            break
        processed += 1

        if ("original" not in rec and "puzzle" not in rec) or "solution" not in rec:
            raise ValueError("Each record must contain 'original' (or 'puzzle') and 'solution'.")

        sudoku_id = rec.get("sudoku_id", i)
        puzzle = rec.get("original", rec.get("puzzle"))
        solution = rec["solution"]

        N_in = rec.get("N", len(puzzle))
        b_in = rec.get("base_n", int(math.isqrt(len(puzzle))))

        if strict_identity:
            if N_in != len(puzzle):
                print(f"[WARN] sudoku_id={sudoku_id}: input N={N_in} != len(original)={len(puzzle)}; keeping input N.")
            if b_in * b_in != N_in:
                print(f"[WARN] sudoku_id={sudoku_id}: base_n^2 != N; keeping input base_n.")

        clues_only_map = build_clues_only_from_puzzle_as_sets(puzzle)
        prompt_text = "Solve this Sudoku:\n" + format_updates_dict_str(clues_only_map)

        cot_lines, final_board = solve_with_serial_cot(puzzle)
        response_text = "\n".join(cot_lines) + "\n" + "<stop>\n" + str(final_board) + "\n</stop>"

        if final_board == solution:
            print(f"sudoku_id={sudoku_id}: solved.")
            out_records.append(
                {
                    "sudoku_id": sudoku_id,
                    "N": N_in,
                    "base_n": b_in,
                    "master_system_prompt": MASTER_SYS,
                    "worker_system_prompt": WORKER_SYS,
                    "original": puzzle,
                    "solution": solution,
                    "master": [{"prompt": prompt_text, "response": response_text}],
                    "workers": {},
                }
            )
        else:
            print(f"sudoku_id={sudoku_id}: not fully solved; skipped.")
            continue

    with open(output_filename, "w", encoding="utf-8") as f:
        dump_with_compact_grids(out_records, f)

    print("\nGeneration complete!")
    print(f"Processed {processed} puzzles -> {output_filename}")

def main():
    parser = argparse.ArgumentParser(description="Serially solve puzzles and output a dataset with a single prompt/response per puzzle.")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to JSON array with original(or puzzle)/solution; sudoku_id is optional")
    parser.add_argument("--out", type=str, required=True,
                        help="Output JSON path")
    parser.add_argument("--max-puzzles", type=int, default=1,
                        help="Max puzzles to process")
    args = parser.parse_args()

    generate_serial_dataset_from_existing(
        input_path=args.input,
        output_filename=args.out,
        max_puzzles=args.max_puzzles,
        strict_identity=False,
    )

if __name__ == "__main__":
    main()
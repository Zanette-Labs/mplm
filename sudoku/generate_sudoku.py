import argparse
import json
from random import Random
from typing import List, Tuple, Optional, Set, Iterable, Union
import time


class SudokuGenerator:
    def __init__(self, base_n: int, rng: Random, p: float = 1.0):
        assert base_n >= 2, "base_n must be >= 2"
        assert 0.0 <= p <= 1.0, "p must be in [0,1]"
        self.base_n = base_n
        self.N = base_n * base_n
        self.rng = rng
        self.p = p  # probability to remove a removable clue

    def domain(self) -> Set[int]:
        return set(range(1, self.N + 1))

    def _box_anchor(self, r: int, c: int) -> Tuple[int, int]:
        b = self.base_n
        return (b * (r // b), b * (c // b))

    def _get_possible_values(self, board: List[List[int]], r: int, c: int) -> Set[int]:
        if board[r][c] != 0:
            return {board[r][c]}
        used = set()
        used |= set(v for v in board[r] if v != 0)
        used |= {board[i][c] for i in range(self.N) if board[i][c] != 0}
        br, bc = self._box_anchor(r, c)
        for i in range(br, br + self.base_n):
            for j in range(bc, bc + self.base_n):
                v = board[i][j]
                if v != 0:
                    used.add(v)
        return self.domain() - used

    def _naked_single_solve(self, puzzle: List[List[int]]) -> Optional[List[List[int]]]:
        N = self.N
        board = [row[:] for row in puzzle]
        cands = [[self._get_possible_values(board, r, c) for c in range(N)] for r in range(N)]

        progress = True
        while progress:
            progress = False
            for r in range(N):
                for c in range(N):
                    if board[r][c] == 0:
                        s = cands[r][c]
                        if len(s) == 0:
                            return None
                        if len(s) == 1:
                            v = next(iter(s))
                            board[r][c] = v
                            # propagate
                            for j in range(N):
                                if j != c and board[r][j] == 0 and v in cands[r][j]:
                                    cands[r][j].discard(v)
                                    if not cands[r][j]:
                                        return None
                            for i in range(N):
                                if i != r and board[i][c] == 0 and v in cands[i][c]:
                                    cands[i][c].discard(v)
                                    if not cands[i][c]:
                                        return None
                            br, bc = self._box_anchor(r, c)
                            for i in range(br, br + self.base_n):
                                for j in range(bc, bc + self.base_n):
                                    if (i, j) != (r, c) and board[i][j] == 0 and v in cands[i][j]:
                                        cands[i][j].discard(v)
                                        if not cands[i][j]:
                                            return None
                            cands[r][c] = {v}
                            progress = True

        # check solved
        for r in range(N):
            for c in range(N):
                if board[r][c] == 0:
                    return None
        return board

    def _generate_solved_board(self) -> List[List[int]]:
        b, N = self.base_n, self.N

        def pattern(r, c):
            return (b * (r % b) + r // b + c) % N

        def shuffled(seq):
            seq = list(seq)
            self.rng.shuffle(seq)
            return seq

        rows = sum(([(g * b + r) for r in shuffled(range(b))] for g in shuffled(range(b))), [])
        cols = sum(([(g * b + c) for c in shuffled(range(b))] for g in shuffled(range(b))), [])
        nums = shuffled(range(1, N + 1))
        return [[nums[pattern(r, c)] for c in cols] for r in rows]

    def _auto_dig(self, solved: List[List[int]]) -> List[List[int]]:
        """
        Try to remove clues in random order. If removing a clue keeps the puzzle
        naked-single-solvable to the *same* solution, we remove it with probability `self.p`.
        Otherwise we restore it.
        """
        N = self.N
        puzzle = [row[:] for row in solved]
        cells = [(i, j) for i in range(N) for j in range(N)]
        self.rng.shuffle(cells)

        for (i, j) in cells:
            saved = puzzle[i][j]
            if saved == 0:
                continue
            # Tentatively remove
            puzzle[i][j] = 0
            ns = self._naked_single_solve(puzzle)
            if ns is None or ns != solved:
                # Not removable -> restore
                puzzle[i][j] = saved
            else:
                # Removable -> keep removal with probability p
                if self.rng.random() < self.p:
                    # keep as 0 (already removed)
                    pass
                else:
                    # decide not to remove this clue
                    puzzle[i][j] = saved
        return puzzle

    def generate(self, idx: int):
        solved = self._generate_solved_board()
        puzzle = self._auto_dig(solved)
        # Safety checks
        assert self._naked_single_solve(puzzle) == solved, "Internal: puzzle not NS-solvable"
        return {"sudoku_id": idx, "original": puzzle, "solution": solved}

    # ---------- Static helpers for de-duplication ----------
    @staticmethod
    def grid_signature(grid: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
        """
        Convert a 2D list into a hashable canonical signature (tuple of tuples).
        This ensures O(1)-ish membership checks in a Python set and avoids JSON/str overhead.
        """
        return tuple(tuple(row) for row in grid)

    @staticmethod
    def item_signature(item: dict, mode: str = "original") -> Union[Tuple[Tuple[int, ...], ...], Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]]]:
        """
        Build a signature according to the selected de-duplication mode.
        - "original": only the given clues layout must be unique.
        - "solution": only the final solved grid must be unique.
        - "both": (puzzle, solution) pair must be unique.
        """
        if mode == "original":
            return SudokuGenerator.grid_signature(item["original"])
        elif mode == "solution":
            return SudokuGenerator.grid_signature(item["solution"])
        elif mode == "both":
            return (
                SudokuGenerator.grid_signature(item["original"]),
                SudokuGenerator.grid_signature(item["solution"]),
            )
        else:
            raise ValueError(f"Unknown dedup mode: {mode}")


def main():
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-n", type=int, default=4, help="block size; grid is (base_n**2)x(base_n**2)")
    parser.add_argument("--size", type=int, default=100, help="number of UNIQUE puzzles to output")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--out", type=str, required=True, help="output json file")
    parser.add_argument(
        "--dedup-by",
        type=str,
        choices=["original", "solution", "both"],
        default="original",
        help="how to deduplicate: by puzzle (default), by solution, or by both"
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="upper bound on generation attempts (0 => auto set to size*50)"
    )
    parser.add_argument(
        "-p", "--p",
        type=float,
        default=0.5,
        help="probability of deleting a removable clue (0..1). 1.0 reproduces original behavior."
    )
    args = parser.parse_args()

    if not (0.0 <= args.p <= 1.0):
        raise ValueError("--p must be in [0,1]")

    rng = Random(args.seed)
    gen = SudokuGenerator(args.base_n, rng, p=args.p)

    target = args.size
    max_attempts = args.max_attempts if args.max_attempts > 0 else target * 50  # safety to avoid infinite loops

    seen = set()
    items = []
    attempts = 0

    # Keep generating until we collect the requested number of UNIQUE items or hit the attempt limit
    while len(items) < target and attempts < max_attempts:
        item = gen.generate(len(items))
        sig = SudokuGenerator.item_signature(item, mode=args.dedup_by)
        if sig in seen:
            attempts += 1
            continue
        seen.add(sig)
        items.append(item)
        attempts += 1

    if len(items) < target:
        print(
            f"Warning: Requested {target} unique items, but only found {len(items)} "
            f"after {attempts} attempts (dedup-by={args.dedup_by})."
        )

    # Write compact JSON with one object per line, arrays on a single line
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, item in enumerate(items):
            s = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            f.write(s)
            if i != len(items) - 1:
                f.write(",\n")
        f.write("\n]")

    print(f"Saved {len(items)} UNIQUE puzzles to {args.out} (dedup-by={args.dedup_by}, attempts={attempts}, p={args.p})")

    end = time.time()
    print(end-start)


if __name__ == "__main__":
    main()
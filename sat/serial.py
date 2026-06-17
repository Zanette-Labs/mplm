import argparse
import json
import os
from typing import Dict, List, Optional, Tuple, Any


SYSTEM_TEXT = "You are a helpful assistant."


# CNF formatting
def format_cnf(clauses: List[List[int]]) -> str:
    """Pretty CNF string using ∨ and ∧, with ¬ for negation."""
    if len(clauses) == 0:
        return "TRUE"
    return " ∧ ".join(
        f"( {' ∨ '.join(str(lit) if lit > 0 else f'¬ {abs(lit)}' for lit in clause)} )"
        for clause in clauses
    )


# SAT / DPLL utilities
def simplify_cnf(clauses: List[List[int]], var: int, value: bool) -> List[List[int]]:
    sat_lit = var if value else -var
    fals_lit = -var if value else var
    out: List[List[int]] = []
    for c in clauses:
        if sat_lit in c:
            continue
        nc = [lit for lit in c if lit != fals_lit]
        out.append(nc)
    return out


def base_case(clauses: List[List[int]]) -> Optional[bool]:
    if len(clauses) == 0:
        return True
    if any(len(c) == 0 for c in clauses):
        return False
    return None


def find_unit_literal(clauses: List[List[int]]) -> Optional[int]:
    for c in clauses:
        if len(c) == 1:
            return c[0]
    return None


def pick_branch_var(clauses: List[List[int]], assignment: Dict[int, bool]) -> Optional[int]:
    s = set()
    for c in clauses:
        for lit in c:
            s.add(abs(lit))
    for v in sorted(s):
        if v not in assignment:
            return v
    return None


def unit_propagate_all(
    clauses: List[List[int]],
    assignment: Dict[int, bool],
) -> Tuple[List[List[int]], List[Tuple[int, bool]], Optional[bool]]:
    """
    Unit propagate to fixpoint.
    Returns: (new_clauses, delta_assignments, base_case_result)
    """
    delta: List[Tuple[int, bool]] = []

    while True:
        bc = base_case(clauses)
        if bc is not None:
            return clauses, delta, bc

        lit = find_unit_literal(clauses)
        if lit is None:
            return clauses, delta, None

        var = abs(lit)
        val = (lit > 0)

        if var in assignment and assignment[var] != val:
            return [[]], delta, False

        if var not in assignment:
            assignment[var] = val
            delta.append((var, val))

        clauses = simplify_cnf(clauses, var, val)


def format_unit_delta(delta: List[Tuple[int, bool]]) -> str:
    if not delta:
        return ""
    parts = [f"x{v}={val}" for v, val in delta]
    return ", ".join(parts)


# Serial CoT DPLL (modified style)
def serial_dpll_cot(
    clauses: List[List[int]],
    max_depth: int,
    max_trace_lines: int = 0,
) -> Tuple[bool, List[str]]:
    """
    Returns (sat, trace_lines).

    Style requirements (your request):
    - No model anywhere.
    - Branch order: try x=0 (False) first, then x=1 (True).
    - At each branch, ONLY print the updated CNF (after that assignment).
    - If SAT found, do NOT print "branch succeeded"; just return and final output is <stop>SAT.</stop>.
    - For unit propagation with multiple unit assignments, print only the final CNF once.
    """
    trace: List[str] = []

    def add(line: str) -> None:
        trace.append(line)

    def indent(depth: int) -> str:
        return "  " * depth

    def solve(
        cnf: List[List[int]],
        assignment: Dict[int, bool],
        depth: int,
    ) -> bool:
        if depth > max_depth:
            add(f"{indent(depth)}Reached max_depth={max_depth}. Treat as UNSAT for this branch.")
            return False

        # Unit propagation at this node
        local_asg = dict(assignment)
        simplified, unit_delta, bc = unit_propagate_all([c[:] for c in cnf], local_asg)

        # If unit chain happened, print once (final CNF after unit propagation)
        if unit_delta:
            add(f"{indent(depth)}Unit assignments: {format_unit_delta(unit_delta)}")
            add(f"{indent(depth)}CNF:")
            add(f"{indent(depth)}{format_cnf(simplified)}")

        if bc is True:
            add(f"{indent(depth)}Result: SAT.")
            return True
        if bc is False:
            add(f"{indent(depth)}Result: UNSAT.")
            return False

        # Need to branch
        var = pick_branch_var(simplified, local_asg)
        if var is None:
            add(f"{indent(depth)}No variables left. SAT.")
            return True

        add(f"{indent(depth)}Branch on x{var}.")

        # ---- Branch order: x=0 first (False), then x=1 (True) ----
        # Branch x=0 (False)
        cnf_0 = simplify_cnf(simplified, var, False)
        add(f"{indent(depth)}Branch x{var}=0")
        add(f"{indent(depth)}CNF:")
        add(f"{indent(depth)}{format_cnf(cnf_0)}")

        if solve(cnf_0, {**local_asg, var: False}, depth + 1):
            return True  # SAT found; no "succeeded" message

        # Branch x=1 (True)
        cnf_1 = simplify_cnf(simplified, var, True)
        add(f"{indent(depth)}Branch x{var}=1")
        add(f"{indent(depth)}CNF:")
        add(f"{indent(depth)}{format_cnf(cnf_1)}")

        if solve(cnf_1, {**local_asg, var: True}, depth + 1):
            return True

        add(f"{indent(depth)}Both branches failed for x{var}. UNSAT.")
        return False

    sat = solve([c[:] for c in clauses], {}, 0)

    # Optional truncation
    if max_trace_lines and len(trace) > max_trace_lines:
        head_n = max_trace_lines // 2
        tail_n = max_trace_lines - head_n - 1
        trace = trace[:head_n] + ["... (trace truncated) ..."] + trace[-tail_n:]

    return sat, trace


# Dataset builder
def build_serial_cot_record(
    instance_id: Any,
    clauses: List[List[int]],
    max_depth: int,
    max_trace_lines: int,
) -> Dict[str, Any]:
    cnf_str = format_cnf(clauses)
    sat, trace_lines = serial_dpll_cot(
        clauses=clauses,
        max_depth=max_depth,
        max_trace_lines=max_trace_lines,
    )

    final_msg = "SAT." if sat else "UNSAT."
    response = "\n".join(trace_lines) + "\n" + f"<stop>{final_msg}</stop>"

    return {
        "instance_id": instance_id,
        "system": SYSTEM_TEXT,
        "prompt": cnf_str,
        "response": response,
    }


# CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True,
                    help="Input JSONL with `clauses`")
    ap.add_argument("--out", dest="out_path", required=True,
                    help="Output JSONL (one line per SAT instance)")
    ap.add_argument("--max_instances", type=int, default=0, help="0=all, else first N instances")
    ap.add_argument("--max_depth", type=int, default=100, help="Max recursion depth for trace")
    ap.add_argument("--max_trace_lines", type=int, default=0,
                    help="0=no limit; else truncate CoT trace lines to this many")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    n_inst = 0
    with open(args.in_path, "r", encoding="utf-8") as fin, open(args.out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            clauses = obj.get("clauses")
            if clauses is None:
                raise ValueError("Input JSONL must include `clauses` field (raw integer clauses).")

            inst_id = obj.get("id", n_inst)
            rec = build_serial_cot_record(
                instance_id=inst_id,
                clauses=clauses,
                max_depth=args.max_depth,
                max_trace_lines=args.max_trace_lines,
            )
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            n_inst += 1
            if n_inst % 100 == 0:
                print(f"Processed {n_inst} instances...")

            if args.max_instances and n_inst >= args.max_instances:
                break

    print(f"Done. Instances: {n_inst}, output: {args.out_path}")


if __name__ == "__main__":
    main()
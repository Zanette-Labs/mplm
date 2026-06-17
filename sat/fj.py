import argparse
import json
import os
from typing import Dict, List, Optional, Tuple, Any


SYSTEM_TEXT = "You are a helpful assistant."


# CNF formatting
def format_cnf(clauses: List[List[int]]) -> str:
    """Format CNF string using ∨ and ∧, with ¬ for negation."""
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


def recv_open_tag(child_ids: List[str]) -> str:
    """Return <recv [id0, id1, ...]> open tag."""
    return "<recv [" + ", ".join(child_ids) + "]>"


def format_assignment_list(delta: List[Tuple[int, bool]]) -> str:
    if not delta:
        return "No unit clauses."
    return "\n".join([f"- Unit: x{v} = {val}" for v, val in delta])


def wrap_msg(msg: str) -> str:
    """Wrap message payload with outermost braces."""
    return "{" + msg + "}"


def is_sat_message(msg: Optional[str]) -> bool:
    return bool(msg) and msg.strip().startswith("SAT")


# FJ trace generation
class CallIdGen:
    def __init__(self):
        self._cid = 0

    def next(self) -> int:
        x = self._cid
        self._cid += 1
        return x


def make_root_prompt(cnf_str: str) -> str:
    return cnf_str


def make_worker_prompt(wid: str, cnf_str: str) -> str:
    return f"Your id is: {wid}.\n{cnf_str}"


def build_fj_tree_for_instance(
    instance_id: Any,
    clauses: List[List[int]],
    max_depth: int,
    max_calls: int,
) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    cid_gen = CallIdGen()

    def append_call(rec: Dict[str, Any]) -> None:
        if len(calls) >= max_calls:
            raise RuntimeError(f"Reached max_calls={max_calls}")
        calls.append(rec)

    def safe_msg(m: Optional[str]) -> str:
        return m if m is not None else "NO_MESSAGE_FROM_CHILD"

    def recurse(
        wid: str,
        parent_id: Optional[str],
        cnf_clauses: List[List[int]],
        depth: int,
        incoming_prompt: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Returns:
          (sat?, raw_message_to_parent_or_None)

        IMPORTANT (your request):
          - raw_message is ONLY "SAT." or "UNSAT."
        """
        if depth > max_depth:
            msg = "UNSAT."
            resp = "Exceeded depth limit; stop search.\n"
            if parent_id is not None:
                resp += f"<send [{parent_id}]>{wrap_msg(msg)}</send> <stop>"
            else:
                resp += f"<stop>{msg}</stop>"

            append_call({
                "instance_id": instance_id,
                "call_id": cid_gen.next(),
                "wid": wid,
                "parent": parent_id,
                "children": [],
                "phase": "leaf",
                "system": SYSTEM_TEXT,
                "prompt": incoming_prompt,
                "response": resp,
                "depth": depth,
            })
            return False, msg if parent_id is not None else None

        assignment: Dict[int, bool] = {}
        simplified, unit_delta, bc = unit_propagate_all([c[:] for c in cnf_clauses], assignment)

        # Leaf after unit propagation
        if bc is not None:
            if bc is True:
                msg = "SAT."
                resp = (
                    "Try unit propagation.\n"
                    + format_assignment_list(unit_delta)
                    + "\nResult: SAT."
                )
            else:
                msg = "UNSAT."
                resp = (
                    "Try unit propagation.\n"
                    + format_assignment_list(unit_delta)
                    + "\nResult: UNSAT."
                )

            if parent_id is not None:
                resp += f"\n<send [{parent_id}]>{wrap_msg(msg)}</send> <stop>"
            else:
                resp += f"\n<stop>{msg}</stop>"

            append_call({
                "instance_id": instance_id,
                "call_id": cid_gen.next(),
                "wid": wid,
                "parent": parent_id,
                "children": [],
                "phase": "leaf",
                "system": SYSTEM_TEXT,
                "prompt": incoming_prompt,
                "response": resp,
                "depth": depth,
            })
            return bc, msg if parent_id is not None else None

        # Branch variable
        var = pick_branch_var(simplified, assignment)
        if var is None:
            msg = "SAT."
            resp = (
                "Try unit propagation.\n"
                + format_assignment_list(unit_delta)
                + "\nNo variables left.\n"
                + f"<stop>{msg}</stop>"
            )
            if parent_id is not None:
                resp += f"\n<send [{parent_id}]>{wrap_msg(msg)}</send> <stop>"

            append_call({
                "instance_id": instance_id,
                "call_id": cid_gen.next(),
                "wid": wid,
                "parent": parent_id,
                "children": [],
                "phase": "leaf",
                "system": SYSTEM_TEXT,
                "prompt": incoming_prompt,
                "response": resp,
                "depth": depth,
            })
            return True, msg if parent_id is not None else None

        # ---- Your requested convention:
        # wid.0 => x=0 (False), wid.1 => x=1 (True)
        child_false = f"{wid}.0"
        child_true = f"{wid}.1"
        child_ids = [child_false, child_true]

        cnf_false = simplify_cnf(simplified, var, False)
        cnf_true = simplify_cnf(simplified, var, True)

        cnf_false_str = format_cnf(cnf_false)
        cnf_true_str = format_cnf(cnf_true)

        resp_a_lines = [
            "Try unit propagation.",
            format_assignment_list(unit_delta),
            f"Branch on x{var}.",
            f"<spawn [{child_false}]>",
            cnf_false_str,
            "</spawn>",
            f"<spawn [{child_true}]>",
            cnf_true_str,
            "</spawn>",
            recv_open_tag(child_ids),  # open tag
        ]
        response_a = "\n".join(resp_a_lines)

        append_call({
            "instance_id": instance_id,
            "call_id": cid_gen.next(),
            "wid": wid,
            "parent": parent_id,
            "children": child_ids,
            "phase": "spawn",
            "system": SYSTEM_TEXT,
            "prompt": incoming_prompt,
            "response": response_a,
            "depth": depth,
        })

        # Recurse on children (Fork/Join: evaluate both)
        prompt_child_false = make_worker_prompt(child_false, cnf_false_str)
        prompt_child_true = make_worker_prompt(child_true, cnf_true_str)

        sat_f, msg_f = recurse(
            wid=child_false,
            parent_id=wid,
            cnf_clauses=cnf_false,
            depth=depth + 1,
            incoming_prompt=prompt_child_false,
        )
        sat_t, msg_t = recurse(
            wid=child_true,
            parent_id=wid,
            cnf_clauses=cnf_true,
            depth=depth + 1,
            incoming_prompt=prompt_child_true,
        )

        recv_all_lines = [
            wrap_msg(safe_msg(msg_f)),
            wrap_msg(safe_msg(msg_t)),
            "</recv>",
        ]
        recv_all_suffix = "\n".join(recv_all_lines)

        prompt_b = incoming_prompt + "\n" + response_a + "\n" + recv_all_suffix

        # Decide after both messages arrived (SAT if any child SAT)
        if sat_f or is_sat_message(msg_f) or sat_t or is_sat_message(msg_t):
            final_sat = True
            up_msg = "SAT."
        else:
            final_sat = False
            up_msg = "UNSAT."

        if parent_id is not None:
            response_b = (
                f"Conclusion for this subproblem: {up_msg}\n"
                f"<send [{parent_id}]>{wrap_msg(up_msg)}</send> <stop>"
            )
        else:
            response_b = f"<stop>{up_msg}</stop>"

        append_call({
            "instance_id": instance_id,
            "call_id": cid_gen.next(),
            "wid": wid,
            "parent": parent_id,
            "children": child_ids,
            "phase": "aggregate",
            "system": SYSTEM_TEXT,
            "prompt": prompt_b,
            "response": response_b,
            "depth": depth,
        })

        return final_sat, up_msg if parent_id is not None else None

    root_wid = "0"
    root_prompt = make_root_prompt(format_cnf(clauses))
    recurse(
        wid=root_wid,
        parent_id=None,
        cnf_clauses=clauses,
        depth=0,
        incoming_prompt=root_prompt,
    )
    return calls


# CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSONL with `clauses`")
    ap.add_argument("--out", dest="out_path", required=True, help="Output JSONL (one line per LLM call)")
    ap.add_argument("--max_instances", type=int, default=0, help="0=all, else first N instances")
    ap.add_argument("--max_depth", type=int, default=100, help="Max recursion depth for trace")
    ap.add_argument("--max_calls", type=int, default=200000, help="Max total LLM calls per instance")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    n_inst = 0
    n_calls = 0
    with open(args.in_path, "r", encoding="utf-8") as fin, open(args.out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            clauses = obj.get("clauses")
            if clauses is None:
                raise ValueError("Input JSONL must include `clauses` field (raw integer clauses).")

            inst_id = obj.get("id", n_inst)

            calls = build_fj_tree_for_instance(
                instance_id=inst_id,
                clauses=clauses,
                max_depth=args.max_depth,
                max_calls=args.max_calls,
            )

            for rec in calls:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            n_inst += 1
            n_calls += len(calls)
            if n_inst % 10 == 0:
                print(f"Processed {n_inst} instances, wrote {n_calls} calls...")

            if args.max_instances and n_inst >= args.max_instances:
                break

    print(f"Done. Instances: {n_inst}, total calls: {n_calls}, output: {args.out_path}")


if __name__ == "__main__":
    main()
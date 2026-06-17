#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mpi_sat_trace_builder_lazy_height.py

Changes vs original:
1) LAZY recursion: parent only recurses into a child when it truly "receives" that child's message.
   => If SAT found early, the other branch subtree is NOT explored.
2) Child id convention: wid.0 corresponds to x=0 (False), wid.1 corresponds to x=1 (True).
3) Parent receive order: determined by subtree height (smaller height received earlier).
   If equal height, tie-break randomly using instance RNG.

NEW CHANGE (your request):
- All children (and all messages) only return "SAT." or "UNSAT." (NO model).
"""

import argparse
import json
import os
import random
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


def format_assignment_list(delta: List[Tuple[int, bool]]) -> str:
    if not delta:
        return "No unit clauses."
    return "\n".join([f"- Unit: x{v} = {val}" for v, val in delta])


def wrap_msg(msg: str) -> str:
    """Wrap message payload with outermost braces."""
    return "{" + msg + "}"


def is_sat_message(msg: Optional[str]) -> bool:
    # Now only SAT/UNSAT exist; keep it robust.
    return bool(msg) and msg.strip().startswith("SAT")


# MPI trace generation
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


def build_mpi_tree_for_instance(
    instance_id: Any,
    clauses: List[List[int]],
    max_depth: int,
    max_calls: int,
    rng: random.Random,
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
    ) -> Tuple[bool, Optional[str], int]:
        """
        Returns:
          (sat?, raw_message_to_parent_or_None, subtree_height)

        subtree_height:
          - leaf: 0
          - internal: 1 + max(child_heights_of_expanded_children)
            (With lazy eval, this is the true height of the explored subtree for this node.)

        raw_message is NOT wrapped; wrapping is applied when embedding into <send>/<recv>.

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
                "subtree_height": 0,
            })
            return False, msg if parent_id is not None else None, 0

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
                "subtree_height": 0,
            })
            return bc, msg if parent_id is not None else None, 0

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
                "subtree_height": 0,
            })
            return True, msg if parent_id is not None else None, 0

        # --- Child id convention: wid.0 => x=0 (False), wid.1 => x=1 (True)
        child_false = f"{wid}.0"
        child_true = f"{wid}.1"
        child_ids = [child_false, child_true]

        cnf_false = simplify_cnf(simplified, var, False)
        cnf_true = simplify_cnf(simplified, var, True)

        cnf_false_str = format_cnf(cnf_false)
        cnf_true_str = format_cnf(cnf_true)

        # Spawn phase response: includes open recv tag only (NO close here)
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
            "<recv>",
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

        # Prepare child prompts
        prompt_child_false = make_worker_prompt(child_false, cnf_false_str)
        prompt_child_true = make_worker_prompt(child_true, cnf_true_str)

        # --- Lazy child evaluation cache: (sat, msg, height)
        child_bundle: Dict[str, Tuple[bool, Optional[str], int]] = {}

        def eval_child(child_wid: str) -> Tuple[bool, Optional[str], int]:
            if child_wid in child_bundle:
                return child_bundle[child_wid]

            if child_wid == child_false:
                out = recurse(
                    wid=child_false,
                    parent_id=wid,
                    cnf_clauses=cnf_false,
                    depth=depth + 1,
                    incoming_prompt=prompt_child_false,
                )
            else:
                out = recurse(
                    wid=child_true,
                    parent_id=wid,
                    cnf_clauses=cnf_true,
                    depth=depth + 1,
                    incoming_prompt=prompt_child_true,
                )
            child_bundle[child_wid] = out
            return out

        def child_height(child_wid: str) -> int:
            # Evaluate to know its subtree height (requirement: order depends on subtree height).
            return eval_child(child_wid)[2]

        # Receive policy:
        # - Decide whether to receive 1 or 2 in the first batch (same as your original).
        # - Order children by subtree height (smaller-height first); if tie, random.
        first_k = 1 if rng.random() < 0.5 else 2

        # Compute heights (forces evaluation of those children; required by the policy)
        h_false = child_height(child_false)
        h_true = child_height(child_true)

        # Sort by (height, random_tiebreak)
        tie_a = rng.random()
        tie_b = rng.random()
        order = sorted(
            [(child_false, h_false, tie_a), (child_true, h_true, tie_b)],
            key=lambda x: (x[1], x[2]),
        )
        ordered_ids = [x[0] for x in order]

        first_batch = ordered_ids[:first_k]
        remaining = ordered_ids[first_k:]

        # First recv block
        recv1_lines = [wrap_msg(safe_msg(eval_child(c)[1])) for c in first_batch] + ["</recv>"]
        recv1_suffix = "\n".join(recv1_lines)

        prompt_b1 = incoming_prompt + "\n" + response_a + "\n" + recv1_suffix

        # Determine if first batch already resolves SAT
        sat_found = False
        for c in first_batch:
            c_sat, c_msg, _h = eval_child(c)
            if c_sat or is_sat_message(c_msg):
                sat_found = True
                break

        # If received 2, OR received 1 and it already SAT -> finish immediately
        if first_k == 2 or sat_found:
            if sat_found:
                final_sat = True
                up_msg = "SAT."
            else:
                final_sat = False
                up_msg = "UNSAT."

            node_height = 1 + max(h_false, h_true)

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
                "prompt": prompt_b1,
                "response": response_b,
                "depth": depth,
                "subtree_height": node_height,
            })

            return final_sat, up_msg if parent_id is not None else None, node_height

        # Else: first_k == 1 and first message did NOT resolve SAT -> do a second receive of ONE more message
        if not remaining:
            up_msg = "UNSAT."
            node_height = 1 + max(h_false, h_true)
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
                "prompt": prompt_b1,
                "response": response_b,
                "depth": depth,
                "subtree_height": node_height,
            })
            return False, up_msg if parent_id is not None else None, node_height

        second_child = remaining[0]
        c_sat2, c_msg2, _h2 = eval_child(second_child)

        # ---- Call #1: WAIT ----
        response_wait = "<recv>"

        append_call({
            "instance_id": instance_id,
            "call_id": cid_gen.next(),
            "wid": wid,
            "parent": parent_id,
            "children": child_ids,
            "phase": "aggregate_wait",
            "system": SYSTEM_TEXT,
            "prompt": prompt_b1,
            "response": response_wait,
            "depth": depth,
        })

        # ---- Call #2: DECIDE ----
        prompt_decide = "\n".join([
            prompt_b1,
            response_wait,
            wrap_msg(safe_msg(c_msg2)),
            "</recv>",
        ])

        if c_sat2 or is_sat_message(c_msg2):
            final_sat = True
            up_msg = "SAT."
        else:
            final_sat = False
            up_msg = "UNSAT."

        node_height = 1 + max(h_false, h_true)

        if parent_id is not None:
            response_decide = (
                f"Conclusion for this subproblem: {up_msg}\n"
                f"<send [{parent_id}]>{wrap_msg(up_msg)}</send> <stop>"
            )
        else:
            response_decide = f"<stop>{up_msg}</stop>"

        append_call({
            "instance_id": instance_id,
            "call_id": cid_gen.next(),
            "wid": wid,
            "parent": parent_id,
            "children": child_ids,
            "phase": "aggregate_decide",
            "system": SYSTEM_TEXT,
            "prompt": prompt_decide,
            "response": response_decide,
            "depth": depth,
            "subtree_height": node_height,
        })

        return final_sat, up_msg if parent_id is not None else None, node_height

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSONL with `clauses`")
    ap.add_argument("--out", dest="out_path", required=True, help="Output JSONL (one line per LLM call)")
    ap.add_argument("--max_instances", type=int, default=0, help="0=all, else first N instances")
    ap.add_argument("--max_depth", type=int, default=100, help="Max recursion depth for trace")
    ap.add_argument("--max_calls", type=int, default=1000000, help="Max total LLM calls per instance")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (0=deterministic default)")
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

            # Mix global seed with instance id to get a unique RNG for each instance
            inst_key = str(inst_id)
            inst_mix = 0
            for ch in inst_key:
                inst_mix = (inst_mix * 131 + ord(ch)) & 0xFFFFFFFF
            inst_rng = random.Random((args.seed * 1000003 + inst_mix) & 0xFFFFFFFF)

            calls = build_mpi_tree_for_instance(
                instance_id=inst_id,
                clauses=clauses,
                max_depth=args.max_depth,
                max_calls=args.max_calls,
                rng=inst_rng,
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
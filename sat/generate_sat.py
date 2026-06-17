import argparse
import json
import os
import random
from typing import List, Tuple, Optional, Dict, Any

from pysat.solvers import Glucose3


def format_cnf(clauses: List[List[int]]) -> str:
    """Format CNF formula using ∨ for OR and ∧ for AND."""
    return " ∧ ".join(
        f"( {' ∨ '.join(str(lit) if lit > 0 else f'¬ {abs(lit)}' for lit in clause)} )"
        for clause in clauses
    )


class SATGenerator:
    """
    Generates random 3-SAT instances.
    - SAT instances are guaranteed satisfiable by construction.
    - UNSAT instances are generated randomly and (optionally) verified with Glucose3.
    """

    def __init__(
        self,
        min_vars: int = 10,
        max_vars: int = 20,
        clause_var_ratio: float = 4.5,
        satisfiable_ratio: float = 0.5,
        max_attempts: int = 5000,
        verify_unsat_always: bool = True,
    ):
        self.min_vars = min_vars
        self.max_vars = max_vars
        self.clause_var_ratio = clause_var_ratio
        self.satisfiable_ratio = satisfiable_ratio
        self.MAX_ATTEMPTS = max_attempts
        self.verify_unsat_always = verify_unsat_always

    def generate_instance(
        self,
        num_variables: Optional[int] = None,
        num_clauses: Optional[int] = None,
        make_satisfiable: Optional[bool] = None,
        return_raw: bool = False,
    ) -> Tuple[bool, int, int, Any]:
        """
        Generate one instance.

        Returns:
          (is_sat_label, num_vars, num_clauses, clauses_or_problem_str)
        """
        num_variables = num_variables or random.randint(self.min_vars, self.max_vars)
        num_clauses = num_clauses or int(num_variables * self.clause_var_ratio)

        if make_satisfiable is None:
            make_satisfiable = random.random() < self.satisfiable_ratio

        selected_var_ids = self._select_variable_ids(num_variables)

        if make_satisfiable:
            clauses, _assignment = self._generate_satisfiable_instance(selected_var_ids, num_clauses)
        else:
            if self.clause_var_ratio == 1:
                clauses, _ = self._generate_unsatisfiable_instance_trivial(selected_var_ids, num_clauses)
                # trivial core is UNSAT by construction, but we can still optionally verify
                if self.verify_unsat_always and not self._verify_unsat(clauses):
                    raise RuntimeError("Trivial UNSAT generator produced SAT (unexpected).")
            else:
                clauses, _ = self._generate_unsatisfiable_instance(selected_var_ids, num_clauses)

        if return_raw:
            return make_satisfiable, num_variables, num_clauses, clauses

        problem_str = format_cnf(clauses)
        return make_satisfiable, num_variables, num_clauses, problem_str

    def generate_dataset_jsonl(
        self,
        num_samples: int,
        out_path: str,
        buffer_size: int = 2000,
        seed: Optional[int] = None,
        include_raw_clauses: bool = True,
    ) -> None:
        """
        Stream-generate a JSONL dataset.

        Each line has separated fields:
          - problem (CNF string)
          - answer (bool) and label (0/1)
        """
        if seed is not None:
            random.seed(seed)

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        buffer: List[str] = []
        with open(out_path, "w", buffering=8192, encoding="utf-8") as f:
            for i in range(num_samples):
                is_sat, nvars, nclauses, clauses = self.generate_instance(return_raw=True)
                item: Dict[str, Any] = {
                    "id": i,
                    "num_vars": nvars,
                    "num_clauses": nclauses,
                    "problem": format_cnf(clauses),
                    "answer": bool(is_sat),
                    "label": 1 if is_sat else 0,
                }
                if include_raw_clauses:
                    item["clauses"] = clauses

                buffer.append(json.dumps(item, ensure_ascii=False))
                if len(buffer) >= buffer_size:
                    f.write("\n".join(buffer) + "\n")
                    buffer.clear()

                if (i + 1) % 1000 == 0:
                    print(f"Generated {i + 1}/{num_samples} samples...")

            if buffer:
                f.write("\n".join(buffer) + "\n")
                buffer.clear()

    def _select_variable_ids(self, num_vars: int) -> List[int]:
        """Randomly select variable IDs from range 1..max_vars (keeps original behavior)."""
        possible_vars = list(range(1, self.max_vars + 1))
        return random.sample(possible_vars, num_vars)

    def _generate_satisfiable_instance(
        self,
        selected_var_ids: List[int],
        num_clauses: int,
    ) -> Tuple[List[List[int]], Dict[int, int]]:
        """Generate a satisfiable 3-SAT instance by fixing a random assignment and enforcing each clause satisfied."""
        assignment = {var_id: random.randint(0, 1) for var_id in selected_var_ids}
        clauses: List[List[int]] = []

        while len(clauses) < num_clauses:
            vars_in_clause = random.sample(selected_var_ids, 3)
            negations = [random.choice([-1, 1]) for _ in range(3)]

            satisfied = False
            for j, var in enumerate(vars_in_clause):
                if (assignment[var] == 1 and negations[j] == 1) or (assignment[var] == 0 and negations[j] == -1):
                    satisfied = True
                    break

            if not satisfied:
                idx = random.randint(0, 2)
                negations[idx] = -negations[idx]

            clause = [var * neg for var, neg in zip(vars_in_clause, negations)]
            clauses.append(clause)

        return clauses, assignment

    def _generate_unsatisfiable_instance(
        self,
        selected_var_ids: List[int],
        num_clauses: int,
    ) -> Tuple[List[List[int]], None]:
        """Generate an unsatisfiable 3-SAT instance by random sampling + verification."""
        for _attempt in range(self.MAX_ATTEMPTS):
            clauses: List[List[int]] = []
            while len(clauses) < num_clauses:
                vars_in_clause = random.sample(selected_var_ids, 3)
                negations = [random.choice([-1, 1]) for _ in range(3)]
                clause = [v * n for v, n in zip(vars_in_clause, negations)]
                clauses.append(clause)

            if self.verify_unsat_always:
                if self._verify_unsat(clauses):
                    return clauses, None
            else:
                # original heuristic: accept when probability upper bound is very small OR verified
                prob_ub = (2 * (7 / 8) ** (num_clauses / len(selected_var_ids))) ** len(selected_var_ids)
                if prob_ub < 0.001 or self._verify_unsat(clauses):
                    return clauses, None

        raise RuntimeError(f"Failed to generate verified UNSAT instance after {self.MAX_ATTEMPTS} attempts")

    def _generate_unsatisfiable_instance_trivial(
        self,
        selected_var_ids: List[int],
        num_clauses: int,
    ) -> Tuple[List[List[int]], None]:
        """Generate UNSAT 3-CNF with a small UNSAT core (has repeated literals; kept from original)."""
        clauses: List[List[int]] = []

        x1, x2 = selected_var_ids[:2]
        sign = random.choice([-1, 1])

        clauses.append([sign * x1, sign * x1, sign * x1])
        clauses.append([-sign * x1, sign * x2, sign * x2])
        clauses.append([-sign * x2, -sign * x1, -sign * x1])

        while len(clauses) < num_clauses:
            vars_in_clause = random.sample(selected_var_ids, 3)
            negations = [random.choice([-1, 1]) for _ in range(3)]
            clause = [v * n for v, n in zip(vars_in_clause, negations)]
            clauses.append(clause)

        return clauses, None

    @staticmethod
    def _verify_unsat(clauses: List[List[int]]) -> bool:
        """Return True iff formula is UNSAT."""
        solver = Glucose3()
        try:
            for clause in clauses:
                solver.add_clause(clause)
            is_sat = solver.solve()
            return not is_sat
        finally:
            solver.delete()


def main():
    parser = argparse.ArgumentParser(description="3-SAT Dataset Generator (problem/answer separated)")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--out", type=str, required=True, help="Output JSONL path")

    parser.add_argument("--min_vars", type=int, default=9, help="Minimum number of variables")
    parser.add_argument("--max_vars", type=int, default=9, help="Maximum number of variables")
    parser.add_argument("--clause_var_ratio", type=float, default=4.3, help="m/n ratio (clauses per variable)")
    parser.add_argument("--satisfiable_ratio", type=float, default=0.5, help="Fraction of SAT instances")

    parser.add_argument("--seed", type=int, default=42, help="Random seed (0 means deterministic seed=0)")
    parser.add_argument("--buffer_size", type=int, default=2000, help="Write buffer size (lines)")
    parser.add_argument(
        "--no_verify_unsat_always",
        action="store_true",
        help="If set, use original heuristic (prob_ub < 0.001) OR verify UNSAT; otherwise always verify UNSAT.",
    )
    parser.add_argument(
        "--no_raw_clauses",
        action="store_true",
        help="If set, do NOT include raw integer clauses in JSONL (only pretty CNF string).",
    )

    args = parser.parse_args()

    gen = SATGenerator(
        min_vars=args.min_vars,
        max_vars=args.max_vars,
        clause_var_ratio=args.clause_var_ratio,
        satisfiable_ratio=args.satisfiable_ratio,
        verify_unsat_always=not args.no_verify_unsat_always,
    )

    gen.generate_dataset_jsonl(
        num_samples=args.num_samples,
        out_path=args.out,
        buffer_size=args.buffer_size,
        seed=args.seed,
        include_raw_clauses=not args.no_raw_clauses,
    )

    print(f"Done. Wrote {args.num_samples} samples to: {args.out}")


if __name__ == "__main__":
    main()
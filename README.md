# Message Passing Language Model

This repository holds data-generation and inference scripts for evaluating
**MPLM** (a multi-agent, message-passing language-model protocol with
`<spawn>` / `<send>` / `<recv>` / `<stop>` directives) on three task families:

| Directory          | Task                           | Data generators                 | Inference / eval script                                   |
| ------------------ | ------------------------------ | ------------------------------- | --------------------------------------------------------- |
| `sudoku/`          | N×N Sudoku                     | `mplm.py`, `fj.py`, `serial.py` | `infer.py`                                             |
| `sat/`             | 3-SAT (SAT / UNSAT)            | `mplm.py`, `fj.py`, `serial.py` | `infer.py`                                             |
| `long_context_qa/` | LongBench-v2 (long-context QA) | *(none — use the HF dataset)*   | `run_mplm_longbenchv2.py`, `run_rlm_longbenchv2.py`       |

For every task there are two distinct data artifacts:

* **Training data** — multi-agent traces produced by the `mplm`/`fj`/`serial`
  generators, used to SFT the model.
* **Evaluation data** — the held-out instances fed to the inference scripts.

The three generators per task differ only in the *solving strategy* they
record, not in the output schema:

* `mplm`   — fully parallel agents that pass messages (cell-per-worker for
  Sudoku; lazy DPLL branch agents for SAT).
* `fj`     — fork/join: a master repeatedly spawns workers and joins their replies.
* `serial` — a single agent that solves the whole instance in one chain of thought.

## Pretrained models

The trained Sudoku and SAT models are released on Hugging Face at
**<https://huggingface.co/xuechengliu/mplm>**. Download the model you need and
pass its local directory to the task's `infer.py` via `--model`.

---

## Prerequisites

All inference scripts talk to an **OpenAI-compatible vLLM server**. Start one
per model before running an eval:

```bash
vllm serve <MODEL_DIR> \
    --served-model-name <SERVED_NAME> \
    --host <HOST> --port <PORT> \
    --max-model-len <CTX_LEN> \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code
```

The clients default to `http://127.0.0.1:8000`, served name `model`, and API
key `EMPTY`; override with the `--vllm-url` / `--served-model-name` / `--api-key`
flags or the `VLLM_URL` / `VLLM_MODEL_NAME` / `VLLM_API_KEY` environment
variables. Use `CUDA_VISIBLE_DEVICES` to pin a server to specific GPUs.

> **Tip:** launch the eval scripts with your environment's Python binary
> directly (e.g. `</path/to/env>/bin/python infer.py ...`, or after
> `conda activate <env>`) rather than through `conda run`; the latter buffers
> stdout and hides the per-task progress these scripts stream as they go.

---

## 1. Sudoku (`sudoku/`)

### 1.1 Prepare training data

Input is a JSON **array** of puzzles, each with an `original` grid (0 = empty)
and its `solution` grid (both N×N, N a perfect square):

```json
[
  {"original":  [[3,0,0,1], [0,0,3,0], [0,3,0,0], [0,0,0,3]],
   "solution":  [[3,4,2,1], [2,1,3,4], [1,3,4,2], [4,2,1,3]]}
]
```

Generate traces with any of the three strategies (they emit the **same**
record schema, so the outputs are interchangeable for training):

```bash
# parallel cell-workers
python sudoku/mplm.py   --input <PUZZLES_JSON> --out <MPLM_TRACES_JSON>   --max-puzzles 1000
# fork/join master+workers
python sudoku/fj.py     --input <PUZZLES_JSON> --out <FJ_TRACES_JSON>     --max-puzzles 1000
# single-agent chain of thought
python sudoku/serial.py --input <PUZZLES_JSON> --out <SERIAL_TRACES_JSON> --max-puzzles 1000
```

`--input` and `--out` are required. Other flags: `--max-puzzles N` (cap), and
for `mplm.py`/`fj.py` `--master-upsample`, `--worker-downsample`,
`--remove-workers` to rebalance the master vs. worker training mix. Each output
record contains `{sudoku_id, N, base_n, master_system_prompt,
worker_system_prompt, original, solution, master[], workers{}}`.

### 1.2 Prepare evaluation data

`infer.py` reads a **JSONL** file (one object per line) with the fields
`sudoku_id`, `initial_prompt`, and `solution`. The prompt format must match
training exactly — `"Solve this Sudoku:\n{ (r,c): {v}, ... }"` listing only the
given cells:

```json
{"sudoku_id": 0, "initial_prompt": "Solve this Sudoku:\n{ (0,0): {3}, (0,5): {1} }", "solution": [[...],[...]]}
```

You can derive this JSONL from raw puzzles:

```python
import json
def clues(grid):
    items = [f"({r},{c}): {{{grid[r][c]}}}"
             for r in range(len(grid)) for c in range(len(grid))
             if grid[r][c] != 0]
    return "{ " + ", ".join(items) + " }"
with open("<PROMPTS_JSONL>", "w") as f:
    for i, p in enumerate(json.load(open("<PUZZLES_JSON>"))):
        f.write(json.dumps({"sudoku_id": i,
                            "initial_prompt": "Solve this Sudoku:\n" + clues(p["original"]),
                            "solution": p["solution"]}) + "\n")
```

### 1.3 Run inference

Serve the trained Sudoku model, then evaluate:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve <SUDOKU_MODEL_DIR> \
    --served-model-name model --host <HOST> --port <PORT> \
    --max-model-len 32768 --gpu-memory-utilization 0.9 --trust-remote-code

python sudoku/infer.py \
    --model         <SUDOKU_MODEL_DIR> \
    --prompts_jsonl <PROMPTS_JSONL> \
    --vllm-url      http://<HOST>:<PORT> \
    --max_rounds 500 --max_new_tokens 15000 --max_tasks -1
```

`--template_path` defaults to `<model>/chat_template.jinja` if present (pass it
explicitly otherwise). The controller runs the spawn/recv/send/stop protocol
round-by-round, parses the master's final `<stop>[[grid]]</stop>`, and scores it
by **strict grid equality** against `solution`. It prints per-task results plus
total accuracy and average latency (over successful solves). Add `--write_trace
--trace_dir <DIR>` to dump full per-task transcripts.

---

## 2. SAT (`sat/`)

### 2.1 Prepare data (one file feeds both)

A single SAT-instance **JSONL** serves both training-trace generation and
evaluation. Each line has:

```json
{"id": 0, "num_vars": 10, "num_clauses": 43,
 "problem": "( 7 ∨ 4 ∨ 1 ) ∧ ( ¬ 8 ∨ 9 ∨ 5 ) ∧ ...",
 "answer": true, "label": "SAT",
 "clauses": [[7, 4, 1], [-8, 9, 5], [10, -6, 3]]}
```

* `clauses` — raw integer clauses (negative = negated literal); **read by the
  trace generators**.
* `problem` — the CNF string the model actually sees; `answer` — ground-truth
  SAT (`true`) / UNSAT (`false`); **read by the evaluator**.

Generate training traces (`--in`/`--out` required; one JSONL record per LLM call):

```bash
python sat/mplm.py   --in <INSTANCES_JSONL> --out <MPLM_TRACES_JSONL>   --max_instances 0   # 0 = all
python sat/fj.py     --in <INSTANCES_JSONL> --out <FJ_TRACES_JSONL>     --max_instances 0
python sat/serial.py --in <INSTANCES_JSONL> --out <SERIAL_TRACES_JSONL> --max_instances 0
```

Useful flags: `--max_depth` (DPLL depth cap), `--max_calls` (per-instance call
cap), and for `mplm.py` a `--seed` (its receive order is randomized). Records
carry `{instance_id, call_id, wid, parent, children, phase, system, prompt,
response, depth}` (`mplm` additionally records `subtree_height`; `serial` emits
one `{instance_id, system, prompt, response}` per instance).

### 2.2 Run inference

Serve the trained SAT model, then evaluate over the instance JSONL (only
`problem` + `answer` are needed):

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve <SAT_MODEL_DIR> \
    --served-model-name model --host <HOST> --port <PORT> \
    --max-model-len 32768 --gpu-memory-utilization 0.9 --trust-remote-code

python sat/infer.py \
    --model         <SAT_MODEL_DIR> \
    --prompts_jsonl <INSTANCES_JSONL> \
    --vllm-url      http://<HOST>:<PORT> \
    --max_new_tokens 25000 --max_steps 1000000 --max_tasks -1 \
    --max_workers 625
```

 `--template_path` defaults to
`<model>/chat_template.jinja`; add `--verbose_steps` to print every agent turn.

---

## 3. Long-context QA — LongBench-v2 (`long_context_qa/`)

Two inference scripts evaluate the **same full benchmark** with two different
frameworks and share identical task selection, result files, skip/retry logic,
and grouped stats:

* `run_mplm_longbenchv2.py` — **MPLM multi-round**: split the context into
  ~10k-token chunks, each read by a reader agent that summarizes it; the master
  then runs `--query-rounds` rounds of `<send>` queries to specific agents
  before giving the final answer.
* `run_rlm_longbenchv2.py` — **RLM**: hand the whole context to a Recursive
  Language Model that reads/reasons over it recursively. Requires the `rlm`
  package — either `pip install` it, or set `RLM_PATH=<path/to/rlm/checkout>`
  so the script can import it.

### 3.1 Prepare data

No generation step — download the LongBench-v2 dataset from Hugging Face:

```bash
huggingface-cli download THUDM/LongBench-v2 --repo-type dataset
```

Pass the resulting `data.json` (503 tasks across 6 domains) to each script via
`--data-path <DATA_JSON>` or the `LONGBENCH_V2_DATA_PATH` environment variable
(it is required — there is no default). Each record has `_id`, `domain`,
`sub_domain`, `difficulty`, `length`, `question`, `choice_A`..`choice_D`,
`answer`, and `context`.

### 3.2 Run inference

Serve the chat model (OpenAI-compatible; the served model id is what you pass to
`--model-name`):

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve <MODEL_DIR> \
    --served-model-name <MODEL_NAME> --host <HOST> --port <PORT> \
    --gpu-memory-utilization 0.9 --trust-remote-code \
    --max-model-len <CTX_LEN>     # size to your per-call context + hardware
```

**MPLM multi-round:**

```bash
python long_context_qa/run_mplm_longbenchv2.py \
    --data-path  <DATA_JSON> \
    --model-name <MODEL_NAME> \
    --vllm-url   http://<HOST>:<PORT>/v1 \
    --query-rounds 3 --tokens-per-chunk 10000 --workers 4
```

**RLM** (install `rlm` or set `RLM_PATH`):

```bash
RLM_PATH=<path/to/rlm> python long_context_qa/run_rlm_longbenchv2.py \
    --data-path  <DATA_JSON> \
    --model-name <MODEL_NAME> \
    --vllm-url   http://<HOST>:<PORT>/v1 \
    --max-iterations 15 --max-depth 2 --max-concurrent-subcalls 4
```

Both scripts:

* select and sort tasks identically; narrow the run with `--domain`,
  `--sub-domain`, `--task-id` (repeatable) and `--limit`;
* write one `results/<task_id>.json` per task plus a rolling `summary.json` and
  `summary_stats.json` under `--output-dir` (default `./outputs/...`), and
  support `--overwrite` / `--retry-errors` to resume;
* parse the model's `"The correct answer is (X)"` and report overall accuracy
  broken down by domain / sub-domain / difficulty / length.

Run `python <script> --help` for the full set of tunables (token budgets,
temperature, timeouts, thinking toggles, etc.).

import re
import json
import os
import time
import argparse
import ast
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer

# vLLM client config (override via CLI flags or environment variables)
VLLM_BASE_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8000")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
VLLM_MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "model")
VLLM_MAX_WORKERS = int(os.environ.get("VLLM_MAX_WORKERS", "625"))

VLLM_TIMEOUT = 120
VLLM_RETRIES = 3

# Tag parsing helpers
TAG_SPAWN_RE = re.compile(
    r"<spawn\s*\[(?P<ids>[^\]]+)\]\s*>(?P<body>.*?)</spawn(?:\s*\[(?P=ids)\])?\s*>",
    re.DOTALL | re.IGNORECASE,
)

TAG_SEND_RE = re.compile(
    r"<send\s*\[(?P<ids>[^\]]+)\]\s*>(?P<body>.*?)</send(?:\s*\[(?P=ids)\])?\s*>",
    re.DOTALL | re.IGNORECASE,
)

# <recv [ids]>（blocking）and single <recv>
TAG_RECV_BLOCKING_RE = re.compile(
    r"<recv\s*\[(?P<ids>[^\]]+)\]\s*>",
    re.IGNORECASE,
)
TAG_RECV_ANY_RE = re.compile(
    r"<recv\s*>",
    re.IGNORECASE,
)

# Paired, or single <stop>
TAG_STOP_RE = re.compile(
    r"(?:<stop\s*>(?P<body>.*?)</stop\s*>)|(?:<stop\s*>)",
    re.DOTALL | re.IGNORECASE,
)

# For open <recv> detection in previously concatenated base
OPEN_RECV_RE = re.compile(
    r"<recv\s*(?:\[(?P<ids>[^\]]+)\])?\s*>",
    re.IGNORECASE,
)
CLOSE_RECV_RE = re.compile(
    r"</recv\s*(?:\[(?P<close_ids>[^\]]+)\])?\s*>",
    re.IGNORECASE,
)

def parse_id_list(s: str) -> List[str]:
    """Split '1, 2,3' -> ['1','2','3']."""
    return [x.strip() for x in s.split(",") if x.strip()]


def add_id_prefix(agent_id: str, body: str) -> str:
    """
    Prepend 'Your id is: {agent_id}.' to the body if it's not already present.
    We do NOT touch the original spawn body inside the assistant output; this
    only affects the prompt we build after a spawn.
    """
    prefix = f"Your id is: {agent_id}."
    # Avoid double prefix if body already starts with it (after stripping leading spaces)
    if body.lstrip().startswith(prefix):
        return body
    return f"{prefix}\n{body}"


# Directive parsing
@dataclass
class ParsedDirectives:
    spawns: List[Tuple[List[str], str]] = field(default_factory=list)
    sends: List[Tuple[List[str], str]] = field(default_factory=list)
    recv_blocking: Optional[List[str]] = None
    recv_any: bool = False
    stop_body: Optional[str] = None  # "" means bare stop


def parse_directives(text: str) -> ParsedDirectives:
    """Parse directives from an assistant output string."""
    result = ParsedDirectives()

    for m in TAG_SPAWN_RE.finditer(text):
        result.spawns.append((parse_id_list(m.group("ids")), m.group("body").strip()))

    for m in TAG_SEND_RE.finditer(text):
        result.sends.append((parse_id_list(m.group("ids")), m.group("body").strip()))

    m_block = TAG_RECV_BLOCKING_RE.search(text)
    m_any = TAG_RECV_ANY_RE.search(text)
    if m_block and m_any:
        result.recv_blocking = parse_id_list(m_block.group("ids"))
    elif m_block:
        result.recv_blocking = parse_id_list(m_block.group("ids"))
    elif m_any:
        result.recv_any = True

    m_stop = TAG_STOP_RE.search(text)
    if m_stop:
        body = m_stop.groupdict().get("body")
        result.stop_body = (body.strip() if body else "")

    return result


# Data structures
@dataclass
class Message:
    role: str   # "system" | "user" | "assistant"
    content: str


@dataclass
class QueuedMsg:
    src_id: str
    content: str


@dataclass
class Event:
    """
    One logical unit since last spawn.
    We keep both the raw assistant output and (optionally) one stitched <recv> block for debug.
    """
    assistant_raw: str = ""
    recv_block: Optional[str] = None


@dataclass
class AgentState:
    agent_id: str

    # Human-readable debug transcript (not used to build next inputs)
    messages: List[Message] = field(default_factory=list)

    # Messaging & control
    inbox: List[QueuedMsg] = field(default_factory=list)
    waiting_for: Optional[List[str]] = None
    wait_any: bool = False
    stopped: bool = False

    # Since last spawn
    seed_instruction: Optional[str] = None
    events_since_spawn: List[Event] = field(default_factory=list)

    # Canonical prompt history used to build future rounds (seed + all recv + all assistant)
    history_since_spawn: str = ""

    def can_run(self) -> bool:
        """Ready if not stopped and not blocked by recv (or recv is now satisfied)."""
        if self.stopped:
            return False
        if self.waiting_for is None and not self.wait_any:
            return True
        if self.waiting_for is not None:
            got = {m.src_id for m in self.inbox}
            return set(self.waiting_for).issubset(got)
        if self.wait_any:
            return len(self.inbox) > 0
        return True

    def _format_recv_line(self, q: QueuedMsg) -> str:
        """Normalize a recv line to dataset style."""
        txt = q.content.strip()
        if txt.startswith("{") and txt.endswith("}"):
            return txt
        return f"{{ From: {q.src_id}, {txt} }}"

    def consume_recv_lines_and_meta(self) -> Tuple[List[str], Optional[List[str]]]:
        """
        Consume messages and return:
            ( list_of_formatted_lines , ids_hint or None )
        DOES NOT wrap with <recv> tags here. Wrapping and persisting into history
        is handled by the controller right after this returns.
        Resets waiting flags appropriately.
        """
        # Blocking case: collect ALL messages from each requested sender id
        if self.waiting_for is not None:
            buckets: Dict[str, List[QueuedMsg]] = {sid: [] for sid in self.waiting_for}
            keep: List[QueuedMsg] = []

            for q in self.inbox:
                if q.src_id in buckets:
                    buckets[q.src_id].append(q)
                else:
                    keep.append(q)

            selected: List[QueuedMsg] = []
            for sid in self.waiting_for:
                selected.extend(buckets[sid])

            self.inbox = keep

            lines = [self._format_recv_line(m) for m in selected]
            ids_hint = list(self.waiting_for)
            self.waiting_for = None
            self.wait_any = False
            return lines, ids_hint

        # ANY case: take everything and clear inbox
        if self.wait_any and self.inbox:
            lines = [self._format_recv_line(m) for m in self.inbox]
            self.inbox.clear()
            self.wait_any = False
            return lines, None

        return [], None

    def queue_message(self, src_id: str, content: str):
        self.inbox.append(QueuedMsg(src_id=src_id, content=content))


# Utilities for <recv> stitching
def _detect_unclosed_recv_suffix(text: str) -> Optional[Tuple[str, Optional[str]]]:
    """
    Detect if the given text ends with an unclosed <recv> (or <recv [ids]>).
    Returns:
        ( opener_tag, ids_str_or_none )
    If no dangling opener is found, returns None.
    """
    last_open = None
    for m in OPEN_RECV_RE.finditer(text):
        last_open = m  # keep last
    last_close = None
    for m in CLOSE_RECV_RE.finditer(text):
        last_close = m

    if last_open is None:
        return None
    # Dangling if there is no close OR the last open occurs after the last close
    if (last_close is None) or (last_open.end() > last_close.end()):
        ids_str = last_open.group("ids")  # may be None
        opener = f"<recv [{ids_str}]>" if ids_str else "<recv>"
        return (opener, ids_str)
    return None


def _wrap_recv_lines_once(base: str, lines: List[str], ids_hint: Optional[List[str]]) -> str:
    """
    Append 'lines' to 'base' using one pair of <recv> ... </recv> tags.
    If 'base' already ends with a dangling opener, only append lines + matching closer.
    Otherwise open and close a fresh pair (using ids_hint when available).
    """
    if not lines:
        return base

    suffix_newline = "\n" if (base and not base.endswith("\n")) else ""
    body = "\n".join(lines) + ("\n" if lines else "")

    dangling = _detect_unclosed_recv_suffix(base)
    if dangling:
        # Close the open recv tag
        _, ids_str = dangling
        close_tag = "</recv>" # f"</recv [{ids_str}]>" if ids_str else "</recv>"
        return f"{base}{suffix_newline}{body}{close_tag}"

    # No open recv tag
    if ids_hint:
        ids_compact = ",".join([s.strip() for s in ids_hint if s.strip()])
        open_tag = f"<recv [{ids_compact}]>"
        close_tag = f"</recv>" # f"</recv [{ids_compact}]>"
    else:
        open_tag = "<recv>"
        close_tag = "</recv>"
    return f"{base}{suffix_newline}{open_tag}\n{body}{close_tag}"


# Runner (vLLM completions)
class Runner:
    """
    vLLM (OpenAI-compatible) runner using /v1/completions.
    - Restore local tokenizer + chat template so the prompt exactly matches SFT.
    - Controller still calls build_prompt_from_user_text(), and we return the
      fully formatted single-string prompt (system+user) using your template.
    - Per round: send ALL prompts in parallel; wait for ALL to finish; return
      texts in the original order.
    """

    FIXED_SYSTEM = "You are a helpful assistant."

    def __init__(
        self,
        model_name_or_path: str,       # used to load tokenizer/template
        template_path: Optional[str],  # load local chat_template.jinja
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.tokenizer.padding_side = "left"
        if template_path:
            with open(template_path, "r", encoding="utf-8") as f:
                self.tokenizer.chat_template = f.read()

        # OpenAI-compatible HTTP client (keep-alive)
        self.base_url = VLLM_BASE_URL
        self.api_key = VLLM_API_KEY
        self.model_name = VLLM_MODEL_NAME
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

        # Concurrency & policy
        self.max_workers = VLLM_MAX_WORKERS
        self.timeout_sec = float(VLLM_TIMEOUT)
        self.max_retries = int(VLLM_RETRIES)

    def build_prompt_from_user_text(self, user_text: str) -> str:
        """
        Build a SINGLE-turn prompt string using your local chat template.
        This reproduces the exact formatting used during SFT (no server-side templating).
        """
        chat = [
            {"role": "system", "content": self.FIXED_SYSTEM},
            {"role": "user",   "content": user_text},
        ]
        # add_generation_prompt=True appends the assistant preamble per template
        return self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False
        )

    # Internal: one /v1/completions call with retries (send "prompt" string)
    def _post_completion(self, prompt: str, payload_common: dict) -> str:
        url = f"{self.base_url}/v1/completions"
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                payload = dict(payload_common)
                payload["prompt"] = prompt  # send the formatted prompt
                resp = self.session.post(url, data=json.dumps(payload), timeout=self.timeout_sec)
                if resp.status_code == 200:
                    data = resp.json()
                    # OpenAI-compatible completions: choices[0].text is the pure completion (without prompt)
                    text = (data.get("choices", [{}])[0].get("text", "") or "").strip()
                    return text
                else:
                    last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            except Exception as e:
                last_err = e
            time.sleep(0.2 * attempt)

        raise RuntimeError(f"Completion failed after {self.max_retries} attempts: {last_err}")

    def generate(
        self,
        prompts: List[str],                 # Each prompt is the fully formatted string
        max_new_tokens: int = 8192,
    ) -> List[str]:
        """
        Parallel remote generation using vLLM /v1/completions.
        - We already did client-side templating; server must not re-template.
        - Send all requests in parallel; wait for all; preserve order.
        """
        if not prompts:
            return []

        payload_common = {
            "model": self.model_name,
            "max_tokens": int(max_new_tokens),
            "n": 1,
            "stream": False,
            "temperature": 0
        }

        results: List[Optional[str]] = [None] * len(prompts)
        max_workers = min(self.max_workers, len(prompts))

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut2idx = {
                ex.submit(self._post_completion, p, payload_common): i
                for i, p in enumerate(prompts)
            }
            for fut in as_completed(fut2idx):
                idx = fut2idx[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    raise RuntimeError(f"vLLM completions failed at index {idx}: {e}") from e

        assert all(r is not None for r in results), "Some vLLM responses are missing"
        return [r for r in results]


# Run tracer
class RunTracer:
    """
    Collects a full trace object and writes it to JSON:
    [
      {
        "master_system_prompt": "...",
        "worker_system_prompt": "...",
        "master": [ { "prompt": "...", "response": "..." }, ... ],
        "workers": { "<id>": [ { "prompt": "...", "response": "..." }, ... ], ... }
      }
    ]
    """
    def __init__(self, master_id: str, system_prompt_master: str, system_prompt_worker: str):
        self.master_id = str(master_id)
        self.trace = {
            "master_system_prompt": system_prompt_master,
            "worker_system_prompt": system_prompt_worker,
            "master": [],
            "workers": {}
        }

    def record(self, agent_id: str, prompt: str, response: str):
        entry = {"prompt": prompt, "response": response}
        if str(agent_id) == self.master_id:
            self.trace["master"].append(entry)
        else:
            self.trace["workers"].setdefault(str(agent_id), []).append(entry)

    def to_json_array(self) -> List[dict]:
        return [self.trace]


# Controller
class SyncParallelController:
    """
    Synchronous rounds controller using the vLLM Runner.

    History completeness invariant:
      - self.agents[aid].history_since_spawn is the single source of truth used to feed the model.
      - It starts from seed_instruction upon (self-)spawn.
      - Every consumed <recv> is stitched into it once (with a single pair of tags).
      - Every assistant output is appended to it verbatim.

    Per round:
      - For each runnable agent, first persist any satisfied <recv> messages into history.
      - Build the prompt solely from history_since_spawn.
      - Generate via vLLM, parse directives, mutate states.
    """

    def __init__(
        self,
        runner: Runner,
        initial_agent_id: str,
        initial_prompt: str,
        tracer: Optional[RunTracer] = None,
    ):
        self.runner = runner
        self.agents: Dict[str, AgentState] = {}

        a0 = AgentState(agent_id=initial_agent_id)
        a0.seed_instruction = initial_prompt.strip()
        a0.history_since_spawn = a0.seed_instruction  # canonical history begins from seed
        a0.messages.append(Message(role="system", content=self.runner.FIXED_SYSTEM))
        a0.messages.append(Message(role="user", content=a0.seed_instruction))
        self.agents[initial_agent_id] = a0
        self.master_id = str(initial_agent_id)

        self.tracer = tracer
        self.final_output: Optional[str] = None
        self.round_idx: int = 0

    def ensure_agent(self, agent_id: str) -> AgentState:
        if agent_id not in self.agents:
            self.agents[agent_id] = AgentState(agent_id=agent_id)
        return self.agents[agent_id]

    def active_agents(self) -> List[AgentState]:
        return [a for a in self.agents.values() if not a.stopped]

    def _build_user_text_for_agent(self, agent: AgentState) -> str:
        """
        Build user text for this round from the canonical history.
        If waiting is satisfied now -> consume messages -> stitch ONCE into history
        (so history stays complete for future rounds too).
        """
        if (agent.waiting_for is not None or agent.wait_any) and agent.can_run():
            lines, ids_hint = agent.consume_recv_lines_and_meta()
            if lines:
                new_hist = _wrap_recv_lines_once(agent.history_since_spawn, lines, ids_hint)
                agent.history_since_spawn = new_hist
                # Optional debug record of the recv block
                agent.events_since_spawn.append(Event(assistant_raw="", recv_block="\n".join(lines)))

        return agent.history_since_spawn

    def run(
        self,
        max_rounds: int = 50,
        max_new_tokens: int = 8192,
    ) -> Optional[str]:
        while self.round_idx < max_rounds and (self.final_output is None) and len(self.active_agents()) > 0:

            ready_agents: List[AgentState] = []
            prompts: List[str] = []
            fed_user_text: Dict[str, str] = {}

            # Build prompts for agents that can run
            for agent in self.active_agents():
                if agent.can_run():
                    user_text = self._build_user_text_for_agent(agent)  # may persist recv into history
                    prompt = self.runner.build_prompt_from_user_text(user_text)
                    ready_agents.append(agent)
                    prompts.append(prompt)
                    fed_user_text[agent.agent_id] = user_text

            if not ready_agents:
                break

            outputs = self.runner.generate(
                prompts,
                max_new_tokens=max_new_tokens,
            )

            # Per-process outputs (optional console log)
            print(f"\n=============== Round {self.round_idx} ===============\n")
            for agent, out_text in zip(ready_agents, outputs):
                print(f"[Agent {agent.agent_id}]")
                print(out_text if out_text else "<empty>")
                print()

            # Record outputs and persist into history
            stops_with_body: List[Tuple[str, str]] = []

            for agent, out_text in zip(ready_agents, outputs):
                # Debug transcript
                agent.messages.append(Message(role="assistant", content=out_text))

                # Store assistant_raw (debug)
                agent.events_since_spawn.append(Event(assistant_raw=out_text, recv_block=None))

                # --- Persist assistant output into canonical history (ensures complete prompt next rounds) ---
                if out_text.strip():
                    if agent.history_since_spawn and not agent.history_since_spawn.endswith("\n"):
                        agent.history_since_spawn += "\n"
                    agent.history_since_spawn += out_text.strip()

            # Parse directives and mutate state
            for agent, out_text in zip(ready_agents, outputs):
                directives = parse_directives(out_text)

                # Track self-spawn
                self_spawned = False

                # Spawns
                for id_list, body in directives.spawns:
                    for sid in id_list:
                        tgt = self.ensure_agent(sid)
                        spawn_body = body.strip()
                        if sid == agent.agent_id:
                            self_spawned = True

                        # Add the "Your id is: {sid}" prefix to the prompt we build after spawn.
                        prefixed_body = add_id_prefix(sid, spawn_body)

                        # Reset per spawn (including canonical history)
                        tgt.events_since_spawn.clear()
                        tgt.seed_instruction = prefixed_body
                        tgt.history_since_spawn = prefixed_body  # reset canonical history to new seed
                        # tgt.inbox.clear()
                        tgt.wait_any = False
                        tgt.waiting_for = None
                        tgt.stopped = False
                        tgt.messages = [
                            Message(role="system", content=self.runner.FIXED_SYSTEM),
                            Message(role="user", content=prefixed_body),
                        ]

                # Sends
                for id_list, body in directives.sends:
                    for rid in id_list:
                        self.ensure_agent(rid).queue_message(src_id=agent.agent_id, content=body)

                # Recv flags (consumption happens when building next prompt; persistence handled there)
                if directives.recv_blocking is not None:
                    agent.waiting_for = directives.recv_blocking
                    agent.wait_any = False
                elif directives.recv_any:
                    agent.wait_any = True
                    agent.waiting_for = None

                # Stop
                if directives.stop_body is not None:
                    if not self_spawned:
                        agent.stopped = True
                        if directives.stop_body.strip():
                            stops_with_body.append((agent.agent_id, directives.stop_body.strip()))
                    # If self-spawned, the new instance remains active

            # Global stop decision
            if len(self.active_agents()) == 0 and len(stops_with_body) > 0:
                self.final_output = stops_with_body[-1][1]
                return self.final_output

            self.round_idx += 1

        return self.final_output


# Eval utilities=
def _to_int_grid(obj: Any) -> Optional[List[List[int]]]:
    """
    Convert a nested structure into a 2D int grid, if possible.
    Returns None if the structure is invalid.
    """
    if not isinstance(obj, list) or not obj:
        return None
    grid: List[List[int]] = []
    for row in obj:
        if not isinstance(row, list) or not row:
            return None
        new_row: List[int] = []
        for v in row:
            try:
                new_row.append(int(v))
            except Exception:
                return None
        grid.append(new_row)
    # Sanity: rectangular
    w = len(grid[0])
    if any(len(r) != w for r in grid):
        return None
    return grid


def parse_grid_from_text(text: str) -> Optional[List[List[int]]]:
    """
    Try to parse the assistant's final stop body into a 2D int grid.
    Strategy:
      1) Direct json.loads(text)
      2) json.loads of the largest [...] slice
      3) ast.literal_eval on the largest [...] slice
    Returns None if all fail.
    """
    text = (text or "").strip()
    # 1) Try direct JSON
    try:
        obj = json.loads(text)
        grid = _to_int_grid(obj)
        if grid is not None:
            return grid
    except Exception:
        pass

    # Find the outermost [ ... ] slice
    first = text.find("[")
    last = text.rfind("]")
    if first != -1 and last != -1 and last > first:
        inner = text[first:last+1].strip()
        # 2) JSON on slice
        try:
            obj = json.loads(inner)
            grid = _to_int_grid(obj)
            if grid is not None:
                return grid
        except Exception:
            pass
        # 3) literal_eval on slice (tolerates single quotes, trailing spaces)
        try:
            obj = ast.literal_eval(inner)
            grid = _to_int_grid(obj)
            if grid is not None:
                return grid
        except Exception:
            pass

    return None


def grids_equal(a: List[List[int]], b: List[List[int]]) -> bool:
    """Strict equality: same shape and every cell equal."""
    if a is None or b is None:
        return False
    if len(a) != len(b):
        return False
    if any(len(ra) != len(rb) for ra, rb in zip(a, b)):
        return False
    for ra, rb in zip(a, b):
        for va, vb in zip(ra, rb):
            if int(va) != int(vb):
                return False
    return True


def read_prompts_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Read JSONL lines of {"sudoku_id": ..., "initial_prompt": "...", "solution": [[...], ...]}.
    Keep ordering as-is.
    """
    tasks: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Minimal validation
            if "initial_prompt" not in obj:
                continue
            tasks.append(obj)
    return tasks


# Batch evaluator
def evaluate_batch(
    prompts_file: str,
    runner: Runner,
    initial_agent_id: str,
    max_rounds: int,
    max_new_tokens: int,
    max_tasks: int = -1,
    write_trace: bool = False,
    trace_dir: Optional[str] = None,
) -> Tuple[int, int, float]:
    """
    Run all prompts sequentially and compute:
      - num_total: how many tasks processed
      - num_correct: strict matches against provided solution
      - avg_time_sec: average wall time per *successful* task only
    """
    tasks = read_prompts_jsonl(prompts_file)
    if max_tasks is not None and max_tasks > 0:
        tasks = tasks[:max_tasks]

    dur_success_list: List[float] = []   # collect durations only for successful solves
    num_total = 0
    num_correct = 0

    for idx, rec in enumerate(tasks):
        sid = rec.get("sudoku_id", None)
        init_prompt = rec.get("initial_prompt", "")
        sol_obj = rec.get("solution", None)
        gold = _to_int_grid(sol_obj) if sol_obj is not None else None

        print("\n" + "=" * 80)
        print(f"[Task {idx}] sudoku_id={sid}")
        print("=" * 80)

        tracer = RunTracer(
            master_id=str(initial_agent_id),
            system_prompt_master=runner.FIXED_SYSTEM,
            system_prompt_worker=runner.FIXED_SYSTEM,
        ) if write_trace else None

        controller = SyncParallelController(
            runner=runner,
            initial_agent_id=str(initial_agent_id),
            initial_prompt=str(init_prompt),
            tracer=tracer,
        )

        t0 = time.perf_counter()
        final_text = controller.run(max_rounds=max_rounds, max_new_tokens=max_new_tokens)
        t1 = time.perf_counter()
        dur = t1 - t0

        pred = parse_grid_from_text(final_text or "")
        ok = (gold is not None) and grids_equal(pred, gold)

        print(f"[Result] solved={'YES' if ok else 'NO '} | time={dur:.3f}s")
        if ok:
            num_correct += 1
            dur_success_list.append(dur)   # <-- only count time if solved
        num_total += 1

        # Optional trace dump
        if write_trace and tracer is not None and trace_dir:
            try:
                import os
                os.makedirs(trace_dir, exist_ok=True)
                tag = f"{sid}" if sid is not None else f"idx{idx}"
                outp = f"{trace_dir}/trace_{tag}.json"
                with open(outp, "w", encoding="utf-8") as f:
                    json.dump(tracer.to_json_array(), f, ensure_ascii=False, indent=2)
                print(f"[Trace] wrote: {outp}")
            except Exception as e:
                print(f"[Trace] failed to write: {e}")

    avg_time_success = statistics.mean(dur_success_list) if dur_success_list else 0.0
    return num_total, num_correct, avg_time_success

# CLI
def main():
    global VLLM_BASE_URL, VLLM_API_KEY, VLLM_MODEL_NAME, VLLM_MAX_WORKERS
    ap = argparse.ArgumentParser(description="Batch-evaluate Sudoku prompts JSONL with strict solution matching.")
    # Model/template
    ap.add_argument("--model", required=True, type=str,
        help="HF model path or id (for tokenizer and local chat template)")
    ap.add_argument("--template_path", type=str, default=None,
        help="Path to local chat_template.jinja (default: <model>/chat_template.jinja if present)")

    # Eval data
    ap.add_argument("--prompts_jsonl", required=True, type=str,
        help="Path to JSONL lines file with fields: sudoku_id, initial_prompt, solution")

    # vLLM connection
    ap.add_argument("--vllm-url", dest="vllm_url", type=str, default=VLLM_BASE_URL,
        help="Base URL of the OpenAI-compatible vLLM server (no /v1 suffix).")
    ap.add_argument("--api-key", dest="api_key", type=str, default=VLLM_API_KEY,
        help="API key for the vLLM server.")
    ap.add_argument("--served-model-name", dest="served_model_name", type=str, default=VLLM_MODEL_NAME,
        help="Model name registered on the vLLM server (its --served-model-name).")
    ap.add_argument("--max-workers", dest="max_workers", type=int, default=VLLM_MAX_WORKERS,
        help="Max concurrent in-flight vLLM requests.")

    # Controller/gen limits
    ap.add_argument("--initial_agent_id", type=str, default="0")
    ap.add_argument("--max_rounds", type=int, default=500)
    ap.add_argument("--max_new_tokens", type=int, default=15000)
    ap.add_argument("--max_tasks", type=int, default=-1,
        help="Process at most N tasks (debug). -1 = all.")

    # Optional traces
    ap.add_argument("--write_trace", action="store_true", help="Write per-task trace JSONs")
    ap.add_argument("--trace_dir", type=str, default="./traces", help="Directory for traces if enabled")

    args = ap.parse_args()

    VLLM_BASE_URL = args.vllm_url
    VLLM_API_KEY = args.api_key
    VLLM_MODEL_NAME = args.served_model_name
    VLLM_MAX_WORKERS = args.max_workers

    template_path = args.template_path
    if template_path is None:
        candidate = os.path.join(args.model, "chat_template.jinja")
        template_path = candidate if os.path.exists(candidate) else None

    runner = Runner(model_name_or_path=args.model, template_path=template_path)

    total, correct, avg_time = evaluate_batch(
        prompts_file=args.prompts_jsonl,
        runner=runner,
        initial_agent_id=args.initial_agent_id,
        max_rounds=args.max_rounds,
        max_new_tokens=args.max_new_tokens,
        max_tasks=args.max_tasks,
        write_trace=args.write_trace,
        trace_dir=args.trace_dir,
    )

    acc = (correct / total) if total > 0 else 0.0
    print("\n" + "=" * 60)
    print(f"Total: {total}")
    print(f"Correct (strict match): {correct}")
    print(f"Accuracy: {acc * 100:.2f}%")
    avg_label = f"{avg_time:.3f}s" if correct > 0 else "N/A (no solved tasks)"
    print(f"Average time (success only): {avg_label}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
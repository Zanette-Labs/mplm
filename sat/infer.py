#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_sat.py

Low-overhead async event-driven SAT evaluator.

Key properties:
- Uses asyncio + aiohttp instead of per-agent mp.Process (no pickle/spawn overhead).
- Event-driven scheduling: whichever agent request finishes first is processed immediately.
- No round barrier.
- <stop> marks descendant agents stopped and cancels their client-side asyncio tasks,
  so no further unnecessary requests are scheduled for them.

Important semantic note:
- asyncio task cancellation stops the client-side wait and discards the result.
- It may not abort computation already accepted by the vLLM server.
- The goal here is to stop sending future unnecessary requests on preemption,
  not hard server-side termination of in-flight generations.
"""

import re
import json
import os
import time
import argparse
import statistics
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set

import aiohttp
from transformers import AutoTokenizer


# vLLM client config (override via CLI flags or environment variables)
VLLM_BASE_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8000")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
VLLM_MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "model")
VLLM_MAX_WORKERS = int(os.environ.get("VLLM_MAX_WORKERS", "625"))

VLLM_TIMEOUT = 120
VLLM_RETRIES = 3


# ---------------------------
# Tag parsing helpers
# ---------------------------

TAG_SPAWN_RE = re.compile(
    r"<spawn\s*\[(?P<ids>[^\]]+)\]\s*>(?P<body>.*?)</spawn(?:\s*\[(?P=ids)\])?\s*>",
    re.DOTALL | re.IGNORECASE,
)

TAG_SEND_RE = re.compile(
    r"<send\s*\[(?P<ids>[^\]]+)\]\s*>(?P<body>.*?)</send(?:\s*\[(?P=ids)\])?\s*>",
    re.DOTALL | re.IGNORECASE,
)

TAG_RECV_BLOCKING_RE = re.compile(
    r"<recv\s*\[(?P<ids>[^\]]+)\]\s*>",
    re.IGNORECASE,
)
TAG_RECV_ANY_RE = re.compile(
    r"<recv\s*>",
    re.IGNORECASE,
)

TAG_STOP_RE = re.compile(
    r"(?:<stop\s*>(?P<body>.*?)</stop\s*>)|(?:<stop\s*>)",
    re.DOTALL | re.IGNORECASE,
)

OPEN_RECV_RE = re.compile(
    r"<recv\s*(?:\[(?P<ids>[^\]]+)\])?\s*>",
    re.IGNORECASE,
)
CLOSE_RECV_RE = re.compile(
    r"</recv\s*(?:\[(?P<close_ids>[^\]]+)\])?\s*>",
    re.IGNORECASE,
)


def parse_id_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def add_id_prefix(agent_id: str, body: str) -> str:
    prefix = f"Your id is: {agent_id}."
    if body.lstrip().startswith(prefix):
        return body
    return f"{prefix}\n{body}"


# ---------------------------
# Directive parsing
# ---------------------------

@dataclass
class ParsedDirectives:
    spawns: List[Tuple[List[str], str]] = field(default_factory=list)
    sends: List[Tuple[List[str], str]] = field(default_factory=list)
    recv_blocking: Optional[List[str]] = None
    recv_any: bool = False
    stop_body: Optional[str] = None  # "" means bare stop


def parse_directives(text: str) -> ParsedDirectives:
    result = ParsedDirectives()

    for m in TAG_SPAWN_RE.finditer(text or ""):
        result.spawns.append((
            parse_id_list(m.group("ids")),
            (m.group("body") or "").strip(),
        ))

    for m in TAG_SEND_RE.finditer(text or ""):
        result.sends.append((
            parse_id_list(m.group("ids")),
            (m.group("body") or "").strip(),
        ))

    m_block = TAG_RECV_BLOCKING_RE.search(text or "")
    m_any = TAG_RECV_ANY_RE.search(text or "")
    if m_block and m_any:
        result.recv_blocking = parse_id_list(m_block.group("ids"))
    elif m_block:
        result.recv_blocking = parse_id_list(m_block.group("ids"))
    elif m_any:
        result.recv_any = True

    m_stop = TAG_STOP_RE.search(text or "")
    if m_stop:
        body = m_stop.groupdict().get("body")
        result.stop_body = body.strip() if body else ""

    return result


# ---------------------------
# Data structures
# ---------------------------

@dataclass
class Message:
    role: str
    content: str


@dataclass
class QueuedMsg:
    src_id: str
    content: str


@dataclass
class Event:
    assistant_raw: str = ""
    recv_block: Optional[str] = None


@dataclass
class AgentState:
    agent_id: str

    messages: List[Message] = field(default_factory=list)

    inbox: List[QueuedMsg] = field(default_factory=list)
    waiting_for: Optional[List[str]] = None
    wait_any: bool = False
    stopped: bool = False

    seed_instruction: Optional[str] = None
    events_since_spawn: List[Event] = field(default_factory=list)
    history_since_spawn: str = ""

    failed_reason: Optional[str] = None

    def can_run(self) -> bool:
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
        txt = (q.content or "").strip()
        if txt.startswith("{") and txt.endswith("}"):
            return txt
        return f"{{ From: {q.src_id}, {txt} }}"

    def consume_recv_lines_and_meta(self) -> Tuple[List[str], Optional[List[str]]]:
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

        if self.wait_any and self.inbox:
            lines = [self._format_recv_line(m) for m in self.inbox]
            self.inbox.clear()
            self.wait_any = False
            return lines, None

        return [], None

    def queue_message(self, src_id: str, content: str):
        self.inbox.append(QueuedMsg(src_id=src_id, content=content))


# ---------------------------
# <recv> stitching helpers
# ---------------------------

def _detect_unclosed_recv_suffix(text: str) -> Optional[Tuple[str, Optional[str]]]:
    last_open = None
    for m in OPEN_RECV_RE.finditer(text or ""):
        last_open = m
    last_close = None
    for m in CLOSE_RECV_RE.finditer(text or ""):
        last_close = m

    if last_open is None:
        return None
    if last_close is None or last_open.end() > last_close.end():
        ids_str = last_open.group("ids")
        opener = f"<recv [{ids_str}]>" if ids_str else "<recv>"
        return opener, ids_str
    return None


def _wrap_recv_lines_once(base: str, lines: List[str], ids_hint: Optional[List[str]]) -> str:
    if not lines:
        return base

    suffix_newline = "\n" if base and not base.endswith("\n") else ""
    body = "\n".join(lines) + "\n"

    dangling = _detect_unclosed_recv_suffix(base)
    if dangling:
        close_tag = "</recv>"
        return f"{base}{suffix_newline}{body}{close_tag}"

    if ids_hint:
        ids_compact = ",".join([s.strip() for s in ids_hint if s.strip()])
        open_tag = f"<recv [{ids_compact}]>"
        close_tag = "</recv>"
    else:
        open_tag = "<recv>"
        close_tag = "</recv>"
    return f"{base}{suffix_newline}{open_tag}\n{body}{close_tag}"


# ---------------------------
# Runner: prompt building only
# ---------------------------

class Runner:
    FIXED_SYSTEM = "You are a helpful assistant."

    def __init__(self, model_name_or_path: str, template_path: Optional[str]):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.tokenizer.padding_side = "left"
        if template_path:
            with open(template_path, "r", encoding="utf-8") as f:
                self.tokenizer.chat_template = f.read()

        self.base_url = VLLM_BASE_URL
        self.api_key = VLLM_API_KEY
        self.model_name = VLLM_MODEL_NAME

        self.max_workers = VLLM_MAX_WORKERS
        self.timeout_sec = float(VLLM_TIMEOUT)
        self.max_retries = int(VLLM_RETRIES)

    def build_prompt_from_user_text(self, user_text: str) -> str:
        chat = [
            {"role": "system", "content": self.FIXED_SYSTEM},
            {"role": "user", "content": user_text},
        ]
        return self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
        )


# ---------------------------
# Async vLLM call
# ---------------------------

async def _llm_completion_call_async(
    session: aiohttp.ClientSession,
    model_name: str,
    timeout_sec: float,
    max_retries: int,
    prompt: str,
    max_new_tokens: int,
) -> Optional[str]:
    payload_common = {
        "model": model_name,
        "max_tokens": int(max_new_tokens),
        "n": 1,
        "stream": False,
        "temperature": 0,
    }

    for attempt in range(1, int(max_retries) + 1):
        try:
            payload = dict(payload_common)
            payload["prompt"] = prompt

            async with session.post(
                "/v1/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout_sec),
            ) as resp:
                if resp.status == 200:
                    try:
                        data = await resp.json()
                    except Exception:
                        await asyncio.sleep(0.2 * attempt)
                        continue

                    text = (data.get("choices", [{}])[0].get("text", "") or "").strip()
                    return text

        except asyncio.CancelledError:
            raise
        except Exception:
            pass

        await asyncio.sleep(0.2 * attempt)

    return None


# ----------------------------
# Async event-driven controller
# ----------------------------

@dataclass
class AsyncJobInfo:
    agent_id: str
    fed_user_text: str


class AsyncController:
    """
    Low-overhead event-driven async controller.

    Scheduling semantics:
    - Ready agents are scheduled immediately up to max_concurrency.
    - The controller waits for FIRST_COMPLETED.
    - Finished agent output is processed immediately.
    - New ready agents are scheduled immediately after each completion batch.

    Stop semantics:
    - Master <stop> sets final output and cancels all client-side pending tasks.
    - Non-master <stop> cancels descendant client-side tasks and marks descendants stopped,
      so no further unnecessary requests are scheduled for that subtree.
    - Cancelled requests may still have reached the vLLM server; their results are ignored.
    """

    def __init__(
        self,
        runner: Runner,
        initial_agent_id: str,
        initial_prompt: str,
    ):
        self.runner = runner

        self.agents: Dict[str, AgentState] = {}

        a0 = AgentState(agent_id=initial_agent_id)
        a0.seed_instruction = (initial_prompt or "").strip()
        a0.history_since_spawn = a0.seed_instruction
        a0.messages.append(Message(role="system", content=self.runner.FIXED_SYSTEM))
        a0.messages.append(Message(role="user", content=a0.seed_instruction))
        self.agents[initial_agent_id] = a0

        self.master_id = str(initial_agent_id)
        self.final_output: Optional[str] = None

        self.max_steps: int = 0
        self.max_new_tokens: int = 0
        self.steps_done: int = 0

        self.stops_with_body: List[Tuple[str, str]] = []
        self.task_errors: List[str] = []
        self.children_map: Dict[str, Set[str]] = {}

        # asyncio.Task -> metadata
        self.agent_tasks: Dict[asyncio.Task, AsyncJobInfo] = {}

        # agent_id -> asyncio.Task
        self.agent_to_task: Dict[str, asyncio.Task] = {}

        # For HTTP-bound vLLM calls, this can be much larger than CPU count.
        self.max_concurrency = max(1, int(VLLM_MAX_WORKERS))

    def ensure_agent(self, agent_id: str) -> AgentState:
        if agent_id not in self.agents:
            self.agents[agent_id] = AgentState(agent_id=agent_id)
        return self.agents[agent_id]

    def active_agents(self) -> List[AgentState]:
        return [a for a in self.agents.values() if not a.stopped]

    def _build_user_text_for_agent(self, agent: AgentState) -> str:
        if (agent.waiting_for is not None or agent.wait_any) and agent.can_run():
            lines, ids_hint = agent.consume_recv_lines_and_meta()
            if lines:
                agent.history_since_spawn = _wrap_recv_lines_once(
                    agent.history_since_spawn,
                    lines,
                    ids_hint,
                )
                agent.events_since_spawn.append(
                    Event(assistant_raw="", recv_block="\n".join(lines))
                )
        return agent.history_since_spawn

    def _collect_descendants(self, agent_id: str) -> List[str]:
        stack = list(self.children_map.get(agent_id, set()))
        seen: Set[str] = set()
        out: List[str] = []

        while stack:
            cid = stack.pop()
            if cid in seen:
                continue
            seen.add(cid)
            out.append(cid)

            for gc in self.children_map.get(cid, set()):
                if gc not in seen:
                    stack.append(gc)

        return out

    def _cancel_agent_task_if_running(self, agent_id: str):
        task = self.agent_to_task.get(agent_id)
        if task is not None and not task.done():
            task.cancel()

    def _cancel_descendants_and_stop_agents(self, agent_id: str, reason: str):
        desc = self._collect_descendants(agent_id)

        for cid in desc:
            self._cancel_agent_task_if_running(cid)

            a = self.agents.get(cid)
            if a is not None and not a.stopped:
                a.stopped = True
                a.failed_reason = a.failed_reason or reason
                a.wait_any = False
                a.waiting_for = None
                a.inbox.clear()

    def _cancel_all_running_tasks(self):
        for task in list(self.agent_tasks.keys()):
            if not task.done():
                task.cancel()

    def _mark_agent_failed(self, agent: AgentState, reason: str):
        agent.failed_reason = reason
        agent.stopped = True
        self.task_errors.append(f"agent {agent.agent_id} failed: {reason}")
        self._cancel_descendants_and_stop_agents(
            agent.agent_id,
            reason=f"ancestor {agent.agent_id} failed",
        )

    async def _call_one_async(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
    ) -> Optional[str]:
        return await _llm_completion_call_async(
            session=session,
            model_name=self.runner.model_name,
            timeout_sec=self.runner.timeout_sec,
            max_retries=self.runner.max_retries,
            prompt=prompt,
            max_new_tokens=self.max_new_tokens,
        )

    def _schedule_ready_agents(self, session: aiohttp.ClientSession):
        if self.max_steps > 0 and self.steps_done >= self.max_steps:
            return

        if len(self.agent_tasks) >= self.max_concurrency:
            return

        for agent in self.active_agents():
            if self.max_steps > 0 and self.steps_done >= self.max_steps:
                break

            if len(self.agent_tasks) >= self.max_concurrency:
                break

            existing = self.agent_to_task.get(agent.agent_id)
            if existing is not None and not existing.done():
                continue

            if not agent.can_run():
                continue

            user_text = self._build_user_text_for_agent(agent)
            prompt = self.runner.build_prompt_from_user_text(user_text)

            task = asyncio.create_task(self._call_one_async(session=session, prompt=prompt))

            info = AsyncJobInfo(
                agent_id=agent.agent_id,
                fed_user_text=user_text,
            )

            self.agent_tasks[task] = info
            self.agent_to_task[agent.agent_id] = task
            self.steps_done += 1

    async def _process_finished_task(self, task: asyncio.Task):
        info = self.agent_tasks.pop(task, None)
        if info is None:
            return

        self.agent_to_task.pop(info.agent_id, None)

        agent = self.agents.get(info.agent_id)
        if agent is None:
            return

        if agent.stopped:
            return

        try:
            out_text = await task
        except asyncio.CancelledError:
            return
        except Exception as e:
            out_text = None
            self.task_errors.append(f"agent {info.agent_id} async error: {e}")

        if out_text is None:
            self._mark_agent_failed(agent, "vLLM completion returned None")
            return

        out_text = (out_text or "").strip()

        agent.messages.append(Message(role="assistant", content=out_text))
        agent.events_since_spawn.append(Event(assistant_raw=out_text, recv_block=None))

        if out_text:
            if agent.history_since_spawn and not agent.history_since_spawn.endswith("\n"):
                agent.history_since_spawn += "\n"
            agent.history_since_spawn += out_text

        directives = parse_directives(out_text)
        self._apply_directives(agent, out_text, info.fed_user_text, directives)

    def _apply_directives(
        self,
        agent: AgentState,
        out_text: str,
        fed_user_text: str,
        directives: ParsedDirectives,
    ):
        self_spawned = False

        # Spawns
        spawned_children_this_step: Set[str] = set()

        for id_list, body in directives.spawns:
            for sid in id_list:
                spawned_children_this_step.add(sid)

                tgt = self.ensure_agent(sid)
                spawn_body = (body or "").strip()

                if sid == agent.agent_id:
                    self_spawned = True

                prefixed_body = add_id_prefix(sid, spawn_body)

                tgt.events_since_spawn.clear()
                tgt.seed_instruction = prefixed_body
                tgt.history_since_spawn = prefixed_body
                tgt.wait_any = False
                tgt.waiting_for = None
                tgt.stopped = False
                tgt.failed_reason = None
                tgt.inbox.clear()
                tgt.messages = [
                    Message(role="system", content=self.runner.FIXED_SYSTEM),
                    Message(role="user", content=prefixed_body),
                ]

        if spawned_children_this_step:
            self.children_map.setdefault(agent.agent_id, set()).update(
                spawned_children_this_step
            )

        # Sends
        for id_list, body in directives.sends:
            for rid in id_list:
                r_agent = self.ensure_agent(rid)
                if r_agent.stopped:
                    continue
                r_agent.queue_message(src_id=agent.agent_id, content=body)

        # Recv flags
        if directives.recv_blocking is not None:
            agent.waiting_for = directives.recv_blocking
            agent.wait_any = False
        elif directives.recv_any:
            agent.wait_any = True
            agent.waiting_for = None

        # Stop
        if directives.stop_body is not None and not self_spawned:
            agent.stopped = True

            if agent.agent_id == self.master_id:
                self.final_output = (directives.stop_body or "").strip()
                self._cancel_all_running_tasks()
                return

            self._cancel_descendants_and_stop_agents(
                agent.agent_id,
                reason=f"ancestor {agent.agent_id} stopped via <stop>",
            )

            if (directives.stop_body or "").strip():
                self.stops_with_body.append(
                    (agent.agent_id, directives.stop_body.strip())
                )

    async def run_async(
        self,
        max_steps: int = 50,
        max_new_tokens: int = 8192,
        verbose: bool = False,
    ) -> Optional[str]:
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.steps_done = 0
        self.stops_with_body.clear()
        self.final_output = None
        self.agent_tasks.clear()
        self.agent_to_task.clear()
        self.task_errors.clear()
        self.children_map.clear()

        connector = aiohttp.TCPConnector(
            limit=self.max_concurrency,
            limit_per_host=self.max_concurrency,
            force_close=False,
            enable_cleanup_closed=True,
        )

        headers = {
            "Authorization": f"Bearer {self.runner.api_key}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession(
            base_url=self.runner.base_url,
            headers=headers,
            connector=connector,
        ) as session:
            try:
                self._schedule_ready_agents(session)

                while True:
                    if self.final_output is not None:
                        break

                    if self.max_steps > 0 and self.steps_done >= self.max_steps and not self.agent_tasks:
                        break

                    if not self.active_agents() and not self.agent_tasks:
                        break

                    if not self.agent_tasks:
                        self._schedule_ready_agents(session)
                        if not self.agent_tasks:
                            break

                    done, _ = await asyncio.wait(
                        list(self.agent_tasks.keys()),
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    for task in done:
                        await self._process_finished_task(task)

                        if self.final_output is not None:
                            break

                    if self.final_output is not None:
                        break

                    if not self.active_agents() and self.stops_with_body and self.final_output is None:
                        self.final_output = self.stops_with_body[-1][1]
                        break

                    self._schedule_ready_agents(session)

            finally:
                self._cancel_all_running_tasks()

                if self.agent_tasks:
                    await asyncio.gather(
                        *list(self.agent_tasks.keys()),
                        return_exceptions=True,
                    )

        return self.final_output

    def run(
        self,
        max_steps: int = 50,
        max_new_tokens: int = 8192,
        verbose: bool = False,
    ) -> Optional[str]:
        return asyncio.run(
            self.run_async(
                max_steps=max_steps,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
            )
        )


# Keep old name so existing evaluation code can stay mostly unchanged.
MPController = AsyncController


# ---------------------------
# SAT stop parsing + scoring
# ---------------------------

MODEL_PAIR_RE = re.compile(
    r"x\s*(\d+)\s*=\s*(true|false|True|False)",
    re.IGNORECASE,
)


def parse_master_stop_sat(stop_body: str) -> Tuple[Optional[bool], Dict[int, bool]]:
    text = (stop_body or "").strip()
    if not text:
        return None, {}

    up = text.upper()
    if "UNSAT" in up:
        return False, {}

    if "SAT" in up:
        assign: Dict[int, bool] = {}
        for var_s, val_s in MODEL_PAIR_RE.findall(text):
            v = int(var_s)
            assign[v] = val_s.lower() == "true"
        return True, assign

    return None, {}


def clause_definitely_false(clause: List[int], assign: Dict[int, bool]) -> bool:
    all_assigned = True
    any_sat = False
    for lit in clause:
        var = abs(int(lit))
        if var not in assign:
            all_assigned = False
            continue
        val = assign[var]
        sat = (lit > 0 and val) or (lit < 0 and not val)
        if sat:
            any_sat = True
    return all_assigned and not any_sat


def model_is_consistent_with_clauses(clauses: List[List[int]], assign: Dict[int, bool]) -> bool:
    for c in clauses:
        if clause_definitely_false(c, assign):
            return False
    return True


# ---------------------------
# Data IO
# ---------------------------

def read_sat_jsonl(path: str) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "problem" not in obj or "answer" not in obj:
                continue
            tasks.append(obj)
    return tasks


# ---------------------------
# Batch evaluator
# ---------------------------

def evaluate_batch(
    prompts_file: str,
    runner: Runner,
    initial_agent_id: str,
    max_steps: int,
    max_new_tokens: int,
    max_tasks: int = -1,
    verbose_steps: bool = False,
) -> Tuple[int, int, float, float]:
    tasks = read_sat_jsonl(prompts_file)
    if max_tasks is not None and max_tasks > 0:
        tasks = tasks[:max_tasks]

    num_total = 0
    num_correct = 0
    dur_all: List[float] = []
    dur_correct: List[float] = []

    for idx, rec in enumerate(tasks):
        sid = rec.get("id", None)
        problem = str(rec.get("problem", "")).strip()
        answer = bool(rec.get("answer", False))

        print("\n" + "=" * 80, flush=True)
        print(f"[Task {idx}] id={sid} | answer={answer}", flush=True)
        print("=" * 80, flush=True)

        controller = MPController(
            runner=runner,
            initial_agent_id=str(initial_agent_id),
            initial_prompt=problem,
        )

        t0 = time.perf_counter()
        final_stop_body = controller.run(
            max_steps=max_steps,
            max_new_tokens=max_new_tokens,
            verbose=verbose_steps,
        )
        t1 = time.perf_counter()
        dur = t1 - t0
        dur_all.append(dur)

        if final_stop_body is None:
            print("[Task] FAILED (no final output).", flush=True)
            if controller.task_errors:
                print("[Task] errors:", flush=True)
                for e in controller.task_errors[:10]:
                    print(f"  - {e}", flush=True)
                if len(controller.task_errors) > 10:
                    print(f"  - ... ({len(controller.task_errors) - 10} more)", flush=True)
            print(f"[Result] correct=NO  | time={dur:.3f}s", flush=True)
            num_total += 1
            print(f"[Progress] {num_correct}/{num_total} correct so far", flush=True)
            continue

        sat_flag, assign = parse_master_stop_sat(final_stop_body or "")

        if answer is False:
            ok = sat_flag is False
        else:
            ok = sat_flag is True

        print(f"[Master <stop>] {final_stop_body!r}", flush=True)
        print(f"[Parsed] sat_flag={sat_flag} | #assign={len(assign)} | time={dur:.3f}s", flush=True)
        print(f"[Result] correct={'YES' if ok else 'NO '}", flush=True)
        if ok:
            num_correct += 1
            dur_correct.append(dur)
        num_total += 1
        print(f"[Progress] {num_correct}/{num_total} correct so far", flush=True)

    avg_time_all = statistics.mean(dur_all) if dur_all else 0.0
    avg_time_correct = statistics.mean(dur_correct) if dur_correct else 0.0
    return num_total, num_correct, avg_time_all, avg_time_correct


# ---------------------------
# CLI
# ---------------------------

def main():
    global VLLM_BASE_URL, VLLM_API_KEY, VLLM_MODEL_NAME, VLLM_MAX_WORKERS
    ap = argparse.ArgumentParser(
        description="Async event-driven batch SAT evaluator (master <stop> SAT/UNSAT scoring)."
    )
    ap.add_argument("--model", required=True, type=str,
                    help="HF model path or id (for tokenizer and local chat template).")
    ap.add_argument("--template_path", type=str, default=None,
                    help="Path to chat_template.jinja (default: <model>/chat_template.jinja if present).")
    ap.add_argument("--prompts_jsonl", required=True, type=str,
                    help="Path to SAT instance JSONL with fields: problem, answer (id optional).")
    ap.add_argument("--initial_agent_id", type=str, default="0")
    ap.add_argument("--vllm-url", dest="vllm_url", type=str, default=VLLM_BASE_URL,
                    help="Base URL of the OpenAI-compatible vLLM server (no /v1 suffix).")
    ap.add_argument("--api-key", dest="api_key", type=str, default=VLLM_API_KEY,
                    help="API key for the vLLM server.")
    ap.add_argument("--served-model-name", dest="served_model_name", type=str, default=VLLM_MODEL_NAME,
                    help="Model name registered on the vLLM server (its --served-model-name).")
    ap.add_argument("--max_steps", type=int, default=1000000)
    ap.add_argument("--max_new_tokens", type=int, default=25000)
    ap.add_argument("--max_tasks", type=int, default=-1,
                    help="Process at most N tasks. -1 = all.")
    ap.add_argument(
        "--max_workers",
        type=int,
        default=VLLM_MAX_WORKERS,
        help="Maximum concurrent in-flight vLLM HTTP requests.",
    )
    ap.add_argument("--verbose_steps", action="store_true")
    args = ap.parse_args()

    VLLM_BASE_URL = args.vllm_url
    VLLM_API_KEY = args.api_key
    VLLM_MODEL_NAME = args.served_model_name
    VLLM_MAX_WORKERS = int(args.max_workers)

    template_path = args.template_path or os.path.join(args.model, "chat_template.jinja")
    if not os.path.exists(template_path):
        template_path = None

    runner = Runner(model_name_or_path=args.model, template_path=template_path)
    runner.base_url = VLLM_BASE_URL
    runner.max_workers = VLLM_MAX_WORKERS

    print(f"Model:       {args.model}")
    print(f"vLLM:        {VLLM_BASE_URL}")
    print(f"Max workers: {VLLM_MAX_WORKERS}")
    print(f"Prompts:     {args.prompts_jsonl}")

    total, correct, avg_all, avg_corr = evaluate_batch(
        prompts_file=args.prompts_jsonl,
        runner=runner,
        initial_agent_id=args.initial_agent_id,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        max_tasks=args.max_tasks,
        verbose_steps=args.verbose_steps,
    )

    acc = (correct / total) if total > 0 else 0.0
    print("\n" + "=" * 60)
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Average time (all): {avg_all:.3f}s")
    if correct > 0:
        print(f"Average time (correct only): {avg_corr:.3f}s")
    else:
        print("Average time (correct only): N/A (no correct tasks)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Trace：把一次 episode 变成可回放、可归因的 JSONL。

一条 trace 至少包含：
  episode_start — task/spec/prompt/工具集合
  llm_call      — 每一步发给模型的 messages、模型返回、usage、延迟
  tool_call     — 工具名、参数、成功/失败、重试次数、错误
  repair_hint   — harness 注入的修复提示（如果配置开启）
  final         — 最终答案与停止原因
  episode_end   — 通过/失败、失败分类、总耗时

为什么存完整 messages：
  失败归因不能只看最后一句。要知道模型是「没读文件」「读错文件」还是
  「读到了但算错」，必须能看到每一步的上下文。
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Iterator


def new_episode_id() -> str:
    """短 id，方便在文件树和报告里定位。"""
    return uuid.uuid4().hex[:12]


def new_run_id() -> str:
    """按时间排序的 run id，例如 20261115-173000。"""
    return time.strftime("%Y%m%d-%H%M%S")


def messages_hash(messages: list[dict]) -> str:
    """对一次请求的 messages 做稳定哈希。

    回放时用它检测「代码/工具/提示词已经变了，历史响应不再适用」。
    只取 16 位，足够定位，不会把 trace 撑大。
    """
    payload = json.dumps(messages, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class TraceWriter:
    """按 episode 写 JSONL；每写一行立即 flush，进程崩了也能保留现场。"""

    def __init__(
        self,
        path: str | Path,
        *,
        episode_id: str,
        task_id: str,
        category: str,
        prompt: str,
        spec: Any,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.episode_id = episode_id
        self.task_id = task_id
        self.category = category
        self._step = 0
        self._fh = self.path.open("w", encoding="utf-8")
        self.write(
            "episode_start",
            prompt=prompt,
            spec=spec.to_dict() if hasattr(spec, "to_dict") else spec,
        )

    @property
    def step(self) -> int:
        return self._step

    def next_step(self) -> int:
        self._step += 1
        return self._step

    def write(self, event: str, **fields: Any) -> None:
        record = {
            "event": event,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "category": self.category,
            "step": self._step,
            "ts": round(time.time(), 4),
            **fields,
        }
        self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def read_trace(path: str | Path) -> list[dict]:
    """读取 JSONL；坏行跳过并保留其余事件，避免一条脏数据毁掉整次分析。"""
    events: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def iter_events(events: list[dict], event_type: str) -> Iterator[dict]:
    for event in events:
        if event.get("event") == event_type:
            yield event


def trace_stats(events: list[dict]) -> dict:
    """从 trace 里提取失败归因需要的计数。

    这些计数不依赖 evaluator 的最终结果，回放时也能独立算出来：
      llm_calls / tool_calls / tool_errors / repair_hints
      repeated_failed_actions — 同一工具 + 同一参数连续失败 ≥ 2 次
    """
    llm_calls = 0
    tool_calls = 0
    tool_errors = 0
    repair_hints = 0
    failed_action_counts: dict[str, int] = {}

    for event in events:
        kind = event.get("event")
        if kind == "llm_call":
            llm_calls += 1
        elif kind == "tool_call":
            tool_calls += 1
            attempts = int(event.get("failed_attempts", 0) or 0)
            tool_errors += attempts
            if attempts > 0 and not event.get("ok", False):
                key = f"{event.get('name')}::{json.dumps(event.get('args', {}), sort_keys=True, ensure_ascii=False)}"
                failed_action_counts[key] = failed_action_counts.get(key, 0) + 1
        elif kind == "repair_hint":
            repair_hints += 1

    repeated_failed_actions = sum(1 for count in failed_action_counts.values() if count >= 2)
    return {
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "tool_errors": tool_errors,
        "repair_hints": repair_hints,
        "repeated_failed_actions": repeated_failed_actions,
    }

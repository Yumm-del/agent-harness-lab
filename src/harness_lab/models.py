# -*- coding: utf-8 -*-
"""跨模块共享的数据结构。

为什么要单独放一个 models.py：
  runner 产生结果、evaluate 做聚合、report 渲染、replay 读取 trace，
  这些模块都要用同一套字段。集中定义可以避免「每个模块各写一份 dict」，
  也方便最后统一序列化成 JSON。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Usage:
    """一次或多次 LLM 调用的 token 用量。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0

    def add(self, other: "Usage") -> None:
        """把另一份 usage 累加进来（原地修改）。"""
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.calls += other.calls

    @classmethod
    def from_api(cls, usage: Any) -> "Usage":
        """把 OpenAI SDK 的 usage 对象或普通 dict 转成 Usage。

        输入可能是 None、pydantic 对象或 dict；统一用 getattr / get 读取，
        避免把 SDK 类型泄漏到其他模块。
        """
        if usage is None:
            return cls()

        def pick(name: str) -> int:
            if isinstance(usage, dict):
                return int(usage.get(name, 0) or 0)
            return int(getattr(usage, name, 0) or 0)

        prompt = pick("prompt_tokens")
        completion = pick("completion_tokens")
        total = pick("total_tokens") or (prompt + completion)
        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            calls=1,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToolCall:
    """模型请求的一次函数调用。

    arguments 保留原始 JSON 字符串：模型可能输出非法 JSON，
    解析失败本身就是要记录和分析的失败模式，不能提前吞掉。
    """

    id: str
    name: str
    arguments: str

    def parsed_arguments(self) -> dict:
        """解析参数；失败时抛 ValueError，由 runner 转成工具错误。"""
        try:
            value = json.loads(self.arguments or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"工具参数不是合法 JSON：{exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("工具参数必须是 JSON 对象")
        return value

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LLMResponse:
    """一次模型调用的归一化结果。"""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    model: str = ""

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "usage": self.usage.to_dict(),
            "latency_ms": round(self.latency_ms, 2),
            "model": self.model,
        }


@dataclass
class ToolExecution:
    """一次工具调用的执行结果（可能包含多次重试）。"""

    name: str
    args: dict
    ok: bool
    content: str
    error: str | None = None
    latency_ms: float = 0.0
    attempts: int = 1
    failed_attempts: int = 0
    attempt_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "args": self.args,
            "ok": self.ok,
            "content": self.content,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 2),
            "attempts": self.attempts,
            "failed_attempts": self.failed_attempts,
            "attempt_errors": self.attempt_errors,
        }


@dataclass
class RunOutput:
    """Agent loop 的原始输出，还没经过任务 validator。"""

    answer: str
    steps: int
    usage: Usage
    latency_ms: float
    tool_calls: int
    tool_errors: int
    recovery_events: int
    stopped_reason: str
    messages: list[dict] = field(default_factory=list)


@dataclass
class EpisodeResult:
    """一个 task × spec × repeat 的完整评测结果。"""

    episode_id: str
    task_id: str
    category: str
    spec_name: str
    model: str
    repeat: int
    passed: bool
    detail: str
    answer: str
    steps: int
    usage: Usage
    latency_ms: float
    tool_calls: int
    tool_errors: int
    recovery_events: int
    failure_type: str
    stopped_reason: str
    trace_path: str
    workspace: str
    escalated: bool = False

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "category": self.category,
            "spec_name": self.spec_name,
            "model": self.model,
            "repeat": self.repeat,
            "passed": self.passed,
            "detail": self.detail,
            "answer": self.answer,
            "steps": self.steps,
            "usage": self.usage.to_dict(),
            "latency_ms": round(self.latency_ms, 2),
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "recovery_events": self.recovery_events,
            "failure_type": self.failure_type,
            "stopped_reason": self.stopped_reason,
            "trace_path": self.trace_path,
            "workspace": self.workspace,
            "escalated": self.escalated,
        }

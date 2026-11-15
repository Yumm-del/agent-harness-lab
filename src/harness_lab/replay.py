# -*- coding: utf-8 -*-
"""按 trace 回放一次 episode。

回放不是「再看一遍日志」，而是重新跑工具：
  - LLM 响应从 trace 里按顺序取，不再花钱调模型；
  - 工具真实执行，所以文件、mock API、故障注入都会重新发生；
  - 每一步比较 request_hash，提示词/工具/代码变了会立即报错。

这对应 JD 里的「流量回放」：线上出了 badcase，把请求序列录下来，
离线重跑，改 harness 后再跑，验证修复是否真的有效。
"""

from __future__ import annotations

import json
from pathlib import Path

from .evaluate import _prepare_workspace, classify_failure
from .llm import LLMClient
from .models import LLMResponse, ToolCall, Usage
from .runner import AgentRunner
from .spec import AgentSpec
from .tasks import get_task
from .tools import FaultInjector, MockAPI, Sandbox, build_default_registry
from .trace import TraceWriter, messages_hash, new_run_id, read_trace


class ReplayMismatch(RuntimeError):
    """回放时 messages 与录制时不一致，说明代码/提示词/工具发生了漂移。"""


class ReplayLLM(LLMClient):
    """按顺序返回 trace 里的 LLM 响应，并校验请求哈希。"""

    def __init__(self, events: list[dict], verify_hash: bool = True) -> None:
        self.records = [event for event in events if event.get("event") == "llm_call"]
        self.verify_hash = verify_hash
        self.index = 0
        self.used: list[int] = []

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        if self.index >= len(self.records):
            raise ReplayMismatch(
                f"trace 只有 {len(self.records)} 次 LLM 调用，回放却请求了第 {self.index + 1} 次"
            )
        record = self.records[self.index]
        expected = record.get("request_hash")
        actual = messages_hash(messages)
        if self.verify_hash and expected and expected != actual:
            raise ReplayMismatch(
                f"第 {self.index + 1} 次 LLM 请求哈希不一致："
                f"录制={expected} 回放={actual}；提示词/工具/消息序列已变化"
            )
        self.used.append(self.index)
        self.index += 1

        payload = record.get("response", {})
        usage_data = payload.get("usage", {})
        usage = Usage(
            prompt_tokens=int(usage_data.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage_data.get("completion_tokens", 0) or 0),
            total_tokens=int(usage_data.get("total_tokens", 0) or 0),
            calls=int(usage_data.get("calls", 1) or 1),
        )
        calls = [
            ToolCall(
                id=str(item.get("id", "")),
                name=str(item.get("name", "")),
                arguments=str(item.get("arguments", "{}")),
            )
            for item in payload.get("tool_calls", [])
        ]
        return LLMResponse(
            content=str(payload.get("content", "")),
            tool_calls=calls,
            usage=usage,
            latency_ms=float(payload.get("latency_ms", 0.0) or 0.0),
            model=str(payload.get("model", model)),
        )


def replay_episode(trace_path: str | Path, out_dir: str | Path) -> dict:
    """回放一个 trace，返回录制/回放的对比结果。"""
    source = Path(trace_path)
    events = read_trace(source)
    start = next((event for event in events if event.get("event") == "episode_start"), None)
    final = next((event for event in reversed(events) if event.get("event") == "final"), None)
    if start is None:
        raise ValueError(f"trace 缺少 episode_start：{source}")

    task = get_task(str(start["task_id"]))
    spec = AgentSpec.from_dict(start["spec"])
    replay_root = Path(out_dir) / new_run_id()
    episode_dir = replay_root / "replay" / task.id / "repeat_0"
    workspace = episode_dir / "workspace"
    _prepare_workspace(task, workspace)

    replay_trace = episode_dir / "trace.jsonl"
    llm = ReplayLLM(events, verify_hash=True)
    registry = build_default_registry(
        Sandbox(workspace),
        MockAPI(task.mock_api),
        FaultInjector(list(task.faults)),
    )

    with TraceWriter(
        replay_trace,
        episode_id=f"replay-{start.get('episode_id', '')}",
        task_id=task.id,
        category=task.category,
        prompt=start.get("prompt", task.prompt),
        spec=spec,
    ) as trace:
        output = AgentRunner(spec, llm, registry, trace).run(
            task.prompt, task.id, task.category
        )
        passed, detail = task.validator(workspace, output.answer)
        replay_events = read_trace(replay_trace)
        failure_type = classify_failure(output, passed, replay_events, detail)
        trace.write(
            "final",
            answer=output.answer,
            passed=passed,
            detail=detail,
            failure_type=failure_type,
            stopped_reason=output.stopped_reason,
        )

    recorded_passed = bool(final.get("passed", False)) if final else False
    result = {
        "source_trace": str(source),
        "replay_trace": str(replay_trace),
        "task_id": task.id,
        "spec": spec.name,
        "recorded_passed": recorded_passed,
        "replay_passed": passed,
        "same_outcome": recorded_passed == passed,
        "recorded_detail": final.get("detail", "") if final else "",
        "replay_detail": detail,
        "llm_calls_recorded": len(llm.records),
        "llm_calls_replayed": llm.index,
        "answer": output.answer,
    }
    (episode_dir / "replay_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result

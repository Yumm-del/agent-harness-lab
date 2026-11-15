# -*- coding: utf-8 -*-
"""Agent loop 的行为测试：不调真实 API，用 ScriptedLLM 控制时序。"""

from __future__ import annotations

import json

from harness_lab.llm import ScriptedLLM
from harness_lab.models import LLMResponse, ToolCall, Usage
from harness_lab.runner import AgentRunner
from harness_lab.spec import AgentSpec
from harness_lab.tools import (
    FaultInjector,
    FaultRule,
    MockAPI,
    Sandbox,
    build_default_registry,
)
from harness_lab.trace import TraceWriter


def make_spec(**overrides) -> AgentSpec:
    data = {
        "name": "test",
        "model": "mock",
        "instruction": "test",
        "tools": ["list_files", "read_file", "write_file", "http_get", "run_python"],
        "max_steps": 4,
    }
    data.update(overrides)
    return AgentSpec.from_dict(data)


def make_registry(tmp_path, faults=None):
    sandbox = Sandbox(tmp_path)
    return build_default_registry(sandbox, MockAPI({}), FaultInjector(faults or []))


def response(content="", calls=None, tokens=10) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=calls or [],
        usage=Usage(prompt_tokens=tokens, completion_tokens=5, total_tokens=tokens + 5, calls=1),
        latency_ms=1.0,
        model="mock",
    )


def test_tool_call_then_final(tmp_path):
    registry = make_registry(tmp_path)
    llm = ScriptedLLM(
        [
            response(
                calls=[
                    ToolCall(
                        id="c1",
                        name="write_file",
                        arguments=json.dumps({"path": "out.txt", "content": "hello"}),
                    )
                ]
            ),
            response("done"),
        ]
    )
    trace_path = tmp_path / "trace.jsonl"
    with TraceWriter(
        trace_path,
        episode_id="e1",
        task_id="t1",
        category="file_data",
        prompt="write hello",
        spec=make_spec(),
    ) as trace:
        output = AgentRunner(make_spec(), llm, registry, trace).run("write hello", "t1", "file_data")

    assert output.answer == "done"
    assert output.stopped_reason == "final"
    assert output.tool_calls == 1
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "hello"
    assert trace_path.exists()


def test_tool_retry_is_counted(tmp_path):
    (tmp_path / "a.txt").write_text("ok", encoding="utf-8")
    registry = make_registry(
        tmp_path,
        faults=[FaultRule(tool="read_file", mode="timeout", times=1)],
    )
    llm = ScriptedLLM(
        [
            response(
                calls=[ToolCall(id="c1", name="read_file", arguments=json.dumps({"path": "a.txt"}))]
            ),
            response("done"),
        ]
    )
    with TraceWriter(
        tmp_path / "trace.jsonl",
        episode_id="e1",
        task_id="t1",
        category="file_data",
        prompt="read",
        spec=make_spec(tool_retries=1),
    ) as trace:
        output = AgentRunner(
            make_spec(tool_retries=1), llm, registry, trace
        ).run("read", "t1", "file_data")

    assert output.tool_calls == 1
    assert output.tool_errors == 1
    assert output.recovery_events == 1


def test_max_steps_stops_loop(tmp_path):
    registry = make_registry(tmp_path)
    llm = ScriptedLLM(
        [
            response(
                calls=[
                    ToolCall(id="c1", name="list_files", arguments=json.dumps({"path": "."}))
                ]
            )
        ]
    )
    spec = make_spec(max_steps=2)
    with TraceWriter(
        tmp_path / "trace.jsonl",
        episode_id="e1",
        task_id="t1",
        category="file_data",
        prompt="loop",
        spec=spec,
    ) as trace:
        output = AgentRunner(spec, llm, registry, trace).run("loop", "t1", "file_data")

    assert output.stopped_reason == "max_steps"
    assert output.steps == 2


def test_invalid_tool_arguments_are_reported_to_model(tmp_path):
    registry = make_registry(tmp_path)
    llm = ScriptedLLM(
        [
            response(
                calls=[ToolCall(id="c1", name="read_file", arguments="{not-json")]
            ),
            response("done"),
        ]
    )
    with TraceWriter(
        tmp_path / "trace.jsonl",
        episode_id="e1",
        task_id="t1",
        category="file_data",
        prompt="bad args",
        spec=make_spec(),
    ) as trace:
        output = AgentRunner(make_spec(), llm, registry, trace).run(
            "bad args", "t1", "file_data"
        )

    assert output.tool_calls == 1
    assert output.tool_errors == 1
    # 第二轮请求里必须有一条 tool message，否则下一次真实 API 调用会 400。
    second_messages = llm.calls[1]["messages"]
    assert any(message.get("role") == "tool" for message in second_messages)

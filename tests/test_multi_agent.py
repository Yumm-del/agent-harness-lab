# -*- coding: utf-8 -*-
"""多智能体测试：planner/executor 交接、失败降级、失败归因与回放。"""

from __future__ import annotations

import json

import pytest

from harness_lab.evaluate import aggregate, run_episode
from harness_lab.llm import ScriptedLLM
from harness_lab.models import LLMResponse, ToolCall, Usage
from harness_lab.replay import replay_episode
from harness_lab.spec import AgentSpec
from harness_lab.tasks import get_task
from harness_lab.trace import read_trace


def response(content="", calls=None, tokens=10) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=calls or [],
        usage=Usage(prompt_tokens=tokens, completion_tokens=5, total_tokens=tokens + 5, calls=1),
        latency_ms=1.0,
        model="mock",
    )


def make_spec(**overrides) -> AgentSpec:
    data = {
        "name": "multi",
        "model": "mock",
        "instruction": "test",
        "tools": ["list_files", "read_file", "write_file", "http_get", "run_python"],
        "max_steps": 6,
        "architecture": "planner_executor",
    }
    data.update(overrides)
    return AgentSpec.from_dict(data)


def write_result_call() -> ToolCall:
    return ToolCall(
        id="c1",
        name="write_file",
        arguments=json.dumps(
            {
                "path": "result.json",
                "content": json.dumps({"rows": 3, "total": 250.0}),
            }
        ),
    )


def test_planner_executor_handoff(tmp_path):
    task = get_task("fd_01_csv_filter")
    llm = ScriptedLLM(
        [
            response("1. Read data.csv\n2. Count rows and sum total\n3. Write result.json"),
            response(calls=[write_result_call()]),
            response("done"),
        ]
    )

    result = run_episode(task, make_spec(), llm, tmp_path, repeat=0)

    assert result.passed
    assert result.handoff_count == 1
    assert result.handoff_ok
    assert result.planner_tokens > 0
    # planner 不应该拿到工具 schema，否则它可能直接动手。
    assert llm.calls[0]["tools"] == []
    # executor 的 system prompt 里应该有 planner 的计划。
    executor_system = llm.calls[1]["messages"][0]["content"]
    assert "Planner handoff" in executor_system
    assert "Read data.csv" in executor_system

    events = read_trace(tmp_path / result.trace_path)
    planner_calls = [
        event
        for event in events
        if event.get("event") == "llm_call" and event.get("agent") == "planner"
    ]
    handoffs = [event for event in events if event.get("event") == "handoff"]
    assert len(planner_calls) == 1
    assert len(handoffs) == 1
    assert handoffs[0]["ok"] is True

    summary = aggregate([result])
    assert summary["multi"]["handoff_count"] == 1
    assert summary["multi"]["handoff_ok_count"] == 1
    assert summary["multi"]["planner_tokens"] == result.planner_tokens
    assert summary["multi"]["planner_token_rate"] > 0


def test_planner_failure_falls_back_to_executor(tmp_path):
    task = get_task("fd_01_csv_filter")

    def planner_boom(messages):
        raise RuntimeError("planner down")

    llm = ScriptedLLM(
        [
            planner_boom,
            response(calls=[write_result_call()]),
            response("done"),
        ]
    )

    result = run_episode(task, make_spec(), llm, tmp_path, repeat=0)

    # planner 挂了，executor 仍按单 Agent 方式完成任务。
    assert result.passed
    assert result.handoff_count == 1
    assert not result.handoff_ok
    assert result.planner_tokens == 0

    events = read_trace(tmp_path / result.trace_path)
    planner_calls = [
        event
        for event in events
        if event.get("event") == "llm_call" and event.get("agent") == "planner"
    ]
    assert len(planner_calls) == 1
    assert planner_calls[0].get("error")
    handoffs = [event for event in events if event.get("event") == "handoff"]
    assert handoffs and handoffs[0]["ok"] is False


def test_planner_failure_is_classified(tmp_path):
    task = get_task("fd_01_csv_filter")

    def planner_boom(messages):
        raise RuntimeError("planner down")

    llm = ScriptedLLM([planner_boom, response("I cannot do that")])
    result = run_episode(task, make_spec(), llm, tmp_path, repeat=0)

    assert not result.passed
    assert result.failure_type == "planner_error"


def test_replay_multi_agent(tmp_path):
    task = get_task("fd_01_csv_filter")
    llm = ScriptedLLM(
        [
            response("1. Read data.csv\n2. Compute rows/total\n3. Write result.json"),
            response(calls=[write_result_call()]),
            response("done"),
        ]
    )
    recorded = run_episode(task, make_spec(), llm, tmp_path / "recorded", repeat=0)
    assert recorded.passed

    replayed = replay_episode(
        tmp_path / "recorded" / recorded.trace_path,
        tmp_path / "replayed",
    )

    assert replayed["same_outcome"]
    assert replayed["replay_passed"]
    # planner 1 次 + executor 2 次，回放必须逐条对齐。
    assert replayed["llm_calls_recorded"] == 3
    assert replayed["llm_calls_replayed"] == 3


def test_invalid_architecture_rejected():
    with pytest.raises(ValueError, match="architecture"):
        AgentSpec.from_dict(
            {
                "name": "bad",
                "model": "m",
                "instruction": "x",
                "tools": ["read_file"],
                "architecture": "swarm",
            }
        )

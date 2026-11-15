# -*- coding: utf-8 -*-
"""评测层：validator、失败分类、聚合。"""

from __future__ import annotations

import json

from harness_lab.evaluate import aggregate, classify_failure, run_episode
from harness_lab.llm import ScriptedLLM
from harness_lab.models import LLMResponse, RunOutput, ToolCall, Usage
from harness_lab.spec import AgentSpec
from harness_lab.tasks import get_task


def response(content="", calls=None) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=calls or [],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15, calls=1),
        latency_ms=1.0,
        model="mock",
    )


def spec(**overrides) -> AgentSpec:
    data = {
        "name": "test",
        "model": "mock",
        "instruction": "test",
        "tools": ["list_files", "read_file", "write_file", "http_get", "run_python"],
        "max_steps": 6,
    }
    data.update(overrides)
    return AgentSpec.from_dict(data)


def test_run_episode_passes_with_validator(tmp_path):
    task = get_task("fd_01_csv_filter")
    llm = ScriptedLLM(
        [
            response(
                calls=[
                    ToolCall(
                        id="c1",
                        name="write_file",
                        arguments=json.dumps(
                            {
                                "path": "result.json",
                                "content": json.dumps({"rows": 3, "total": 250.0}),
                            }
                        ),
                    )
                ]
            ),
            response("done"),
        ]
    )
    result = run_episode(task, spec(), llm, tmp_path, repeat=0)
    assert result.passed
    assert result.failure_type == "none"
    assert result.trace_path.endswith("trace.jsonl")


def test_run_episode_fails_without_tool_call(tmp_path):
    task = get_task("fd_01_csv_filter")
    llm = ScriptedLLM([response("I cannot do that")])
    result = run_episode(task, spec(), llm, tmp_path, repeat=0)
    assert not result.passed
    assert result.failure_type == "planning_no_tool"


def test_classify_step_limit():
    output = RunOutput(
        answer="",
        steps=3,
        usage=Usage(),
        latency_ms=1,
        tool_calls=2,
        tool_errors=0,
        recovery_events=0,
        stopped_reason="max_steps",
    )
    assert classify_failure(output, False, [{"event": "llm_call"}], "缺少输出文件") == "step_limit"


def test_aggregate_success_rate(tmp_path):
    task = get_task("fd_01_csv_filter")
    llm = ScriptedLLM(
        [
            response(
                calls=[
                    ToolCall(
                        id="c1",
                        name="write_file",
                        arguments=json.dumps(
                            {
                                "path": "result.json",
                                "content": json.dumps({"rows": 3, "total": 250.0}),
                            }
                        ),
                    )
                ]
            ),
            response("done"),
        ]
    )
    result = run_episode(task, spec(), llm, tmp_path / "run", repeat=0)
    summary = aggregate([result])
    assert summary["test"]["success_rate"] == 1.0
    assert summary["test"]["total"] == 1

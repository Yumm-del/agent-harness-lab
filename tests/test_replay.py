# -*- coding: utf-8 -*-
"""回放测试：录制一次 episode，再用 trace 重跑工具。"""

from __future__ import annotations

import json

from harness_lab.evaluate import run_episode
from harness_lab.llm import ScriptedLLM
from harness_lab.models import LLMResponse, ToolCall, Usage
from harness_lab.replay import replay_episode
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


def make_spec() -> AgentSpec:
    return AgentSpec.from_dict(
        {
            "name": "test",
            "model": "mock",
            "instruction": "test",
            "tools": ["list_files", "read_file", "write_file", "http_get", "run_python"],
            "max_steps": 6,
        }
    )


def test_replay_reproduces_outcome(tmp_path):
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
    recorded = run_episode(task, make_spec(), llm, tmp_path / "recorded", repeat=0)
    assert recorded.passed

    replayed = replay_episode(
        tmp_path / "recorded" / recorded.trace_path,
        tmp_path / "replayed",
    )
    assert replayed["same_outcome"]
    assert replayed["replay_passed"]
    assert replayed["llm_calls_replayed"] == replayed["llm_calls_recorded"]

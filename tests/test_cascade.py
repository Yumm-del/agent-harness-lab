# -*- coding: utf-8 -*-
"""Cascade 测试：弱配置失败后升级，结果合并 usage。"""

from __future__ import annotations

import json

from harness_lab.evaluate import run_cascade_episode
from harness_lab.llm import ScriptedLLM
from harness_lab.models import LLMResponse, ToolCall, Usage
from harness_lab.spec import AgentSpec
from harness_lab.tasks import get_task


def response(content="", calls=None, tokens=10) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=calls or [],
        usage=Usage(prompt_tokens=tokens, completion_tokens=5, total_tokens=tokens + 5, calls=1),
        latency_ms=1.0,
        model="mock",
    )


def make_spec(name: str) -> AgentSpec:
    return AgentSpec.from_dict(
        {
            "name": name,
            "model": "mock",
            "instruction": "test",
            "tools": ["list_files", "read_file", "write_file", "http_get", "run_python"],
            "max_steps": 6,
        }
    )


def test_cascade_escalates_on_failure(tmp_path):
    task = get_task("fd_01_csv_filter")
    llm = ScriptedLLM(
        [
            # 弱配置：没有工具调用，直接给错误答案
            response("I cannot do that"),
            # 强配置：写正确文件，然后结束
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
    result = run_cascade_episode(
        task,
        make_spec("weak"),
        make_spec("strong"),
        llm,
        tmp_path,
        repeat=0,
    )
    assert result.passed
    assert result.escalated
    assert result.spec_name == "cascade"
    assert result.usage.total_tokens > 0

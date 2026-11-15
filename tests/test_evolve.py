# -*- coding: utf-8 -*-
"""AEvo-lite v0 的规则测试。"""

from __future__ import annotations

from harness_lab.evolve import propose_evolution
from harness_lab.spec import AgentSpec


def base_spec() -> AgentSpec:
    return AgentSpec.from_dict(
        {
            "name": "baseline",
            "model": "deepseek-chat",
            "instruction": "Do the task.",
            "tools": ["read_file", "write_file"],
        }
    )


def test_step_limit_turns_on_retry():
    evolved, reason = propose_evolution(
        base_spec(),
        [{"passed": False, "failure_type": "step_limit"}],
    )
    assert evolved.tool_retries == 2
    assert evolved.repair_on_error
    assert "自动重试" in reason


def test_missing_artifact_adds_instruction():
    evolved, reason = propose_evolution(
        base_spec(),
        [{"passed": False, "failure_type": "missing_artifact"}],
    )
    assert "verify that every requested output file exists" in evolved.instruction
    assert "缺输出文件" in reason


def test_no_failure_keeps_spec():
    spec = base_spec()
    evolved, reason = propose_evolution(spec, [{"passed": True, "failure_type": "none"}])
    assert evolved is spec
    assert "不生成修改" in reason

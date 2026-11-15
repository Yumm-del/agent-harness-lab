# -*- coding: utf-8 -*-
"""AgentSpec 配置测试。"""

from __future__ import annotations

import pytest

from harness_lab.config import PROJECT_ROOT
from harness_lab.spec import AgentSpec, load_spec, load_specs


def test_load_baseline_and_repair_differ_only_in_control():
    baseline = load_spec(PROJECT_ROOT / "configs" / "baseline.yaml")
    repair = load_spec(PROJECT_ROOT / "configs" / "repair.yaml")
    assert baseline.model == repair.model
    assert baseline.tools == repair.tools
    assert baseline.context_policy == repair.context_policy
    assert baseline.tool_retries == 0
    assert repair.tool_retries == 2
    assert not baseline.repair_on_error
    assert repair.repair_on_error


def test_duplicate_spec_names_rejected(tmp_path):
    path_a = tmp_path / "a.yaml"
    path_b = tmp_path / "b.yaml"
    text = """
name: same
model: deepseek-chat
instruction: test
tools: [read_file]
"""
    path_a.write_text(text, encoding="utf-8")
    path_b.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="重复"):
        load_specs([path_a, path_b])


def test_invalid_context_policy_rejected():
    with pytest.raises(ValueError, match="context_policy"):
        AgentSpec.from_dict(
            {
                "name": "bad",
                "model": "m",
                "instruction": "x",
                "tools": ["read_file"],
                "context_policy": "summarize",
            }
        )

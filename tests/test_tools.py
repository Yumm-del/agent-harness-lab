# -*- coding: utf-8 -*-
"""工具与沙箱的单元测试。"""

from __future__ import annotations

import json

import pytest

from harness_lab.tools import (
    FaultInjector,
    FaultRule,
    MockAPI,
    Sandbox,
    ToolError,
    build_default_registry,
)


def test_sandbox_blocks_path_traversal(tmp_path):
    sandbox = Sandbox(tmp_path)
    with pytest.raises(ToolError):
        sandbox.read_file("../outside.txt")
    with pytest.raises(ToolError):
        sandbox.write_file(str(tmp_path / "absolute.txt"), "x")


def test_sandbox_write_read_list(tmp_path):
    sandbox = Sandbox(tmp_path)
    sandbox.write_file("nested/out.txt", "hello")
    assert (tmp_path / "nested" / "out.txt").read_text(encoding="utf-8") == "hello"

    result = sandbox.read_file("nested/out.txt")
    assert result.ok
    payload = json.loads(result.content)
    assert payload["content"] == "hello"

    listing = sandbox.list_files(".")
    files = json.loads(listing.content)["files"]
    assert "nested/out.txt" in files


def test_sandbox_python_runs_in_workspace(tmp_path):
    sandbox = Sandbox(tmp_path)
    result = sandbox.run_python("print(6 * 7)")
    payload = json.loads(result.content)
    assert payload["returncode"] == 0
    assert payload["stdout"].strip() == "42"


def test_mock_api_is_case_sensitive():
    api = MockAPI({"/weather?city=Beijing": {"temp": 22}})
    assert json.loads(api.get("mock://api/weather?city=Beijing").content)["temp"] == 22
    with pytest.raises(ToolError):
        api.get("mock://api/weather?city=beijing")


def test_fault_injector_consumes_rule(tmp_path):
    sandbox = Sandbox(tmp_path)
    sandbox.write_file("a.txt", "ok")
    registry = build_default_registry(
        sandbox,
        MockAPI({}),
        FaultInjector([FaultRule(tool="read_file", mode="timeout", times=1)]),
    )
    first = registry.execute("read_file", {"path": "a.txt"})
    second = registry.execute("read_file", {"path": "a.txt"})
    assert not first.ok
    assert second.ok
    assert second.failed_attempts == 0


def test_tool_retry_recovers(tmp_path):
    sandbox = Sandbox(tmp_path)
    sandbox.write_file("a.txt", "ok")
    registry = build_default_registry(
        sandbox,
        MockAPI({}),
        FaultInjector([FaultRule(tool="read_file", mode="timeout", times=1)]),
    )
    result = registry.execute("read_file", {"path": "a.txt"}, retries=1)
    assert result.ok
    assert result.attempts == 2
    assert result.failed_attempts == 1

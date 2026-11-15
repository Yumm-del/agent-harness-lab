# -*- coding: utf-8 -*-
"""AEvo-lite v0：从失败报告生成一个最小 harness 修改。

这不是完整的 AEvo（论文里的 meta-agent 会编辑 procedure/context），
而是一个可解释的 v0：
  1. 读一次 run 的 results.json；
  2. 统计失败类型；
  3. 只针对最明确的失败模式改一个配置；
  4. 写出新 YAML，交给下一次 eval 验证。

为什么要做成规则而不是直接让 LLM 改：
  - 规则版本可复现、可审计，适合先验证「失败 → 修改 → 回归」这条链路；
  - 等链路稳定后，再把「选哪个修改」换成 LLM meta-agent；
  - 面试时能清楚区分「已实现」和「下一步」。
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from .spec import AgentSpec


def load_results(path: str | Path) -> tuple[dict, list[dict]]:
    """读取 results.json，返回 (meta, results)。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("meta", {}), payload.get("results", [])


def propose_evolution(
    base_spec: AgentSpec,
    results: list[dict],
) -> tuple[AgentSpec, str]:
    """根据失败分布提出一个最小修改。

    输入：
      base_spec — 原始配置（通常是 baseline）
      results   — 该配置下的 episode 结果列表

    输出：
      (新 spec, 修改理由)。如果没有可执行的失败模式，返回原 spec + 说明。

    规则优先级：
      1. 工具/步数/恢复类失败 → 开自动重试 + 修复提示（harness 控制）
      2. 缺输出文件 → 在 instruction 里要求完成前检查文件
      3. JSON 格式错 → 在 instruction 里要求严格 JSON
      4. 完全不调工具 → 在 instruction 里要求使用工具
    """
    failures = Counter(
        str(item.get("failure_type", "unknown"))
        for item in results
        if not item.get("passed", False)
    )
    changes: dict[str, Any] = {}
    reasons: list[str] = []
    instruction = base_spec.instruction

    if any(failures[key] for key in ("step_limit", "tool_failure", "recovery_loop")):
        changes["tool_retries"] = max(base_spec.tool_retries, 2)
        changes["repair_on_error"] = True
        reasons.append(
            "工具/步数/恢复类失败 → 开启 2 次自动重试与失败修复提示"
        )

    if failures["missing_artifact"]:
        instruction += (
            "\nBefore finishing, verify that every requested output file exists "
            "and contains the required content."
        )
        reasons.append("缺输出文件 → 在 instruction 中要求完成前检查产物")

    if failures["format_error"]:
        instruction += (
            "\nWhen JSON is requested, write valid JSON with exactly the requested keys."
        )
        reasons.append("JSON 格式错误 → 在 instruction 中要求严格 JSON")

    if failures["planning_no_tool"]:
        instruction += (
            "\nUse the available tools to inspect and modify the workspace; "
            "do not answer from memory."
        )
        reasons.append("未调用工具 → 在 instruction 中要求先动手")

    if not changes and instruction == base_spec.instruction:
        return base_spec, "没有可执行的失败模式，不生成修改"

    evolved = replace(
        base_spec,
        name="evolved",
        # description 必须重写：直接继承 baseline 会写出与实际配置矛盾的注释
        description=f"AEvo-lite v0 自动生成（基于 {base_spec.name} 的失败报告）：{'；'.join(reasons)}",
        instruction=instruction,
        **changes,
    )
    return evolved, "；".join(reasons)


def write_spec(spec: AgentSpec, path: str | Path) -> Path:
    """把 spec 写成 YAML，字段顺序保持可读。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "name": spec.name,
        "description": spec.description or "生成的 evolved 配置",
        "model": spec.model,
        "instruction": spec.instruction,
        "tools": list(spec.tools),
        "max_steps": spec.max_steps,
        "temperature": spec.temperature,
        "max_tokens": spec.max_tokens,
        "context_policy": spec.context_policy,
        "context_window": spec.context_window,
        "tool_retries": spec.tool_retries,
        "repair_on_error": spec.repair_on_error,
    }
    # 只在非默认时写出，避免给旧的单 Agent 配置平白增加两行。
    if spec.architecture != "single":
        data["architecture"] = spec.architecture
    if spec.planner_instruction:
        data["planner_instruction"] = spec.planner_instruction
    target.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target


def evolve_from_results(
    results_path: str | Path,
    spec_name: str | None,
    out_path: str | Path,
) -> tuple[AgentSpec, str, Path]:
    """从 results.json 找到指定 spec，生成 evolved YAML。"""
    meta, results = load_results(results_path)
    specs = meta.get("specs", [])
    if not specs:
        raise ValueError("results.json 里没有 meta.specs")

    if spec_name:
        spec_dict = next((item for item in specs if item.get("name") == spec_name), None)
        if spec_dict is None:
            raise ValueError(f"找不到 spec：{spec_name}")
    else:
        spec_dict = specs[0]
        spec_name = str(spec_dict.get("name", ""))

    base_spec = AgentSpec.from_dict(spec_dict)
    subset = [item for item in results if item.get("spec_name") == spec_name]
    if not subset:
        raise ValueError(f"results 里没有 spec={spec_name} 的 episode")

    evolved, reason = propose_evolution(base_spec, subset)
    path = write_spec(evolved, out_path)
    return evolved, reason, path

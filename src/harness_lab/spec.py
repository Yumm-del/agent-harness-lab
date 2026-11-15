# -*- coding: utf-8 -*-
"""AgentSpec：一个 harness 配置的不可变描述。

借 AOrchestra 的思路，把 Agent 拆成四个可替换的部分：
  Instruction（system prompt）
  Context（上下文策略）
  Tools（可用工具集合）
  Model（模型与采样参数）

这样对比实验不是「改代码」，而是「换一个 YAML」；
trace 里记录完整 spec，回放时才能重建同一次运行。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AgentSpec:
    """一次运行的静态配置。frozen=True 防止 runner 中途改配置。"""

    name: str
    model: str
    instruction: str
    tools: tuple[str, ...]
    max_steps: int = 8
    temperature: float = 0.0
    max_tokens: int = 900
    context_policy: str = "full"       # full | prune
    context_window: int = 4            # prune 时保留最近多少个 tool observation
    tool_retries: int = 0              # harness 层自动重试次数
    repair_on_error: bool = False      # 工具失败后是否注入修复提示
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("spec.name 不能为空")
        if not self.model.strip():
            raise ValueError("spec.model 不能为空")
        if not self.tools:
            raise ValueError("spec.tools 不能为空")
        if not 1 <= self.max_steps <= 30:
            raise ValueError("max_steps 必须在 1..30 之间")
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError("temperature 必须在 0..1 之间")
        if not 64 <= self.max_tokens <= 8000:
            raise ValueError("max_tokens 必须在 64..8000 之间")
        if self.context_policy not in {"full", "prune"}:
            raise ValueError("context_policy 只支持 full / prune")
        if self.context_window < 1:
            raise ValueError("context_window 必须 >= 1")
        if not 0 <= self.tool_retries <= 5:
            raise ValueError("tool_retries 必须在 0..5 之间")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentSpec":
        """从 YAML/dict 构造；tools 允许 list，内部转 tuple。"""
        if not isinstance(data, dict):
            raise ValueError("spec 必须是 YAML 对象")
        tools = data.get("tools", [])
        if not isinstance(tools, list) or not all(isinstance(t, str) for t in tools):
            raise ValueError("spec.tools 必须是字符串列表")
        return cls(
            name=str(data["name"]),
            model=str(data.get("model", "deepseek-chat")),
            instruction=str(data["instruction"]),
            tools=tuple(tools),
            max_steps=int(data.get("max_steps", 8)),
            temperature=float(data.get("temperature", 0.0)),
            max_tokens=int(data.get("max_tokens", 900)),
            context_policy=str(data.get("context_policy", "full")),
            context_window=int(data.get("context_window", 4)),
            tool_retries=int(data.get("tool_retries", 0)),
            repair_on_error=bool(data.get("repair_on_error", False)),
            description=str(data.get("description", "")),
        )

    def to_dict(self) -> dict:
        """trace 里保存的 spec；回放和报告都用它。"""
        data = asdict(self)
        data["tools"] = list(self.tools)
        return data


def load_spec(path: str | Path) -> AgentSpec:
    """读取单个 YAML 配置文件。"""
    spec_path = Path(path)
    if not spec_path.exists():
        raise FileNotFoundError(f"spec 不存在：{spec_path}")
    data = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"spec 文件必须是 YAML 对象：{spec_path}")
    return AgentSpec.from_dict(data)


def load_specs(paths: list[str | Path]) -> list[AgentSpec]:
    """批量读取，并检查 name 不重复（否则报告里会混淆）。"""
    specs = [load_spec(path) for path in paths]
    names = [spec.name for spec in specs]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ValueError(f"spec.name 重复：{sorted(duplicates)}")
    return specs

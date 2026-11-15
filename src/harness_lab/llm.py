# -*- coding: utf-8 -*-
"""LLM 客户端抽象。

runner 只依赖 `LLMClient.chat(...)` 这个协议，不关心底层是 DeepSeek、
OpenAI 还是测试里的脚本模型。这样：
  - 真实实验用 OpenAICompatibleClient
  - 单元测试用 ScriptedLLM，不花钱、不依赖网络
  - 回放时用 ReplayLLM，按 trace 重放历史响应
"""

from __future__ import annotations

import time
from typing import Any, Callable, Protocol

from .models import LLMResponse, ToolCall, Usage


class LLMClient(Protocol):
    """所有客户端都要实现的最小接口。"""

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        ...


class OpenAICompatibleClient:
    """OpenAI 兼容接口客户端（DeepSeek 用同一套协议，只换 base_url）。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        timeout: float = 90.0,
        max_retries: int = 2,
    ) -> None:
        if not api_key:
            raise ValueError("缺少 API Key")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - 依赖缺失时给出可执行提示
            raise RuntimeError("请先安装 openai：python -m pip install openai") from exc

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """调用 chat.completions，并把 SDK 对象归一化成 LLMResponse。

        注意：
          - tools 为空时不要传 tools/tool_choice，否则部分兼容服务会报错；
          - content 可能是 None（模型只返回 tool_calls），统一转成空字符串；
          - arguments 保留原始字符串，非法 JSON 由 runner 记录为工具错误。
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        started = time.perf_counter()
        response = self._client.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - started) * 1000

        message = response.choices[0].message
        tool_calls: list[ToolCall] = []
        for call in (message.tool_calls or []):
            function = getattr(call, "function", None)
            if function is None:
                continue
            tool_calls.append(
                ToolCall(
                    id=str(getattr(call, "id", "") or ""),
                    name=str(getattr(function, "name", "") or ""),
                    arguments=str(getattr(function, "arguments", "") or "{}"),
                )
            )

        return LLMResponse(
            content=str(message.content or ""),
            tool_calls=tool_calls,
            usage=Usage.from_api(getattr(response, "usage", None)),
            latency_ms=latency_ms,
            model=model,
        )


ResponseFactory = LLMResponse | Callable[[list[dict]], LLMResponse]


class ScriptedLLM:
    """测试用：按顺序返回预设响应，或调用一个根据 messages 生成响应的函数。

    输入：
      responses — LLMResponse 列表，或 (messages) -> LLMResponse 的函数。
    输出：
      每次 chat 返回一个响应，并把请求记在 self.calls 里供断言。

    原理：
      单元测试不应该依赖真实 API。脚本模型让「模型第 1 步调工具、第 2 步给答案」
      这种时序变得完全确定，才能稳定测试 runner 的循环和错误处理。
    """

    def __init__(self, responses: list[ResponseFactory]) -> None:
        if not responses:
            raise ValueError("ScriptedLLM 至少需要一个响应")
        self._responses = list(responses)
        self.calls: list[dict] = []

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        item = self._responses[index]
        response = item(messages) if callable(item) else item
        # 脚本响应通常不填 model；补上当前请求的 model，方便 trace 一致。
        if not response.model:
            response.model = model
        return response

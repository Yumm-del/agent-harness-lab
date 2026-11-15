# -*- coding: utf-8 -*-
"""Agent loop：模型 ↔ 工具循环，直到给最终答案或用完步数。

这是 harness 的核心。它不负责「任务是什么」，只负责：
  - 组装 system prompt 和工具 schema
  - 按 spec 的上下文策略裁剪历史
  - 执行 tool calls、记录 trace、累计 usage
  - 在工具失败时按配置自动重试或注入修复提示
"""

from __future__ import annotations

import json
import time
from typing import Any

from .llm import LLMClient
from .models import LLMResponse, RunOutput, ToolExecution, Usage
from .spec import AgentSpec
from .tools import ToolRegistry
from .trace import TraceWriter, messages_hash


DEFAULT_RULES = """Operating rules:
- You are in a sandboxed workspace. Use relative paths only.
- Read files before assuming their contents. Do not invent data.
- Use tools step by step; after each tool result, decide the next action.
- When the task is complete, reply with a short final answer and do not call tools.
- If a tool fails, read the error and either correct the arguments or try another approach.
"""


class AgentRunner:
    """一个 spec 对应一个 runner；每个 task episode 调用一次 run()。"""

    def __init__(
        self,
        spec: AgentSpec,
        client: LLMClient,
        registry: ToolRegistry,
        trace: TraceWriter,
    ) -> None:
        self.spec = spec
        self.client = client
        self.registry = registry
        self.trace = trace
        self._tool_defs = registry.definitions(spec.tools)

    def _system_prompt(self) -> str:
        return f"{self.spec.instruction.strip()}\n\n{DEFAULT_RULES}"

    def _apply_context_policy(self, messages: list[dict]) -> None:
        """按 spec 裁剪上下文。

        prune 只替换「旧的 tool observation」的内容，不删除 message。
        原因：OpenAI/DeepSeek 要求 assistant 的每个 tool_call 都有对应的
        tool message；直接删消息会破坏消息序列，下一次请求直接 400。
        """
        if self.spec.context_policy != "prune":
            return

        tool_indices = [i for i, msg in enumerate(messages) if msg.get("role") == "tool"]
        if len(tool_indices) <= self.spec.context_window:
            return

        marker = json.dumps(
            {
                "ok": True,
                "pruned": True,
                "note": "older tool observation removed by context policy",
            },
            ensure_ascii=False,
        )
        for index in tool_indices[: -self.spec.context_window]:
            current = str(messages[index].get("content", ""))
            if "older tool observation removed" not in current:
                messages[index] = {**messages[index], "content": marker}

    @staticmethod
    def _assistant_message(response: LLMResponse) -> dict:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": response.content or None,
        }
        if response.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                }
                for call in response.tool_calls
            ]
        return message

    def run(self, prompt: str, task_id: str, category: str) -> RunOutput:
        """执行一个 task episode。

        输入：
          prompt   — 用户任务描述（不含 system prompt）。
          task_id  — 写 trace 用。
          category — 任务类别，写 trace 用。

        输出：
          RunOutput；是否通过由 evaluator 的 validator 决定，runner 不越权判断。
        """
        started = time.perf_counter()
        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": prompt},
        ]

        usage = Usage()
        tool_calls = 0
        tool_errors = 0
        recovery_events = 0
        last_content = ""
        stopped_reason = "max_steps"

        for _ in range(self.spec.max_steps):
            self._apply_context_policy(messages)
            step = self.trace.next_step()

            try:
                response = self.client.chat(
                    messages,
                    self._tool_defs,
                    model=self.spec.model,
                    temperature=self.spec.temperature,
                    max_tokens=self.spec.max_tokens,
                )
            except Exception as exc:  # noqa: BLE001 —— API/网络错误要记录成 episode 结果
                self.trace.write(
                    "llm_error",
                    error=f"{type(exc).__name__}: {exc}",
                )
                stopped_reason = "llm_error"
                break

            usage.add(response.usage)
            last_content = response.content.strip()
            self.trace.write(
                "llm_call",
                messages=messages,
                request_hash=messages_hash(messages),
                response=response.to_dict(),
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                latency_ms=round(response.latency_ms, 2),
            )

            if not response.tool_calls:
                messages.append({"role": "assistant", "content": last_content})
                stopped_reason = "final"
                break

            # 少数兼容服务不返回 tool_call.id；补一个，保证 assistant 的
            # tool_calls 和下面的 tool message 能一一对应，否则下一次请求会 400。
            for index, call in enumerate(response.tool_calls):
                if not call.id:
                    call.id = f"call_{step}_{index}"
            messages.append(self._assistant_message(response))
            had_error = False

            for index, call in enumerate(response.tool_calls):
                tool_calls += 1
                call_id = call.id or f"call_{step}_{index}"

                try:
                    args = call.parsed_arguments()
                except ValueError as exc:
                    result = ToolExecution(
                        name=call.name,
                        args={},
                        ok=False,
                        content=json.dumps(
                            {"ok": False, "error": str(exc)}, ensure_ascii=False
                        ),
                        error=str(exc),
                        attempts=0,
                        failed_attempts=1,
                        attempt_errors=[str(exc)],
                    )
                else:
                    result = self.registry.execute(
                        call.name,
                        args,
                        retries=self.spec.tool_retries,
                    )

                tool_errors += result.failed_attempts
                if result.ok and result.failed_attempts > 0:
                    # harness 层自动重试把一次失败救了回来
                    recovery_events += 1
                if not result.ok:
                    had_error = True

                self.trace.write(
                    "tool_call",
                    name=call.name,
                    args=result.args,
                    ok=result.ok,
                    content=result.content,
                    error=result.error,
                    attempts=result.attempts,
                    failed_attempts=result.failed_attempts,
                    attempt_errors=result.attempt_errors,
                    latency_ms=round(result.latency_ms, 2),
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": result.content,
                    }
                )

            if had_error and self.spec.repair_on_error:
                hint = (
                    "One or more tool calls failed. Read the error carefully, then "
                    "either retry with corrected arguments or choose a different tool. "
                    "Do not repeat the exact same failing call."
                )
                messages.append({"role": "user", "content": hint})
                self.trace.write("repair_hint", content=hint)

        if stopped_reason == "max_steps":
            self.trace.write(
                "max_steps",
                max_steps=self.spec.max_steps,
                note="model still requested tools when the step budget ran out",
            )

        elapsed_ms = (time.perf_counter() - started) * 1000
        return RunOutput(
            answer=last_content,
            steps=self.trace.step,
            usage=usage,
            latency_ms=elapsed_ms,
            tool_calls=tool_calls,
            tool_errors=tool_errors,
            recovery_events=recovery_events,
            stopped_reason=stopped_reason,
            messages=messages,
        )

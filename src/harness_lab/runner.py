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

# planner 只负责拆步骤，不接触工具。默认提示词放在代码里，
# 便于所有 multi-agent 配置共享同一份基线；YAML 可用 planner_instruction 覆盖。
DEFAULT_PLANNER_PROMPT = """You are the planner in a planner/executor agent.
Read the task and produce a short, concrete plan for the executor.
Rules:
- Do not call tools and do not solve the task yourself.
- Output a numbered plan of 3-6 steps.
- For each step, say which tool to use and what to read/write.
- Call out likely failure points and how to verify the result.
- Do not invent file contents or API data.
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

    def _system_prompt(self, plan: str = "") -> str:
        """executor 的 system prompt；多智能体时把 planner 的计划附在后面。

        计划只是提示，不是事实。把它放在 system prompt 而不是伪造一条
        assistant 消息，是为了保持「system → user(task) → assistant/tool ...」
        的标准消息序列，回放时也更容易对齐。
        """
        prompt = f"{self.spec.instruction.strip()}\n\n{DEFAULT_RULES}"
        if plan:
            prompt += (
                "\n\nPlanner handoff (a hint, not ground truth; verify with tools):\n"
                f"{plan.strip()}"
            )
        return prompt

    def _run_planner(self, prompt: str) -> tuple[str, Usage, bool]:
        """先让 planner 出一份计划，返回 (计划, planner usage, 是否成功)。

        失败策略是「降级而不是中断」：planner 调用失败时返回空计划，
        executor 仍按原来的单 Agent 方式执行。这样多智能体不会因为
        一次额外调用把原本能过的任务拖死。

        注意：即使调用失败也写一条空 response 的 llm_call，
        否则 replay 按 llm_call 顺序取响应时会错位。
        """
        planner_prompt = (
            self.spec.planner_instruction.strip() or DEFAULT_PLANNER_PROMPT
        )
        messages = [
            {"role": "system", "content": planner_prompt},
            {"role": "user", "content": prompt},
        ]
        started = time.perf_counter()

        try:
            response = self.client.chat(
                messages,
                [],  # planner 不允许调用工具
                model=self.spec.model,
                temperature=self.spec.temperature,
                max_tokens=self.spec.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 —— planner 失败要能降级
            latency_ms = (time.perf_counter() - started) * 1000
            error = f"{type(exc).__name__}: {exc}"
            self.trace.write(
                "llm_call",
                agent="planner",
                messages=messages,
                request_hash=messages_hash(messages),
                response={
                    "content": "",
                    "tool_calls": [],
                    "usage": Usage().to_dict(),
                    "latency_ms": round(latency_ms, 2),
                    "model": self.spec.model,
                },
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_ms=round(latency_ms, 2),
                error=error,
            )
            self.trace.write("llm_error", agent="planner", error=error)
            self.trace.write(
                "handoff",
                from_agent="planner",
                to_agent="executor",
                ok=False,
                plan="",
                error=error,
            )
            return "", Usage(), False

        plan = response.content.strip()
        self.trace.write(
            "llm_call",
            agent="planner",
            messages=messages,
            request_hash=messages_hash(messages),
            response=response.to_dict(),
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            latency_ms=round(response.latency_ms, 2),
        )
        self.trace.write(
            "handoff",
            from_agent="planner",
            to_agent="executor",
            ok=bool(plan),
            plan=plan,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            latency_ms=round(response.latency_ms, 2),
        )
        return plan, response.usage, bool(plan)

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
        usage = Usage()
        plan = ""
        planner_tokens = 0
        handoff_count = 0
        handoff_ok = False

        # 多智能体只在主循环前多一次 planner 调用；executor 的 loop 与单 Agent 完全相同。
        if self.spec.architecture == "planner_executor":
            plan, planner_usage, handoff_ok = self._run_planner(prompt)
            usage.add(planner_usage)
            planner_tokens = planner_usage.total_tokens
            handoff_count = 1

        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt(plan)},
            {"role": "user", "content": prompt},
        ]

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
                    agent="executor",
                    error=f"{type(exc).__name__}: {exc}",
                )
                stopped_reason = "llm_error"
                break

            usage.add(response.usage)
            last_content = response.content.strip()
            self.trace.write(
                "llm_call",
                agent="executor",
                messages=messages,
                request_hash=messages_hash(messages),
                response=response.to_dict(),
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
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
                    agent="executor",
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
                self.trace.write("repair_hint", agent="executor", content=hint)

        if stopped_reason == "max_steps":
            self.trace.write(
                "max_steps",
                agent="executor",
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
            planner_tokens=planner_tokens,
            handoff_count=handoff_count,
            handoff_ok=handoff_ok,
            messages=messages,
        )

# -*- coding: utf-8 -*-
"""Agent 可调用的工具、沙箱和故障注入。

设计原则：
  1. 工具只接受相对路径，统一由 Sandbox 做边界检查，模型不能读写工作区外；
  2. 工具失败返回结构化 JSON（ok/error），而不是让 Python 异常冒到模型；
  3. 故障注入是「评测装置」而不是工具本身：同一个工具在 baseline/repair
     两种 harness 下可以表现不同，才能做对照实验。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from .models import ToolExecution


class ToolError(RuntimeError):
    """工具层的可预期错误：路径越界、404、超时、参数不合法等。"""


@dataclass
class ToolResult:
    """工具执行成功/失败的统一返回。"""

    ok: bool
    content: str
    meta: dict = field(default_factory=dict)

    @classmethod
    def ok_payload(cls, payload: Any) -> "ToolResult":
        """把任意可 JSON 序列化的对象转成模型可读的字符串。"""
        return cls(
            ok=True,
            content=json.dumps(payload, ensure_ascii=False, default=str),
        )

    @classmethod
    def failure(cls, message: str) -> "ToolResult":
        return cls(
            ok=False,
            content=json.dumps({"ok": False, "error": message}, ensure_ascii=False),
        )


@dataclass(frozen=True)
class Tool:
    """一个工具的元数据 + 实现。"""

    name: str
    description: str
    parameters: dict
    func: Callable[..., ToolResult]

    def definition(self) -> dict:
        """OpenAI Function Calling 的工具 schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Sandbox:
    """把任务限制在一个临时工作目录内。

    注意：这是本地实验沙箱，不是生产级安全沙箱。
    `run_python` 用子进程 + 超时 + 隔离模式降低风险，但没有做系统调用级隔离，
    也不能阻止 Python 访问网络；README 里会明确写这个边界。
    """

    MAX_READ_BYTES = 500_000
    MAX_WRITE_BYTES = 200_000
    MAX_LIST_ENTRIES = 200
    PYTHON_TIMEOUT_SEC = 8

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative: str) -> Path:
        """把相对路径解析到工作区内；绝对路径和 `..` 越界都拒绝。"""
        if not isinstance(relative, str) or not relative.strip():
            raise ToolError("path 必须是非空字符串")
        raw = Path(relative)
        if raw.is_absolute():
            raise ToolError("只允许相对路径，不能使用绝对路径")
        target = (self.root / raw).resolve()
        if target != self.root and not target.is_relative_to(self.root):
            raise ToolError("路径越界：不能访问工作区外的文件")
        return target

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def read_file(self, path: str) -> ToolResult:
        target = self.resolve(path)
        if not target.exists():
            raise ToolError(f"文件不存在：{path}")
        if not target.is_file():
            raise ToolError(f"不是文件：{path}")
        if target.stat().st_size > self.MAX_READ_BYTES:
            raise ToolError(f"文件太大（>{self.MAX_READ_BYTES} bytes）：{path}")
        content = target.read_bytes().decode("utf-8", errors="replace")
        return ToolResult.ok_payload(
            {"path": self._relative(target), "content": content, "bytes": target.stat().st_size}
        )

    def write_file(self, path: str, content: str) -> ToolResult:
        if not isinstance(content, str):
            raise ToolError("content 必须是字符串")
        if len(content.encode("utf-8")) > self.MAX_WRITE_BYTES:
            raise ToolError(f"写入内容太大（>{self.MAX_WRITE_BYTES} bytes）")
        target = self.resolve(path)
        if target == self.root:
            raise ToolError("不能把文件写到工作区根目录本身")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return ToolResult.ok_payload(
            {"path": self._relative(target), "bytes": len(content.encode("utf-8"))}
        )

    def list_files(self, path: str = ".") -> ToolResult:
        target = self.resolve(path)
        if not target.exists():
            raise ToolError(f"路径不存在：{path}")
        if target.is_file():
            files = [self._relative(target)]
        else:
            files = []
            for item in sorted(target.rglob("*")):
                if item.is_file():
                    try:
                        resolved = item.resolve()
                    except OSError:
                        continue
                    if resolved.is_relative_to(self.root):
                        files.append(self._relative(resolved))
                if len(files) >= self.MAX_LIST_ENTRIES:
                    break
        return ToolResult.ok_payload({"path": path, "files": files, "count": len(files)})

    def run_python(self, code: str) -> ToolResult:
        """在子进程中执行一段 Python，返回 stdout/stderr。

        为什么用子进程而不是 exec()：
          exec() 和 Agent loop 共享内存，一段死循环或 os._exit 会拖垮整个评测；
          子进程可以超时杀掉，失败也只影响当前 episode。
        """
        if not isinstance(code, str) or not code.strip():
            raise ToolError("code 必须是非空字符串")
        if len(code) > 20_000:
            raise ToolError("code 太长（>20000 字符）")

        # 最小环境：保留 Windows 启动 Python 必需的变量，去掉绝大多数自定义变量。
        env = {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "PATH": os.environ.get("PATH", ""),
        }
        for key in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP"):
            if key in os.environ:
                env[key] = os.environ[key]

        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=self.root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.PYTHON_TIMEOUT_SEC,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(
                f"Python 执行超时（>{self.PYTHON_TIMEOUT_SEC}s），已终止"
            ) from exc

        stdout = (completed.stdout or "")[:4000]
        stderr = (completed.stderr or "")[:4000]
        return ToolResult.ok_payload(
            {
                "returncode": completed.returncode,
                "stdout": stdout,
                "stderr": stderr,
            }
        )


class MockAPI:
    """确定性的 mock HTTP 服务。

    不启真实端口，避免端口冲突和网络不稳定；工具接口保持 `http_get(url)`，
    以后要换成真实 HTTP 只需替换这个类。
    """

    def __init__(self, routes: dict[str, Any]) -> None:
        self.routes = dict(routes)

    @staticmethod
    def _key(url: str) -> str:
        parsed = urlparse(url)
        key = parsed.path or url
        if parsed.query:
            key = f"{key}?{parsed.query}"
        return key

    def get(self, url: str) -> ToolResult:
        key = self._key(url)
        if key not in self.routes:
            available = ", ".join(sorted(self.routes)) or "(none)"
            raise ToolError(f"404 Not Found: {key}；可用端点：{available}")
        return ToolResult.ok_payload(self.routes[key])


@dataclass(frozen=True)
class FaultRule:
    """在某个工具的前 N 次调用上注入故障。"""

    tool: str
    mode: str = "timeout"  # timeout | error
    times: int = 1
    message: str = ""


class FaultInjector:
    """按规则消耗故障次数；每个 episode 新建一个实例。"""

    def __init__(self, rules: list[FaultRule] | None = None) -> None:
        self._rules = list(rules or [])
        self._used = [0] * len(self._rules)

    def maybe_fail(self, tool_name: str) -> None:
        for index, rule in enumerate(self._rules):
            if rule.tool != tool_name or self._used[index] >= rule.times:
                continue
            self._used[index] += 1
            if rule.mode == "timeout":
                raise ToolError(rule.message or f"工具 {tool_name} 调用超时（注入故障）")
            raise ToolError(rule.message or f"工具 {tool_name} 返回错误（注入故障）")


class ToolRegistry:
    """工具注册表：schema 给模型，execute 给 runner。"""

    def __init__(
        self,
        tools: dict[str, Tool],
        sandbox: Sandbox,
        mock_api: MockAPI,
        fault_injector: FaultInjector | None = None,
    ) -> None:
        self.tools = dict(tools)
        self.sandbox = sandbox
        self.mock_api = mock_api
        self.fault_injector = fault_injector or FaultInjector()

    def definitions(self, names: list[str] | tuple[str, ...]) -> list[dict]:
        missing = [name for name in names if name not in self.tools]
        if missing:
            raise ValueError(f"未知工具：{missing}；已注册：{sorted(self.tools)}")
        return [self.tools[name].definition() for name in names]

    def execute(self, name: str, args: dict, retries: int = 0) -> ToolExecution:
        """执行工具，按 retries 自动重试；所有尝试都记录在返回值里。"""
        started = time.perf_counter()
        tool = self.tools.get(name)
        if tool is None:
            return ToolExecution(
                name=name,
                args=args,
                ok=False,
                content=json.dumps({"ok": False, "error": f"未知工具：{name}"}, ensure_ascii=False),
                error=f"未知工具：{name}",
                latency_ms=(time.perf_counter() - started) * 1000,
                attempts=0,
            )

        if not isinstance(args, dict):
            return ToolExecution(
                name=name,
                args={},
                ok=False,
                content=json.dumps({"ok": False, "error": "工具参数必须是对象"}, ensure_ascii=False),
                error="工具参数必须是对象",
                latency_ms=(time.perf_counter() - started) * 1000,
                attempts=0,
            )

        attempt_errors: list[str] = []
        attempts = max(0, retries) + 1
        for _ in range(attempts):
            try:
                self.fault_injector.maybe_fail(name)
                result = tool.func(**args)
                if not isinstance(result, ToolResult):
                    raise ToolError("工具实现没有返回 ToolResult")
                latency = (time.perf_counter() - started) * 1000
                return ToolExecution(
                    name=name,
                    args=args,
                    ok=result.ok,
                    content=result.content,
                    error=None if result.ok else _error_from_content(result.content),
                    latency_ms=latency,
                    attempts=len(attempt_errors) + 1,
                    failed_attempts=len(attempt_errors),
                    attempt_errors=attempt_errors,
                )
            except TypeError as exc:
                # 常见于模型少传/多传参数；直接告诉它 schema 不匹配。
                attempt_errors.append(f"参数不匹配：{exc}")
            except ToolError as exc:
                attempt_errors.append(str(exc))
            except Exception as exc:  # noqa: BLE001 —— 工具异常不能拖垮整个评测
                attempt_errors.append(f"{type(exc).__name__}: {exc}")

        last_error = attempt_errors[-1] if attempt_errors else "未知工具错误"
        return ToolExecution(
            name=name,
            args=args,
            ok=False,
            content=json.dumps({"ok": False, "error": last_error}, ensure_ascii=False),
            error=last_error,
            latency_ms=(time.perf_counter() - started) * 1000,
            attempts=len(attempt_errors),
            failed_attempts=len(attempt_errors),
            attempt_errors=attempt_errors,
        )


def _error_from_content(content: str) -> str | None:
    """从工具返回的 JSON 里取 error 字段，纯粹为了让 trace 更好读。"""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict) and payload.get("ok") is False:
        return str(payload.get("error", "")) or None
    return None


def _schema(properties: dict[str, dict], required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def build_default_registry(
    sandbox: Sandbox,
    mock_api: MockAPI,
    fault_injector: FaultInjector | None = None,
) -> ToolRegistry:
    """注册本实验场使用的 5 个基础工具。

    每个工具的描述都写清楚「返回什么格式」，减少模型靠猜导致的格式错误。
    """
    tools = {
        "list_files": Tool(
            name="list_files",
            description=(
                "List files under a relative directory in the workspace. "
                "Returns JSON: {path, files: [...], count}."
            ),
            parameters=_schema(
                {"path": {"type": "string", "description": "Relative directory, default '.'"}},
                [],
            ),
            func=sandbox.list_files,
        ),
        "read_file": Tool(
            name="read_file",
            description=(
                "Read a UTF-8 text file in the workspace. "
                "Returns JSON: {path, content, bytes}. Read before assuming file contents."
            ),
            parameters=_schema(
                {"path": {"type": "string", "description": "Relative file path"}},
                ["path"],
            ),
            func=sandbox.read_file,
        ),
        "write_file": Tool(
            name="write_file",
            description=(
                "Write a UTF-8 text file in the workspace, creating parent directories. "
                "Returns JSON: {path, bytes}."
            ),
            parameters=_schema(
                {
                    "path": {"type": "string", "description": "Relative file path"},
                    "content": {"type": "string", "description": "Exact file content"},
                },
                ["path", "content"],
            ),
            func=sandbox.write_file,
        ),
        "http_get": Tool(
            name="http_get",
            description=(
                "GET a mock HTTP endpoint. URLs look like mock://api/orders. "
                "Returns the endpoint's JSON payload. Endpoints are case-sensitive."
            ),
            parameters=_schema(
                {"url": {"type": "string", "description": "mock://api/... URL"}},
                ["url"],
            ),
            func=mock_api.get,
        ),
        "run_python": Tool(
            name="run_python",
            description=(
                "Run a short Python 3 script in the workspace with a timeout. "
                "stdout/stderr are captured. Use it for calculations or data processing. "
                "The script can only see files in the current workspace."
            ),
            parameters=_schema(
                {"code": {"type": "string", "description": "Python source code"}},
                ["code"],
            ),
            func=sandbox.run_python,
        ),
    }
    return ToolRegistry(tools, sandbox, mock_api, fault_injector)

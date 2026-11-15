# -*- coding: utf-8 -*-
"""可复现任务集。

任务设计原则：
  1. 每个任务都有确定性 validator，不靠 LLM 裁判判对错；
  2. 输入、输出、边界都写死在 prompt 里，换模型/换 harness 仍可比较；
  3. 覆盖 4 类能力：文件与数据、多步工具、错误恢复、长链路；
  4. 任务规模小（几秒到几十秒），适合反复跑 repeat，量出噪声。

注意：这些是合成任务，不是生产 benchmark。它们的作用是让 harness 的
「成功率 / 步数 / 错误恢复 / 上下文压力」有可观察的差异，而不是证明
模型在真实业务上有多强。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .tools import FaultRule

Validator = Callable[[Path, str], tuple[bool, str]]


# ---------------------------------------------------------------------------
# validator 小工具
# ---------------------------------------------------------------------------


def _short(value: Any, limit: int = 220) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _json_equals(path: str, expected: Any) -> Validator:
    """读取 JSON 文件并做精确比较（float/int 在 Python 里可相等）。"""

    def validate(workspace: Path, answer: str) -> tuple[bool, str]:
        target = workspace / path
        if not target.exists():
            return False, f"缺少输出文件：{path}"
        try:
            actual = json.loads(target.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            return False, f"{path} 不是合法 JSON：{exc}"
        if actual == expected:
            return True, f"{path} 与期望一致"
        return False, f"{path} 不一致：期望 {_short(expected)}，实际 {_short(actual)}"

    return validate


def _text_equals(path: str, expected: str) -> Validator:
    """比较文本文件（忽略首尾空白）。"""

    def validate(workspace: Path, answer: str) -> tuple[bool, str]:
        target = workspace / path
        if not target.exists():
            return False, f"缺少输出文件：{path}"
        actual = target.read_text(encoding="utf-8", errors="replace").strip()
        want = expected.strip()
        if actual == want:
            return True, f"{path} 内容正确"
        return False, f"{path} 不一致：期望 {_short(want)}，实际 {_short(actual)}"

    return validate


def _text_number(path: str, expected: float, tol: float = 1e-6) -> Validator:
    """比较文本文件里的单个数字。"""

    def validate(workspace: Path, answer: str) -> tuple[bool, str]:
        target = workspace / path
        if not target.exists():
            return False, f"缺少输出文件：{path}"
        raw = target.read_text(encoding="utf-8", errors="replace").strip()
        try:
            actual = float(raw)
        except ValueError:
            return False, f"{path} 不是数字：{_short(raw)}"
        if abs(actual - expected) <= tol:
            return True, f"{path} 数值正确（{actual}）"
        return False, f"{path} 数值不符：期望 {expected}，实际 {actual}"

    return validate


def _filtered_and_count() -> Validator:
    """lh_01 专用：同时检查 filtered.json 和 count.txt。"""

    check_json = _json_equals(
        "filtered.json",
        [
            {"id": 2, "amount": 60},
            {"id": 3, "amount": 75},
            {"id": 5, "amount": 90},
            {"id": 6, "amount": 55},
        ],
    )
    check_count = _text_equals("count.txt", "4")

    def validate(workspace: Path, answer: str) -> tuple[bool, str]:
        ok_json, detail_json = check_json(workspace, answer)
        if not ok_json:
            return False, detail_json
        return check_count(workspace, answer)

    return validate


# ---------------------------------------------------------------------------
# Task 定义
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    difficulty: str
    prompt: str
    fixtures: dict[str, str]
    validator: Validator
    mock_api: dict[str, Any] = field(default_factory=dict)
    faults: tuple[FaultRule, ...] = ()
    max_steps: int = 8


SALES_CSV = """id,region,amount
1,East,120.5
2,West,80
3,East,99.5
4,North,200
5,East,30
6,West,15
"""

CONFIG_JSON = """{
  "database": {"host": "db.internal", "port": 5432},
  "cache": {"ttl": 60}
}
"""

APP_LOG = """2026-01-01 INFO start
2026-01-01 ERROR disk full
2026-01-01 WARN low memory
2026-01-01 ERROR timeout
2026-01-02 INFO retry
2026-01-02 ERROR auth failed
"""

A_CSV = """id,name
1,Alice
2,Bob
3,Cara
"""

B_CSV = """id,score
1,90
2,85
4,70
"""

RAW_CSV = """id,amount
1,10
2,60
3,75
4,20
5,90
6,55
"""

JAN_CSV = """amount
10
20
"""

FEB_CSV = """amount
5
15
20
"""

MAR_CSV = """amount
100
"""

A_TXT = "1\n2\n3\n4\n5\n"
B_TXT = "2\n4\n6\n"

RULES_JSON = """{"mode": "double", "endpoint": "/numbers"}
"""

NUMBERS_API = [1, 2, 3]

LH_LOG = """2026-01-01 ERROR E42 disk
2026-01-01 INFO ok
2026-01-02 ERROR E17 auth
2026-01-02 ERROR E42 disk
2026-01-03 WARN E42 slow
"""

CODES_JSON = """{"E42": "disk failure", "E17": "auth failure"}
"""


TASKS: list[Task] = [
    # ------------------------------------------------------------------
    # 文件与数据：先读再算，考察基本工具使用和格式遵守
    # ------------------------------------------------------------------
    Task(
        id="fd_01_csv_filter",
        category="file_data",
        difficulty="easy",
        prompt=(
            "Read sales.csv. Keep only rows where region is 'East'. "
            "Write result.json with exactly two keys: rows (integer count of kept rows) "
            "and total (sum of their amount). Do not add extra keys."
        ),
        fixtures={"sales.csv": SALES_CSV},
        validator=_json_equals("result.json", {"rows": 3, "total": 250.0}),
    ),
    Task(
        id="fd_02_json_extract",
        category="file_data",
        difficulty="easy",
        prompt=(
            "Read config.json. Write db.txt containing exactly the database host, "
            "a colon, and the port, with no spaces (for example host:1234)."
        ),
        fixtures={"config.json": CONFIG_JSON},
        validator=_text_equals("db.txt", "db.internal:5432"),
    ),
    Task(
        id="fd_03_log_count",
        category="file_data",
        difficulty="easy",
        prompt=(
            "Read app.log. Count the lines whose level is ERROR. "
            "Write count.txt containing only that integer."
        ),
        fixtures={"app.log": APP_LOG},
        validator=_text_equals("count.txt", "3"),
    ),
    Task(
        id="fd_04_merge_csv",
        category="file_data",
        difficulty="medium",
        prompt=(
            "a.csv has columns id,name. b.csv has columns id,score. "
            "Inner-join them on id, keeping only ids present in both files. "
            "Write merged.json as a list of objects with keys id, name, score, "
            "sorted by id ascending. Numbers must stay numbers."
        ),
        fixtures={"a.csv": A_CSV, "b.csv": B_CSV},
        validator=_json_equals(
            "merged.json",
            [
                {"id": 1, "name": "Alice", "score": 90},
                {"id": 2, "name": "Bob", "score": 85},
            ],
        ),
    ),
    # ------------------------------------------------------------------
    # 多步工具：读配置 → 调 API → 计算 → 写文件
    # ------------------------------------------------------------------
    Task(
        id="ms_01_api_sum",
        category="multi_step",
        difficulty="easy",
        prompt=(
            "GET mock://api/orders. Sum amount for orders whose status is 'paid'. "
            "Write paid_total.txt containing only the number."
        ),
        fixtures={},
        mock_api={
            "/orders": [
                {"id": 1, "status": "paid", "amount": 100},
                {"id": 2, "status": "pending", "amount": 50},
                {"id": 3, "status": "paid", "amount": 25.5},
            ]
        },
        validator=_text_number("paid_total.txt", 125.5),
    ),
    Task(
        id="ms_02_api_join",
        category="multi_step",
        difficulty="medium",
        prompt=(
            "GET mock://api/users and mock://api/orders. Join orders to users on user_id, "
            "sum each user's order amount, and find the user with the highest total. "
            "Write report.json with exactly keys top_user (name) and total (number)."
        ),
        fixtures={},
        mock_api={
            "/users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            "/orders": [
                {"user_id": 1, "amount": 100},
                {"user_id": 1, "amount": 50},
                {"user_id": 2, "amount": 200},
            ],
        },
        validator=_json_equals("report.json", {"top_user": "Bob", "total": 200}),
    ),
    Task(
        id="ms_03_config_then_api",
        category="multi_step",
        difficulty="medium",
        prompt=(
            "Read settings.json. It contains an endpoint path and a numeric threshold. "
            "GET mock://api followed by that endpoint. Keep only records whose value is "
            "strictly greater than the threshold, and write their names as a JSON array "
            "sorted alphabetically to alerts.json."
        ),
        fixtures={"settings.json": '{"endpoint": "/metrics", "threshold": 10}\n'},
        mock_api={
            "/metrics": [
                {"name": "cpu", "value": 12},
                {"name": "mem", "value": 8},
                {"name": "disk", "value": 15},
            ]
        },
        validator=_json_equals("alerts.json", ["cpu", "disk"]),
    ),
    Task(
        id="ms_04_monthly_totals",
        category="multi_step",
        difficulty="medium",
        prompt=(
            "Read jan.csv, feb.csv, and mar.csv. Each has one column named amount. "
            "Compute the sum for each month and write summary.json with exactly keys "
            "jan, feb, mar; values are numbers."
        ),
        fixtures={"jan.csv": JAN_CSV, "feb.csv": FEB_CSV, "mar.csv": MAR_CSV},
        validator=_json_equals("summary.json", {"jan": 30, "feb": 40, "mar": 100}),
    ),
    # ------------------------------------------------------------------
    # 错误恢复：工具失败、脏数据、文件缺失、参数错误
    # ------------------------------------------------------------------
    Task(
        id="rc_01_timeout_retry",
        category="recovery",
        difficulty="hard",
        prompt=(
            "GET mock://api/flaky. The endpoint may be temporarily unavailable; retry if needed. "
            "Write the returned value to value.txt containing only the number."
        ),
        fixtures={},
        mock_api={"/flaky": {"value": 42}},
        faults=(FaultRule(tool="http_get", mode="timeout", times=2),),
        # 故意把步数压到 3：baseline 没有自动重试，会在两次失败后耗尽步数；
        # repair 配置在 harness 层自动重试，能在第 1 步拿到结果，再写文件。
        max_steps=3,
        validator=_text_number("value.txt", 42),
    ),
    Task(
        id="rc_02_dirty_records",
        category="recovery",
        difficulty="medium",
        prompt=(
            "GET mock://api/items. Some records may have a non-numeric amount. "
            "Sum amount only for records where amount is a number, skip malformed records, "
            "and write valid_total.txt containing only the number."
        ),
        fixtures={},
        mock_api={
            "/items": [
                {"id": 1, "amount": 10},
                {"id": 2, "amount": "oops"},
                {"id": 3, "amount": 30},
            ]
        },
        validator=_text_number("valid_total.txt", 40),
    ),
    Task(
        id="rc_03_missing_file_fallback",
        category="recovery",
        difficulty="medium",
        prompt=(
            "Read primary.txt. If it does not exist, read backup.txt instead. "
            "Write the content you read to result.txt."
        ),
        fixtures={"backup.txt": "fallback-value\n"},
        validator=_text_equals("result.txt", "fallback-value"),
    ),
    Task(
        id="rc_04_case_sensitive_api",
        category="recovery",
        difficulty="hard",
        prompt=(
            "GET mock://api/weather?city=beijing. The API is case-sensitive: if the request "
            "returns 404, correct the city name and retry. Write the returned temperature "
            "to temp.txt containing only the number."
        ),
        fixtures={},
        mock_api={"/weather?city=Beijing": {"temp": 22}},
        validator=_text_number("temp.txt", 22),
    ),
    # ------------------------------------------------------------------
    # 长链路：需要多步组合，容易在中途丢上下文
    # ------------------------------------------------------------------
    Task(
        id="lh_01_filter_then_count",
        category="long_horizon",
        difficulty="medium",
        prompt=(
            "Read raw.csv. Keep rows whose amount is greater than 50 and write them to "
            "filtered.json as a list of objects in the original order, with keys id and amount. "
            "id and amount must be numeric JSON values, not strings. Then read filtered.json "
            "and write the number of rows to count.txt."
        ),
        fixtures={"raw.csv": RAW_CSV},
        validator=_filtered_and_count(),
    ),
    Task(
        id="lh_02_missing_ids",
        category="long_horizon",
        difficulty="medium",
        prompt=(
            "a.txt and b.txt each contain one integer per line. Find the integers that appear "
            "in a.txt but not in b.txt. Write them to missing.txt, one per line, sorted ascending."
        ),
        fixtures={"a.txt": A_TXT, "b.txt": B_TXT},
        validator=_text_equals("missing.txt", "1\n3\n5"),
    ),
    Task(
        id="lh_03_rule_branch",
        category="long_horizon",
        difficulty="hard",
        prompt=(
            "Read rules.json. It has mode and endpoint. GET mock://api followed by endpoint. "
            "If mode is 'double', multiply each number by 2 and sum the results; if mode is "
            "'sum', just sum the numbers. Write the final number to result.txt."
        ),
        fixtures={"rules.json": RULES_JSON},
        mock_api={"/numbers": NUMBERS_API},
        validator=_text_number("result.txt", 12),
    ),
    Task(
        id="lh_04_error_report",
        category="long_horizon",
        difficulty="hard",
        prompt=(
            "Read app.log and codes.json. Each log line has the form: date level code message. "
            "Consider only lines whose level is ERROR, count them by code, find the most "
            "frequent code, map it through codes.json, and write report.json with exactly "
            "keys code, count, description."
        ),
        fixtures={"app.log": LH_LOG, "codes.json": CODES_JSON},
        validator=_json_equals(
            "report.json",
            {"code": "E42", "count": 2, "description": "disk failure"},
        ),
    ),
]


def get_task(task_id: str) -> Task:
    for task in TASKS:
        if task.id == task_id:
            return task
    raise KeyError(f"未知任务：{task_id}")


def select_tasks(task_ids: list[str] | None = None) -> list[Task]:
    """按 id 过滤；None/空列表表示全部。"""
    if not task_ids:
        return list(TASKS)
    return [get_task(task_id) for task_id in task_ids]


def task_summary() -> dict[str, int]:
    """给 CLI 的 list-tasks 用：每个类别多少个任务。"""
    summary: dict[str, int] = {}
    for task in TASKS:
        summary[task.category] = summary.get(task.category, 0) + 1
    return summary

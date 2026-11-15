# -*- coding: utf-8 -*-
"""把 results 渲染成 Markdown 报告。

报告不追求花哨，只回答四个问题：
  1. 哪个 harness 配置成功率最高？
  2. 差异有没有超过 repeat 之间的噪声？
  3. 成本/延迟/步数付出了多少？
  4. 失败集中在哪一类任务、哪一种失败模式？
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .evaluate import aggregate
from .models import EpisodeResult


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _num(value: float) -> str:
    return f"{value:.1f}"


def _md_escape(text: str, limit: int = 120) -> str:
    clean = str(text).replace("|", "\\|").replace("\n", " ").strip()
    return clean if len(clean) <= limit else clean[:limit] + "..."


def render_report(
    meta: dict[str, Any],
    results: list[EpisodeResult],
    path: str | Path,
) -> Path:
    summary = aggregate(results)
    specs = [spec["name"] for spec in meta.get("specs", [])] or list(summary)
    categories = sorted(
        {item.category for item in results}
    )

    lines: list[str] = []
    lines.append(f"# agent-harness-lab · {meta.get('run_id', 'run')}")
    lines.append("")
    lines.append(
        f"- 开始时间：{meta.get('started_at', '')}；总耗时：{meta.get('duration_sec', 0)}s"
    )
    lines.append(
        f"- 配置：{', '.join(s['name'] for s in meta.get('specs', [])) or '—'}；"
        f"任务数：{len(meta.get('tasks', []))}；重复次数：{meta.get('repeats', 0)}"
    )
    lines.append("")
    lines.append("## 总览")
    lines.append("")
    lines.append(
        "| 配置 | 成功率 | repeat 标准差 | 平均步数 | 平均 token | planner token | handoff（成功/次数） | 缓存命中率 | P50 延迟 | P95 延迟 | 工具错误率 | recovery 成功率 | 升级率 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in specs:
        item = summary.get(name)
        if not item:
            continue
        handoff_count = item.get("handoff_count", 0)
        handoff_cell = (
            f"{item.get('handoff_ok_count', 0)}/{handoff_count}"
            if handoff_count
            else "—"
        )
        lines.append(
            f"| {name} | {_pct(item['success_rate'])} | {_pct(item['success_std'])} | "
            f"{_num(item['avg_steps'])} | {item['avg_tokens']:.0f} | "
            f"{item.get('avg_planner_tokens', 0.0):.0f} | {handoff_cell} | "
            f"{_pct(item.get('cache_hit_rate', 0.0))} | "
            f"{item['p50_latency_ms']:.0f} ms | {item['p95_latency_ms']:.0f} ms | "
            f"{_pct(item['tool_error_rate'])} | "
            f"{_pct(item['recovery_success'])} ({item['recovery_passed']}/{item['recovery_total']}) | "
            f"{_pct(item['escalation_rate'])} |"
        )

    lines.append("")
    lines.append("## 分类任务成功率")
    lines.append("")
    header = "| 配置 | " + " | ".join(categories) + " |"
    divider = "|---|" + "|".join(["---:"] * len(categories)) + "|"
    lines.append(header)
    lines.append(divider)
    for name in specs:
        item = summary.get(name)
        if not item:
            continue
        cells = []
        for category in categories:
            bucket = item["categories"].get(category)
            cells.append(_pct(bucket["success_rate"]) if bucket else "—")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## 失败分布")
    lines.append("")
    lines.append("| 配置 | 失败类型 | 次数 |")
    lines.append("|---|---|---:|")
    any_failure = False
    for name in specs:
        item = summary.get(name)
        if not item:
            continue
        for failure_type, count in sorted(item["failures"].items(), key=lambda kv: -kv[1]):
            any_failure = True
            lines.append(f"| {name} | {failure_type} | {count} |")
    if not any_failure:
        lines.append("| — | 无失败 | 0 |")

    lines.append("")
    lines.append("## 失败样本（最多 40 条）")
    lines.append("")
    lines.append("| 配置 | 任务 | repeat | 类型 | 详情 | trace |")
    lines.append("|---|---|---:|---|---|---|")
    failed = [item for item in results if not item.passed]
    for item in failed[:40]:
        lines.append(
            f"| {item.spec_name} | {item.task_id} | {item.repeat + 1} | "
            f"{item.failure_type} | {_md_escape(item.detail)} | "
            f"`{item.trace_path}` |"
        )
    if not failed:
        lines.append("| — | — | — | — | 本次没有失败 | — |")
    if len(failed) > 40:
        lines.append(f"| … | 其余 {len(failed) - 40} 条 | | | | |")

    lines.append("")
    lines.append("## 读表注意")
    lines.append("")
    lines.append(
        "- 这是合成任务集，成功率只能说明该配置在这 16 个任务上的表现，不能外推成真实业务能力。"
    )
    lines.append(
        "- `repeat 标准差` 是各 repeat 通过率的标准差；样本小时它只是噪声提示，不是置信区间。"
    )
    lines.append(
        "- `工具错误率` = 失败尝试次数 / 工具调用次数；开启自动重试后可能超过 100%，这是预期的。"
    )
    lines.append(
        "- `planner token` 只统计 planner 调用，是总 token 的一部分；"
        "`handoff` 是 planner→executor 的交接（成功 episode / 有交接的 episode），单 Agent 配置显示 —。"
    )
    lines.append(
        "- planner 调用失败会降级为无计划的 executor，并在 trace 里记 `llm_error`；"
        "如果 episode 最终失败，归因会标为 `planner_error`。"
    )
    lines.append(
        "- `缓存命中率` = DeepSeek 返回的 `prompt_cache_hit_tokens` /（命中 + 未命中）；"
        "服务未返回该字段时显示 0.0%，不能据此推断「没有缓存」。"
    )
    lines.append(
        "- 每个 episode 的完整 messages、tool call、错误与 token 都在对应 `trace.jsonl` 里。"
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

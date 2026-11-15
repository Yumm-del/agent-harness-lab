# -*- coding: utf-8 -*-
"""实验执行与聚合。

一次 run 的结构：
  runs/<run_id>/<spec>/<task>/repeat_<n>/
      workspace/       任务输入与 Agent 写出的文件
      trace.jsonl      完整运行轨迹

聚合层只读 EpisodeResult，不重新跑模型；这样报告可以反复重算，
也方便以后把 results.json 拿去做别的分析。
"""

from __future__ import annotations

import json
import math
import shutil
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .llm import LLMClient
from .models import EpisodeResult, RunOutput, Usage
from .runner import AgentRunner
from .spec import AgentSpec
from .tasks import Task
from .tools import FaultInjector, MockAPI, Sandbox, build_default_registry
from .trace import TraceWriter, new_episode_id, new_run_id, read_trace, trace_stats


def _prepare_workspace(task: Task, workspace: Path) -> None:
    """清空并重建 episode 工作区，然后写入任务 fixtures。"""
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    for relative, content in task.fixtures.items():
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _percentile(values: list[float], pct: float) -> float:
    """线性插值百分位；空列表返回 0。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def classify_failure(
    output: RunOutput,
    passed: bool,
    events: list[dict],
    detail: str,
) -> str:
    """把失败 episode 归到一个可解释的类别。

    这不是「模型自评」，而是基于 trace 的规则：
      先看有没有跑完，再看工具错误，最后看输出格式/内容。
    规则不完美，但每个判断都能在 trace 里找到证据，面试时可以逐条解释。
    """
    if passed:
        return "none"

    stats = trace_stats(events)
    if output.stopped_reason == "llm_error":
        return "llm_error"
    # planner 没产出计划时 executor 是降级运行的；如果最终失败，
    # 根因应归到交接层，而不是笼统地归到 executor 的规划/输出问题。
    if output.handoff_count > 0 and not output.handoff_ok:
        return "planner_error"
    if output.stopped_reason == "max_steps":
        return "step_limit"
    if stats["llm_calls"] == 0:
        return "harness_error"
    if output.tool_calls == 0:
        return "planning_no_tool"
    if stats["repeated_failed_actions"] > 0:
        return "recovery_loop"
    if stats["tool_errors"] > 0:
        return "tool_failure"
    if "缺少输出文件" in detail:
        return "missing_artifact"
    if "不是合法 JSON" in detail:
        return "format_error"
    if not output.answer:
        return "no_final_answer"
    return "execution_wrong_result"


def run_episode(
    task: Task,
    spec: AgentSpec,
    client: LLMClient,
    run_root: Path,
    repeat: int,
) -> EpisodeResult:
    """跑一个 task × spec × repeat，返回结构化结果。"""
    # 任务可以压低步数上限（例如 rc_01 故意让 baseline 耗尽步数）；
    # 但不能超过 spec 自己的预算。
    effective_spec = (
        replace(spec, max_steps=min(spec.max_steps, task.max_steps))
        if task.max_steps < spec.max_steps
        else spec
    )

    episode_id = new_episode_id()
    episode_dir = run_root / spec.name / task.id / f"repeat_{repeat}"
    workspace = episode_dir / "workspace"
    trace_path = episode_dir / "trace.jsonl"
    _prepare_workspace(task, workspace)

    sandbox = Sandbox(workspace)
    registry = build_default_registry(
        sandbox,
        MockAPI(task.mock_api),
        FaultInjector(list(task.faults)),
    )

    with TraceWriter(
        trace_path,
        episode_id=episode_id,
        task_id=task.id,
        category=task.category,
        prompt=task.prompt,
        spec=effective_spec,
    ) as trace:
        runner = AgentRunner(effective_spec, client, registry, trace)
        output = runner.run(task.prompt, task.id, task.category)

        try:
            passed, detail = task.validator(workspace, output.answer)
        except Exception as exc:  # noqa: BLE001 —— validator 自身异常也要变成失败结果
            passed, detail = False, f"validator 异常：{type(exc).__name__}: {exc}"

        events = read_trace(trace_path)
        failure_type = classify_failure(output, passed, events, detail)

        trace.write(
            "final",
            answer=output.answer,
            passed=passed,
            detail=detail,
            failure_type=failure_type,
            stopped_reason=output.stopped_reason,
        )
        trace.write(
            "episode_end",
            passed=passed,
            steps=output.steps,
            total_tokens=output.usage.total_tokens,
            latency_ms=round(output.latency_ms, 2),
            tool_calls=output.tool_calls,
            tool_errors=output.tool_errors,
            recovery_events=output.recovery_events,
            planner_tokens=output.planner_tokens,
            handoff_count=output.handoff_count,
            handoff_ok=output.handoff_ok,
        )

    return EpisodeResult(
        episode_id=episode_id,
        task_id=task.id,
        category=task.category,
        spec_name=spec.name,
        model=effective_spec.model,
        repeat=repeat,
        passed=passed,
        detail=detail,
        answer=output.answer,
        steps=output.steps,
        usage=output.usage,
        latency_ms=output.latency_ms,
        tool_calls=output.tool_calls,
        tool_errors=output.tool_errors,
        recovery_events=output.recovery_events,
        failure_type=failure_type,
        stopped_reason=output.stopped_reason,
        trace_path=str(trace_path.relative_to(run_root)),
        workspace=str(workspace.relative_to(run_root)),
        planner_tokens=output.planner_tokens,
        handoff_count=output.handoff_count,
        handoff_ok=output.handoff_ok,
    )


def _unique_run_dir(out_dir: Path) -> Path:
    """同一秒内跑多次时避免覆盖：追加 -2、-3。"""
    base = out_dir / new_run_id()
    candidate = base
    index = 2
    while candidate.exists():
        candidate = Path(f"{base}-{index}")
        index += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def run_experiment(
    specs: list[AgentSpec],
    tasks: list[Task],
    repeats: int,
    out_dir: str | Path,
    client: LLMClient,
    progress: bool = True,
    concurrency: int = 1,
) -> tuple[Path, list[EpisodeResult]]:
    """跑完整实验并落盘 results.json / report.md。

    concurrency > 1 时用线程池并行跑 episode。每个 episode 有独立的
    workspace/trace 目录，唯一共享的是 LLM client；OpenAI SDK 的 client
    是线程安全的，所以这里不需要额外加锁。
    """
    if repeats < 1:
        raise ValueError("repeats 必须 >= 1")
    if concurrency < 1:
        raise ValueError("concurrency 必须 >= 1")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    run_root = _unique_run_dir(out_path)

    started = time.time()
    results: list[EpisodeResult] = []
    jobs = [
        (spec, task, repeat)
        for spec in specs
        for task in tasks
        for repeat in range(repeats)
    ]
    total = len(jobs)
    done = 0

    def report(result: EpisodeResult, spec: AgentSpec, task: Task, repeat: int) -> None:
        nonlocal done
        results.append(result)
        done += 1
        if progress:
            status = "PASS" if result.passed else f"FAIL/{result.failure_type}"
            print(
                f"[{done}/{total}] {spec.name} · {task.id} · "
                f"repeat {repeat + 1}/{repeats} → {status} "
                f"(steps={result.steps}, tokens={result.usage.total_tokens})",
                flush=True,
            )

    if concurrency == 1:
        for spec, task, repeat in jobs:
            result = run_episode(task, spec, client, run_root, repeat)
            report(result, spec, task, repeat)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_map = {
                pool.submit(run_episode, task, spec, client, run_root, repeat): (spec, task, repeat)
                for spec, task, repeat in jobs
            }
            for future in as_completed(future_map):
                spec, task, repeat = future_map[future]
                result = future.result()
                report(result, spec, task, repeat)

    # 并行完成顺序不稳定；落盘前按 spec/task/repeat 排序，保证报告可复现。
    results.sort(key=lambda item: (item.spec_name, item.task_id, item.repeat))

    meta = {
        "run_id": run_root.name,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "duration_sec": round(time.time() - started, 2),
        "specs": [spec.to_dict() for spec in specs],
        "tasks": [task.id for task in tasks],
        "repeats": repeats,
        "concurrency": concurrency,
    }

    (run_root / "results.json").write_text(
        json.dumps(
            {"meta": meta, "results": [result.to_dict() for result in results]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 延迟导入，避免 report ↔ evaluate 循环依赖
    from .report import render_report

    render_report(meta, results, run_root / "report.md")
    return run_root, results


def run_cascade_episode(
    task: Task,
    weak_spec: AgentSpec,
    strong_spec: AgentSpec,
    client: LLMClient,
    run_root: Path,
    repeat: int,
) -> EpisodeResult:
    """先跑便宜配置；失败后再跑强配置。

    这是最小可用的 model routing：
      - 成功的大多数任务只付便宜模型的钱；
      - 只有验证失败才升级，避免所有任务都用最贵的模型。

    注意：强模型也会失败。cascade 不是「一定成功」，它只是把
    「弱模型失败」这个信号当作路由条件。
    """
    weak = run_episode(task, weak_spec, client, run_root, repeat)
    if weak.passed:
        return replace(weak, spec_name="cascade", escalated=False)

    strong = run_episode(task, strong_spec, client, run_root, repeat)
    combined_usage = Usage()
    combined_usage.add(weak.usage)
    combined_usage.add(strong.usage)
    detail = (
        f"弱配置 {weak_spec.name} 失败（{weak.failure_type}），"
        f"升级到 {strong_spec.name}：{strong.detail}"
    )
    # cascade 的一次 episode 可能发生两次 handoff（弱、强各一次）；
    # 报告按「episode 是否有交接」计数，只有全部交接成功才算成功。
    handoff_flags = [
        item.handoff_ok for item in (weak, strong) if item.handoff_count > 0
    ]
    return replace(
        strong,
        spec_name="cascade",
        model=f"{weak.model}->{strong.model}",
        detail=detail,
        steps=weak.steps + strong.steps,
        usage=combined_usage,
        latency_ms=weak.latency_ms + strong.latency_ms,
        tool_calls=weak.tool_calls + strong.tool_calls,
        tool_errors=weak.tool_errors + strong.tool_errors,
        recovery_events=weak.recovery_events + strong.recovery_events,
        planner_tokens=weak.planner_tokens + strong.planner_tokens,
        handoff_count=1 if handoff_flags else 0,
        handoff_ok=all(handoff_flags) if handoff_flags else False,
        escalated=True,
    )


def run_cascade_experiment(
    weak_spec: AgentSpec,
    strong_spec: AgentSpec,
    tasks: list[Task],
    repeats: int,
    out_dir: str | Path,
    client: LLMClient,
    progress: bool = True,
    concurrency: int = 1,
) -> tuple[Path, list[EpisodeResult]]:
    """cascade 实验：结果统一记在 spec_name='cascade' 下。"""
    if repeats < 1:
        raise ValueError("repeats 必须 >= 1")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    run_root = _unique_run_dir(out_path)

    started = time.time()
    jobs = [(task, repeat) for task in tasks for repeat in range(repeats)]
    results: list[EpisodeResult] = []
    total = len(jobs)
    done = 0

    def report(result: EpisodeResult, task: Task, repeat: int) -> None:
        nonlocal done
        results.append(result)
        done += 1
        if progress:
            status = "PASS" if result.passed else f"FAIL/{result.failure_type}"
            flag = "escalated" if result.escalated else "weak-only"
            print(
                f"[{done}/{total}] cascade · {task.id} · repeat {repeat + 1}/{repeats} "
                f"→ {status} ({flag}, steps={result.steps}, tokens={result.usage.total_tokens})",
                flush=True,
            )

    if concurrency == 1:
        for task, repeat in jobs:
            result = run_cascade_episode(
                task, weak_spec, strong_spec, client, run_root, repeat
            )
            report(result, task, repeat)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_map = {
                pool.submit(
                    run_cascade_episode,
                    task,
                    weak_spec,
                    strong_spec,
                    client,
                    run_root,
                    repeat,
                ): (task, repeat)
                for task, repeat in jobs
            }
            for future in as_completed(future_map):
                task, repeat = future_map[future]
                report(future.result(), task, repeat)

    results.sort(key=lambda item: (item.task_id, item.repeat))
    cascade_spec = AgentSpec(
        name="cascade",
        model=f"{weak_spec.model}->{strong_spec.model}",
        instruction="cascade: run weak first, escalate to strong only when the validator fails",
        tools=weak_spec.tools,
    )
    meta = {
        "run_id": run_root.name,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "duration_sec": round(time.time() - started, 2),
        "specs": [cascade_spec.to_dict()],
        "weak_spec": weak_spec.to_dict(),
        "strong_spec": strong_spec.to_dict(),
        "tasks": [task.id for task in tasks],
        "repeats": repeats,
        "concurrency": concurrency,
    }
    (run_root / "results.json").write_text(
        json.dumps(
            {"meta": meta, "results": [result.to_dict() for result in results]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    from .report import render_report

    render_report(meta, results, run_root / "report.md")
    return run_root, results


def aggregate(results: Iterable[EpisodeResult]) -> dict[str, dict[str, Any]]:
    """按 spec 聚合指标。

    关键点：
      - success_rate 是总体通过率；
      - success_std 是「每个 repeat 的通过率」的标准差，用来判断差异是否
        可能只是模型随机性，而不是 harness 改动真的有效；
      - tool_error_rate 是失败尝试次数 / 工具调用次数（自动重试可能让它 >100%）；
      - recovery_success 只看 recovery 类任务，避免被简单任务稀释。
    """
    grouped: dict[str, list[EpisodeResult]] = {}
    for result in results:
        grouped.setdefault(result.spec_name, []).append(result)

    summary: dict[str, dict[str, Any]] = {}
    for spec_name, items in grouped.items():
        total = len(items)
        passed = sum(1 for item in items if item.passed)
        repeats = sorted({item.repeat for item in items})
        per_repeat: list[float] = []
        for repeat in repeats:
            subset = [item for item in items if item.repeat == repeat]
            if subset:
                per_repeat.append(sum(1 for item in subset if item.passed) / len(subset))

        categories: dict[str, dict[str, Any]] = {}
        for item in items:
            bucket = categories.setdefault(item.category, {"total": 0, "passed": 0})
            bucket["total"] += 1
            bucket["passed"] += int(item.passed)

        failures: dict[str, int] = {}
        for item in items:
            if not item.passed:
                failures[item.failure_type] = failures.get(item.failure_type, 0) + 1

        recovery_items = [item for item in items if item.category == "recovery"]
        recovery_passed = sum(1 for item in recovery_items if item.passed)
        escalations = sum(1 for item in items if item.escalated)

        total_tokens = sum(item.usage.total_tokens for item in items)
        planner_tokens = sum(item.planner_tokens for item in items)
        handoff_items = [item for item in items if item.handoff_count > 0]
        handoff_ok_count = sum(1 for item in handoff_items if item.handoff_ok)

        latencies = [item.latency_ms for item in items]
        summary[spec_name] = {
            "total": total,
            "passed": passed,
            "success_rate": passed / total if total else 0.0,
            "success_std": statistics.pstdev(per_repeat) if len(per_repeat) > 1 else 0.0,
            "avg_steps": statistics.mean(item.steps for item in items) if items else 0.0,
            "avg_tokens": total_tokens / total if total else 0.0,
            "total_tokens": total_tokens,
            # 多智能体：planner token 是总 token 的子集，不是额外叠加。
            # handoff 按 episode 计数（单 Agent 配置为 0），成功率高表示
            # planner 确实产出了非空计划，而不是模型调用失败后的降级运行。
            "planner_tokens": planner_tokens,
            "avg_planner_tokens": planner_tokens / total if total else 0.0,
            "planner_token_rate": planner_tokens / total_tokens if total_tokens else 0.0,
            "handoff_count": len(handoff_items),
            "handoff_ok_count": handoff_ok_count,
            "handoff_success_rate": (
                handoff_ok_count / len(handoff_items) if handoff_items else 0.0
            ),
            "cache_hit_tokens": sum(item.usage.cache_hit_tokens for item in items),
            "cache_miss_tokens": sum(item.usage.cache_miss_tokens for item in items),
            # 命中率 = 命中 /（命中 + 未命中）。服务不返回缓存字段时两者都是 0，
            # 这里保持 0.0 而不是推断成「没有缓存」。
            "cache_hit_rate": (
                sum(item.usage.cache_hit_tokens for item in items)
                / max(
                    1,
                    sum(
                        item.usage.cache_hit_tokens + item.usage.cache_miss_tokens
                        for item in items
                    ),
                )
            ),
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0.0,
            "p50_latency_ms": _percentile(latencies, 0.50),
            "p95_latency_ms": _percentile(latencies, 0.95),
            "tool_calls": sum(item.tool_calls for item in items),
            "tool_errors": sum(item.tool_errors for item in items),
            "tool_error_rate": (
                sum(item.tool_errors for item in items) / max(1, sum(item.tool_calls for item in items))
            ),
            "recovery_total": len(recovery_items),
            "recovery_passed": recovery_passed,
            "recovery_success": recovery_passed / len(recovery_items) if recovery_items else 0.0,
            "escalations": escalations,
            "escalation_rate": escalations / total if total else 0.0,
            "categories": {
                name: {
                    "total": bucket["total"],
                    "passed": bucket["passed"],
                    "success_rate": bucket["passed"] / bucket["total"] if bucket["total"] else 0.0,
                }
                for name, bucket in categories.items()
            },
            "failures": failures,
        }
    return summary

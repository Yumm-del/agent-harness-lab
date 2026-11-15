# -*- coding: utf-8 -*-
"""命令行入口。

常用：
  python -m harness_lab list-tasks
  python -m harness_lab eval --repeat 1 --tasks fd_01_csv_filter
  python -m harness_lab eval --specs configs/baseline.yaml configs/repair.yaml --repeat 3
  python -m harness_lab show-trace runs/<run_id>/baseline/fd_01_csv_filter/repeat_0/trace.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .config import PROJECT_ROOT, ensure_project_dirs, get_api_key, get_base_url
from .evaluate import run_cascade_experiment, run_experiment
from .evolve import evolve_from_results
from .llm import OpenAICompatibleClient
from .replay import ReplayMismatch, replay_episode
from .spec import load_specs
from .tasks import TASKS, select_tasks, task_summary
from .trace import read_trace, trace_stats


DEFAULT_SPECS = [
    PROJECT_ROOT / "configs" / "baseline.yaml",
    PROJECT_ROOT / "configs" / "repair.yaml",
    PROJECT_ROOT / "configs" / "prune.yaml",
]


def _parse_task_ids(raw: str | None) -> list[str] | None:
    if not raw or raw.strip().lower() == "all":
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def _cmd_list_tasks(args: argparse.Namespace) -> int:
    print(f"共 {len(TASKS)} 个任务")
    print(f"分类：{json.dumps(task_summary(), ensure_ascii=False)}")
    print("")
    print(f"{'id':28} {'category':14} {'difficulty':10} max_steps")
    print("-" * 72)
    for task in TASKS:
        print(f"{task.id:28} {task.category:14} {task.difficulty:10} {task.max_steps}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    spec_paths = [Path(p) for p in (args.specs or DEFAULT_SPECS)]
    missing = [str(path) for path in spec_paths if not path.exists()]
    if missing:
        print(f"缺少 spec 文件：{missing}", file=sys.stderr)
        return 2

    specs = load_specs(spec_paths)
    if args.model:
        specs = [replace(spec, model=args.model) for spec in specs]

    try:
        tasks = select_tasks(_parse_task_ids(args.tasks))
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.limit:
        tasks = tasks[: args.limit]

    ensure_project_dirs()
    client = OpenAICompatibleClient(
        api_key=get_api_key(),
        base_url=get_base_url(),
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    print(
        f"开始实验：{len(specs)} 个配置 × {len(tasks)} 个任务 × {args.repeat} 次重复",
        flush=True,
    )
    print(f"模型：{', '.join(sorted({spec.model for spec in specs}))}", flush=True)

    run_root, results = run_experiment(
        specs=specs,
        tasks=tasks,
        repeats=args.repeat,
        out_dir=args.out,
        client=client,
        progress=not args.quiet,
        concurrency=args.concurrency,
    )
    passed = sum(1 for item in results if item.passed)
    print("")
    print(f"完成：{passed}/{len(results)} 通过")
    print(f"运行目录：{run_root}")
    print(f"报告：{run_root / 'report.md'}")
    return 0


def _cmd_cascade(args: argparse.Namespace) -> int:
    try:
        weak_spec = load_specs([args.weak])[0]
        strong_spec = load_specs([args.strong])[0]
        tasks = select_tasks(_parse_task_ids(args.tasks))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.limit:
        tasks = tasks[: args.limit]

    ensure_project_dirs()
    client = OpenAICompatibleClient(
        api_key=get_api_key(),
        base_url=get_base_url(),
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    print(
        f"cascade：{weak_spec.name}({weak_spec.model}) → "
        f"{strong_spec.name}({strong_spec.model})，{len(tasks)} 个任务 × {args.repeat} 次",
        flush=True,
    )
    run_root, results = run_cascade_experiment(
        weak_spec=weak_spec,
        strong_spec=strong_spec,
        tasks=tasks,
        repeats=args.repeat,
        out_dir=args.out,
        client=client,
        progress=not args.quiet,
        concurrency=args.concurrency,
    )
    passed = sum(1 for item in results if item.passed)
    escalated = sum(1 for item in results if item.escalated)
    print("")
    print(f"完成：{passed}/{len(results)} 通过，升级 {escalated} 次")
    print(f"运行目录：{run_root}")
    print(f"报告：{run_root / 'report.md'}")
    return 0


def _cmd_evolve(args: argparse.Namespace) -> int:
    try:
        evolved, reason, path = evolve_from_results(
            args.results,
            args.spec,
            args.out,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"已生成：{path}")
    print(f"修改理由：{reason}")
    print("")
    print("下一步：")
    print(f"  harness-lab eval --specs {path} --repeat 2 --concurrency 4")
    return 0


def _cmd_show_trace(args: argparse.Namespace) -> int:
    path = Path(args.trace)
    if not path.exists():
        print(f"trace 不存在：{path}", file=sys.stderr)
        return 2
    events = read_trace(path)
    stats = trace_stats(events)
    print(f"文件：{path}")
    print(f"事件数：{len(events)}")
    print(f"统计：{json.dumps(stats, ensure_ascii=False)}")
    print("")
    for event in events:
        kind = event.get("event")
        step = event.get("step", 0)
        if kind == "llm_call":
            response = event.get("response", {})
            calls = response.get("tool_calls", [])
            names = ", ".join(call.get("name", "?") for call in calls) or "(final)"
            agent = "planner " if event.get("agent") == "planner" else ""
            print(
                f"[{step}] {agent}llm_call → {names}  "
                f"tokens={response.get('usage', {}).get('total_tokens', 0)}"
            )
        elif kind == "tool_call":
            status = "ok" if event.get("ok") else "FAIL"
            print(f"[{step}] tool_call {event.get('name')} {status}  attempts={event.get('attempts')}")
        elif kind == "handoff":
            plan = str(event.get("plan", "")).replace("\n", " ")
            print(
                f"[{step}] handoff planner→executor ok={event.get('ok')}  "
                f"plan={plan[:80]}"
            )
        elif kind in {"repair_hint", "max_steps", "llm_error"}:
            print(f"[{step}] {kind}  {event.get('error', '')}")
        elif kind == "final":
            print(
                f"[{step}] final  passed={event.get('passed')}  "
                f"failure={event.get('failure_type')}  {event.get('detail', '')}"
            )
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    try:
        result = replay_episode(args.trace, args.out)
    except ReplayMismatch as exc:
        print(f"回放失败：{exc}", file=sys.stderr)
        return 3
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["same_outcome"] else 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness-lab",
        description="Trace-first Agent Harness：评测、归因、回放。",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list-tasks", help="列出任务集")
    list_parser.set_defaults(func=_cmd_list_tasks)

    eval_parser = sub.add_parser("eval", help="跑对照实验")
    eval_parser.add_argument(
        "--specs",
        nargs="+",
        help="一个或多个 YAML 配置；缺省用 configs/baseline+repair+prune",
    )
    eval_parser.add_argument(
        "--tasks",
        default="all",
        help="逗号分隔的任务 id；all 表示全部",
    )
    eval_parser.add_argument("--repeat", type=int, default=1, help="每个任务重复次数")
    eval_parser.add_argument("--out", default=str(PROJECT_ROOT / "results"), help="结果根目录")
    eval_parser.add_argument("--model", help="覆盖所有 spec 的模型（方便快速试跑）")
    eval_parser.add_argument("--limit", type=int, help="只跑前 N 个任务")
    eval_parser.add_argument("--timeout", type=float, default=90.0, help="单次 LLM 请求超时秒数")
    eval_parser.add_argument("--max-retries", type=int, default=2, help="API 层自动重试次数")
    eval_parser.add_argument("--concurrency", type=int, default=1, help="并行 episode 数")
    eval_parser.add_argument("--quiet", action="store_true", help="不打印每条 episode")
    eval_parser.set_defaults(func=_cmd_eval)

    cascade_parser = sub.add_parser("cascade", help="弱配置失败后升级到强配置（model routing）")
    cascade_parser.add_argument("--weak", default=str(PROJECT_ROOT / "configs" / "baseline.yaml"))
    cascade_parser.add_argument("--strong", default=str(PROJECT_ROOT / "configs" / "reasoner.yaml"))
    cascade_parser.add_argument("--tasks", default="all")
    cascade_parser.add_argument("--repeat", type=int, default=1)
    cascade_parser.add_argument("--out", default=str(PROJECT_ROOT / "results" / "cascade"))
    cascade_parser.add_argument("--limit", type=int)
    cascade_parser.add_argument("--timeout", type=float, default=90.0)
    cascade_parser.add_argument("--max-retries", type=int, default=2)
    cascade_parser.add_argument("--concurrency", type=int, default=1)
    cascade_parser.add_argument("--quiet", action="store_true")
    cascade_parser.set_defaults(func=_cmd_cascade)

    evolve_parser = sub.add_parser("evolve", help="从失败报告生成一个最小 harness 修改（AEvo-lite v0）")
    evolve_parser.add_argument("--from", dest="results", required=True, help="results.json 路径")
    evolve_parser.add_argument("--spec", help="分析哪个配置；缺省用 meta 里的第一个")
    evolve_parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "configs" / "evolved.yaml"),
        help="输出的 YAML 路径",
    )
    evolve_parser.set_defaults(func=_cmd_evolve)

    trace_parser = sub.add_parser("show-trace", help="按人类可读格式打印 trace")
    trace_parser.add_argument("trace", help="trace.jsonl 路径")
    trace_parser.set_defaults(func=_cmd_show_trace)

    replay_parser = sub.add_parser("replay", help="按 trace 回放，重新执行工具")
    replay_parser.add_argument("trace", help="源 trace.jsonl 路径")
    replay_parser.add_argument("--out", default=str(PROJECT_ROOT / "runs" / "replay"), help="回放输出目录")
    replay_parser.set_defaults(func=_cmd_replay)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

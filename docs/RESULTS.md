# 实验结果（2026-11-15）

本文档保存正式实验的完整数据和解读。README 里只放摘要，这里放逐任务矩阵、关键 trace 和复现命令。

## 1. 实验设置

| 项 | 值 |
|---|---|
| 时间 | 2026-11-15 17:55:19，总耗时 186.08 s |
| 规模 | 4 配置 × 16 任务 × 3 次重复 = 192 个 episode |
| 并发 | 4 |
| 模型 | `deepseek-chat`（baseline / repair / prune）、`deepseek-reasoner`（reasoner） |
| 采样 | `temperature=0`，`max_tokens=900`（reasoner 为 1200） |
| 运行目录 | `results/main/20261115-175519/` |

被比较的四个配置**只在 harness 控制变量上不同**：

- `baseline`：完整上下文、`tool_retries=0`、`repair_on_error=false`；
- `repair`：`tool_retries=2`、`repair_on_error=true`，其余完全相同；
- `prune`：`context_policy=prune`、`context_window=3`，其余同 baseline；
- `reasoner`：harness 同 baseline，只换 `deepseek-reasoner`。

## 2. 总表

| 配置 | 成功率 | repeat 标准差 | 平均步数 | 平均 token/任务 | 缓存命中率 | P50 延迟 | P95 延迟 | 工具错误率 | 恢复成功率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 93.8% (45/48) | 0.0% | 3.79 | 4040 | 81.5% | 3690 ms | 5109 ms | 5.4% | 75.0% (9/12) |
| repair | **100.0% (48/48)** | 0.0% | 3.83 | 4119 | 81.7% | 3701 ms | 4481 ms | 9.0% | **100.0% (12/12)** |
| prune | 93.8% (45/48) | 0.0% | 3.77 | 4002 | 79.5% | 3632 ms | 4628 ms | 5.5% | 75.0% (9/12) |
| reasoner | 93.8% (45/48) | 0.0% | 4.25 | 4962 | 79.1% | 4599 ms | 5998 ms | 6.3% | 75.0% (9/12) |

## 3. 逐任务矩阵（通过次数 / 3 次重复）

| 任务 | baseline | repair | prune | reasoner |
|---|---:|---:|---:|---:|
| `fd_01_csv_filter` | 3/3 | 3/3 | 3/3 | 3/3 |
| `fd_02_json_extract` | 3/3 | 3/3 | 3/3 | 3/3 |
| `fd_03_log_count` | 3/3 | 3/3 | 3/3 | 3/3 |
| `fd_04_merge_csv` | 3/3 | 3/3 | 3/3 | 3/3 |
| `ms_01_api_sum` | 3/3 | 3/3 | 3/3 | 3/3 |
| `ms_02_api_join` | 3/3 | 3/3 | 3/3 | 3/3 |
| `ms_03_config_then_api` | 3/3 | 3/3 | 3/3 | 3/3 |
| `ms_04_monthly_totals` | 3/3 | 3/3 | 3/3 | 3/3 |
| `rc_01_timeout_retry` | **0/3** | **3/3** | **0/3** | **0/3** |
| `rc_02_dirty_records` | 3/3 | 3/3 | 3/3 | 3/3 |
| `rc_03_missing_file_fallback` | 3/3 | 3/3 | 3/3 | 3/3 |
| `rc_04_case_sensitive_api` | 3/3 | 3/3 | 3/3 | 3/3 |
| `lh_01_filter_then_count` | 3/3 | 3/3 | 3/3 | 3/3 |
| `lh_02_missing_ids` | 3/3 | 3/3 | 3/3 | 3/3 |
| `lh_03_rule_branch` | 3/3 | 3/3 | 3/3 | 3/3 |
| `lh_04_error_report` | 3/3 | 3/3 | 3/3 | 3/3 |

**这张表最重要的信息不是 48/48，而是 15 个任务在所有配置下都饱和。**
任务集目前只有 `rc_01` 一个真正区分 harness 的对抗样本。这不是实验失败，是 benchmark 区分度不足：
下一步要补 held-out 任务和更难的上下文/恢复场景（见 README Roadmap），而不是把 100% 当成能力结论。

## 4. 关键样本：`rc_01_timeout_retry`

任务设计：`http_get` 前两次调用注入超时，`max_steps=3`，要求把结果写入 `value.txt`。

### baseline（deepseek-chat，无重试）

```
[1] llm_call → http_get   tokens=841
[1] tool_call http_get FAIL  attempts=1
[2] llm_call → http_get   tokens=922
[2] tool_call http_get FAIL  attempts=1
[3] llm_call → http_get   tokens=999
[3] tool_call http_get ok    attempts=1
[3] max_steps
[3] final passed=False failure=step_limit 缺少输出文件：value.txt
```

### reasoner（更强的模型，同样的 harness）

结果相同，也是 0/3。多花约 23% token、P50 延迟高约 25%，没有换来通过。

### repair（同一模型，harness 层重试）

```
[1] llm_call → http_get   tokens=841
[1] tool_call http_get ok    attempts=3   ← 重试在工具层消化
[2] llm_call → write_file tokens=912
[2] tool_call write_file ok  attempts=1
[3] llm_call → (final)    tokens=966
[3] final passed=True  value.txt 数值正确（42.0）
```

同一个模型，只改了 `tool_retries` 和 `repair_on_error`，从 0/3 变成 3/3。

**结论：瓶颈不在模型推理，而在「一次工具失败消耗一整个 agent step」的 harness 设计。**
baseline 的第三次调用其实已经成功，但步数预算用完了，没有机会写文件。这是控制层问题，不是模型问题。

## 5. AEvo-lite 闭环验证

用 baseline 的失败报告生成修改，再回测：

```bash
python -m harness_lab evolve --from results/main/20261115-175519/results.json \
  --spec baseline --out configs/evolved.yaml
```

生成的 `configs/evolved.yaml`：`tool_retries=2`、`repair_on_error=true`、instruction 不变——
与人工写的 `repair` 等价，且这个结论是从失败类型 `step_limit` 推出来的，不是预设的。

回测（16 任务 × 3 次，`results/evolved/20261115-175910/`）：

| 配置 | 成功率 | 平均 token/任务 |
|---|---:|---:|
| baseline（修改前） | 93.8% | 4040 |
| evolved（修改后） | **100.0% (48/48)** | 4090 |

evolved 与 repair 是同一份配置（只差 name/description），两次独立运行的平均 token 是 4090 与 4119，
差 0.7%。这是正常的采样波动——跨 run 比较 token 时要谨慎，报告里给的是原始值，不做平滑。

说明「失败报告 → 最小修改 → 回归验证」这条链路是通的。
注意 v0 是规则式的，不是 LLM meta-agent；它能证明闭环，不能证明会自动发现更细的 harness 修改。

## 6. 模型级联：能省钱，但救不了设计错误

`cascade` = 先跑 baseline，validator 失败才升级到 reasoner（16 任务 × 3 次，`results/cascade/20261115-175959/`）。

| 方案 | 成功率 | 升级次数 | 平均 token/任务 |
|---|---:|---:|---:|
| 始终 baseline | 93.8% | 0 | 4040 |
| **cascade** | **93.8%** | 3 | **4216** |
| 始终 reasoner | 93.8% | 0 | 4962 |

- 相比「始终用强模型」，cascade 省了 **15.0% token**，成功率相同；
- 但 3 次升级全部发生在 `rc_01`，且**全部失败**（5642 / 5650 / 5767 token 纯属浪费）；
- 路由只解决「任务难度不匹配」，不解决「harness 缺重试」。两者是不同层面的问题。

## 7. 读表注意

- 16 个合成任务，规模小；成功率不能外推成真实业务能力。
- `repeat 标准差` 在 3 次重复下基本是 0（temperature=0 本身就是确定性的），它只能提示噪声，不是置信区间。
- `工具错误率` = 失败尝试次数 / 工具调用次数。开启重试后这个比率会上升，因为失败尝试也被计入——这是指标定义，不代表工具质量变差。
- `缓存命中率` = `prompt_cache_hit_tokens` /（命中 + 未命中），由 DeepSeek 返回。四个配置都在 79%–82%，因为 agent loop 每步重发同一段不断增长的前缀。项目不把命中的 token 折算成金额，避免使用会过时的价格。
- `prune` 与 baseline 结果完全相同：裁剪只在 26/181 次 LLM 调用中真正触发，且被裁掉的都是可重读的中间结果。要验证裁剪价值，需要构造 observation 很长（大文件、长日志）的任务。

## 8. 复现

```bash
cd D:\file_of_claude\agent-harness-lab
python -m pip install -e ".[dev]"

# 主实验（192 episodes，约 3 分钟）
python -m harness_lab eval \
  --specs configs/baseline.yaml configs/repair.yaml configs/prune.yaml configs/reasoner.yaml \
  --repeat 3 --concurrency 4 --out results/main

# AEvo-lite 闭环
python -m harness_lab evolve --from results/main/<run_id>/results.json --spec baseline --out configs/evolved.yaml
python -m harness_lab eval --specs configs/evolved.yaml --repeat 3 --concurrency 4 --out results/evolved

# 模型级联
python -m harness_lab cascade --weak configs/baseline.yaml --strong configs/reasoner.yaml \
  --repeat 3 --concurrency 4 --out results/cascade

# 回放任意一条 trace（不调模型，重跑工具）
python -m harness_lab replay results/main/<run_id>/baseline/rc_01_timeout_retry/repeat_0/trace.jsonl
```

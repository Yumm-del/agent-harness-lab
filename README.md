# agent-harness-lab

[![tests](https://github.com/Yumm-del/agent-harness-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Yumm-del/agent-harness-lab/actions/workflows/ci.yml)

> 一个 trace-first 的 Agent Harness 实验场：把 Agent 的运行过程变成可评测、可归因、可回放、可迭代的数据。

这不是又一个 Agent 产品 demo。它解决的是 harness 工程里的一个核心问题：

**同一个模型，换一套 harness（提示词、工具接口、上下文策略、重试策略、模型路由），任务成功率会明显不同。但大多数 Agent 项目只展示“能跑”，没有留下可比较的证据。**

这个仓库把每次运行拆成 5 层：

```
AgentSpec（Instruction / Context / Tools / Model）
        ↓
AgentRunner（loop / tool calls / retry / context policy）
        ↓
Trace（每步 messages、tool call、错误、token、延迟）
        ↓
Validator（确定性检查，不靠模型自评）
        ↓
Report（成功率 / 步数 / token / P50/P95 / 失败分类 / 升级率）
```

## 这个项目能回答什么

- 加工具自动重试，成功率提升多少？代价是多花多少步、多少 token？
- 换更强的模型，能不能替代 harness 层的修复？
- 压缩上下文是省钱还是丢关键证据？
- 一个失败 episode 到底是“没规划”“工具报错没恢复”还是“算错了”？
- 录下来的 badcase，改完 harness 后能不能离线重放验证？

## 快速开始

```bash
cd D:\file_of_claude\agent-harness-lab

# 1. 安装（可编辑模式，改代码不用重装）
python -m pip install -e ".[dev]"

# 2. 配置 API Key（.env 已在 .gitignore 里）
copy .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY=sk-...

# 3. 看任务集
harness-lab list-tasks

# 4. 跑对照实验：baseline vs repair
harness-lab eval --specs configs/baseline.yaml configs/repair.yaml --repeat 2 --concurrency 4

# 5. 看某一条 trace
harness-lab show-trace results\<run_id>\baseline\fd_01_csv_filter\repeat_0\trace.jsonl

# 6. 回放（重新执行工具，不重新调模型）
harness-lab replay results\<run_id>\baseline\fd_01_csv_filter\repeat_0\trace.jsonl

# 7. 模型级联：便宜模型先跑，失败再升级
harness-lab cascade --weak configs/baseline.yaml --strong configs/reasoner.yaml --repeat 2
```

> Windows 下如果 `harness-lab` 不在 PATH，用 `python -m harness_lab ...` 代替。

## 任务集：16 个可复现任务

| 类别 | 任务 | 难度 | 测什么 |
|---|---|---|---|
| file_data | `fd_01_csv_filter` | easy | 读 CSV、过滤、聚合、写 JSON |
| file_data | `fd_02_json_extract` | easy | 嵌套 JSON 取值、精确格式 |
| file_data | `fd_03_log_count` | easy | 文本扫描、计数 |
| file_data | `fd_04_merge_csv` | medium | 两表 join、排序、类型保持 |
| multi_step | `ms_01_api_sum` | easy | 调 mock API、条件求和 |
| multi_step | `ms_02_api_join` | medium | 两个 API join、按用户聚合 |
| multi_step | `ms_03_config_then_api` | medium | 先读配置，再决定请求哪个端点 |
| multi_step | `ms_04_monthly_totals` | medium | 多文件汇总 |
| recovery | `rc_01_timeout_retry` | hard | 工具连续超时，测试 harness 自动重试 |
| recovery | `rc_02_dirty_records` | medium | 脏数据：跳过非数字 amount |
| recovery | `rc_03_missing_file_fallback` | medium | 主文件缺失，回退到备份 |
| recovery | `rc_04_case_sensitive_api` | hard | 404 后修正大小写参数重试 |
| long_horizon | `lh_01_filter_then_count` | medium | 过滤 → 写文件 → 再读回 → 计数 |
| long_horizon | `lh_02_missing_ids` | medium | 两个集合求差、排序输出 |
| long_horizon | `lh_03_rule_branch` | hard | 读规则 → 分支 → 调 API → 计算 |
| long_horizon | `lh_04_error_report` | hard | 解析日志、按 code 计数、映射描述 |

每个任务都有：
- 固定的 fixtures（CSV/JSON/log 文本）；
- mock API 路由；
- 可选的故障注入规则；
- 确定性 validator（文件内容 / JSON / 数字），不依赖 LLM 裁判。

## 四种配置 + 一种路由，只改一个变量

| 配置 | 改了什么 | 目的 |
|---|---|---|
| `baseline` | 完整上下文、不自动重试、不注入修复提示 | 对照组 |
| `repair` | `tool_retries=2` + 失败后注入修复提示 | 测 harness 层恢复能力 |
| `prune` | 只保留最近 3 条 tool observation | 测上下文压缩的收益/代价 |
| `reasoner` | 同一 harness，换 `deepseek-reasoner` | 把“模型能力”和“harness 能力”拆开 |
| `cascade` | 先跑 baseline，validator 失败才升级 reasoner | 测模型路由的成本-成功率 |

`AgentSpec` 借用了 AOrchestra 的四元组思路：

```yaml
name: repair
model: deepseek-chat
instruction: |
  You are a careful autonomous agent...
tools: [list_files, read_file, write_file, http_get, run_python]
max_steps: 8
context_policy: full
tool_retries: 2
repair_on_error: true
```

换配置不需要改代码；trace 里保存完整 spec，回放时可以重建同一次运行。

## Trace：失败归因的证据链

每个 episode 一个 `trace.jsonl`：

```jsonc
{"event":"episode_start","task_id":"rc_01_timeout_retry","spec":{...},"prompt":"..."}
{"event":"llm_call","step":1,"messages":[...],"request_hash":"a1b2c3d4","response":{...},"latency_ms":1200}
{"event":"tool_call","step":1,"name":"http_get","args":{...},"ok":false,"attempts":1,"failed_attempts":1,"attempt_errors":["工具 http_get 调用超时"]}
{"event":"repair_hint","step":1,"content":"One or more tool calls failed..."}
{"event":"final","step":3,"passed":false,"failure_type":"step_limit","detail":"缺少输出文件：value.txt"}
```

为什么存完整 messages：
- 只看最后一句，无法区分「没读文件」「读错文件」「读到了但算错」；
- 有完整 messages 才能做 replay，才能验证改完 harness 后行为是否真的变了；
- `request_hash` 让回放能检测提示词/工具/代码漂移。

## 失败分类

| 类型 | 判定依据 |
|---|---|
| `step_limit` | 用完 `max_steps` 仍在调工具 |
| `tool_failure` | 工具失败，且没有恢复 |
| `recovery_loop` | 同一工具 + 同一参数重复失败 ≥ 2 次 |
| `planning_no_tool` | 一次工具都没调用，直接给答案 |
| `missing_artifact` | 任务要求的输出文件不存在 |
| `format_error` | 输出不是合法 JSON |
| `execution_wrong_result` | 文件存在、格式对，但内容错 |
| `llm_error` | API/网络错误 |
| `harness_error` | 一次 LLM 调用都没有，通常是 harness 自身问题 |

分类是规则式的，不完美，但每个判断都能在 trace 里找到证据。面试时应该能逐条解释，而不是背一个分数。

## AEvo-lite v0：从失败报告生成修改

```bash
# 读一次 run 的 results.json，按失败类型生成一个最小修改
harness-lab evolve --from results\<run_id>\results.json --spec baseline --out configs/evolved.yaml

# 再跑一次，验证修改是否真的有效
harness-lab eval --specs configs/evolved.yaml --repeat 2 --concurrency 4
```

**本次闭环验证**：用 baseline 的失败报告（唯一失败类型 `step_limit`）生成 `configs/evolved.yaml`，
结果为 `tool_retries=2` + `repair_on_error=true`，与人工写的 `repair` 等价；
回测 16 任务 × 3 次 **48/48 通过**（修改前 45/48），平均 token 4040 → 4090
（与 `repair` 的 4119 有 0.7% 差异，因为这是另一次独立采样——同一配置跨 run 的 token 不是完全确定的）。

v0 是规则式的，不是 LLM meta-agent：

- `step_limit` / `tool_failure` / `recovery_loop` → 开自动重试 + 修复提示；
- `missing_artifact` → 在 instruction 里要求完成前检查输出文件；
- `format_error` → 要求严格 JSON；
- `planning_no_tool` → 要求先调用工具。

它先验证「失败 → 修改 → 回归」这条链路可复现；下一步才是让 LLM 读 trace、提出更细的 harness 修改。这个区别在面试里要说清楚，不能把 v0 说成完整 AEvo。

## 回放：不是“再看一遍日志”

`harness-lab replay` 会：
1. 从 trace 里按顺序取出 LLM 响应，不再调模型；
2. 重新执行工具（文件、mock API、故障注入都会重新发生）；
3. 每一步比较 `request_hash`，提示词/工具/代码变了会直接报 `ReplayMismatch`；
4. 用同一个 validator 重新判定，输出 `same_outcome`。

这对应「线上 badcase 录制 → 离线复现 → 改 harness → 回归验证」的闭环。

## 实验结果

<!-- RESULTS_START -->
**实验规模**：4 配置 × 16 任务 × 3 次重复 = 192 个 episode，并发 4，总耗时 186 秒。
模型均为 DeepSeek（`deepseek-chat` / `deepseek-reasoner`），`temperature=0`。

| 配置 | 成功率 | repeat 标准差 | 平均步数 | 平均 token/任务 | 缓存命中率 | P50 延迟 | P95 延迟 | 工具错误率 | 恢复成功率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 93.8% (45/48) | 0.0% | 3.79 | 4040 | 81.5% | 3690 ms | 5109 ms | 5.4% | 75.0% (9/12) |
| repair | **100.0% (48/48)** | 0.0% | 3.83 | 4119 | 81.7% | 3701 ms | 4481 ms | 9.0% | **100.0% (12/12)** |
| prune | 93.8% (45/48) | 0.0% | 3.77 | 4002 | 79.5% | 3632 ms | 4628 ms | 5.5% | 75.0% (9/12) |
| reasoner | 93.8% (45/48) | 0.0% | 4.25 | **4962** | 79.1% | 4599 ms | 5998 ms | 6.3% | 75.0% (9/12) |

### 结论一：harness 层重试解决模型换不掉的问题

三个配置的 9 次失败全部是同一个任务：`rc_01_timeout_retry`（`step_limit`）。
这是一个故意设计的对抗任务：`http_get` 前两次调用注入超时，`max_steps=3`。

| | baseline（chat 模型） | reasoner（更强模型） | repair（chat + harness 重试） |
|---|---|---|---|
| 第 1 步 | `http_get` 失败（attempts=1） | `http_get` 失败 | `http_get` 成功（attempts=3，harness 内部重试） |
| 第 2 步 | 再试一次，又失败 | 再试一次，又失败 | **写出 `value.txt`** |
| 第 3 步 | 第三次成功，但步数耗尽，没机会写文件 | 同左 | 直接给最终答案 |
| 结果 | 0/3 通过 | 0/3 通过 | **3/3 通过** |

**换更贵的模型没有用，因为瓶颈不在推理能力，而在"一次工具失败就消耗一整个 agent step"这个 harness 设计。**
`repair` 把重试下沉到工具层，同样的失败在一个 step 内消化掉。

需要诚实说明：这 6.2 个百分点的差距**只来自这一个任务 × 3 次重复**，其余 15 个任务四个配置全部通过。
所以正确的说法是「在工具连续失败的场景下，harness 重试把 0/3 变成 3/3」，而不是「repair 整体更强」。

### 结论二：更强模型在这套任务上是纯成本

`reasoner` 成功率与 baseline 相同，但平均多花 **23% token**（4962 vs 4040）、多 **0.46 步**、P50 延迟高约 **25%**。
在这 16 个短程任务上，推理模型的额外思考没有换来任何通过率收益——这是任务集的性质，不是对模型的普遍结论。

### 结论三：上下文裁剪是 null result，但 null 本身有信息

`prune` 与 baseline 结果完全一致。查 trace 发现裁剪只在 **26/181 次 LLM 调用**里真的触发过
（单次最多 5 条 tool observation，窗口是 3），且被裁掉的都是可以从文件重读的中间结果。
**不是 prune 无效，而是这套任务的上下文太短，裁剪没有可作用的压力。**
要验证裁剪的价值，需要构造 observation 很长（大文件、长日志）的任务——这条留在 Roadmap。

### 结论四：重试没有带来成本爆炸，前缀缓存命中率稳定在 80% 左右

`repair` 平均 token（4119）比 baseline（4040）只高 **2.0%**：
baseline 用两个完整 LLM step 处理两次失败，repair 在一个 tool call 内重试完成，把"重试成本"从 step 级降到了 tool 级。
工具错误率 9.0% 高于 baseline 的 5.4%，因为重试会把失败尝试计入分母——这是指标定义问题，报告里已注明。

四个配置的 DeepSeek 前缀缓存命中率都在 **79.1%–81.7%**：agent loop 每一步都重发同一段不断增长的前缀，天然适合 context cache。
项目只记录 `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`，**不按厂商价格折算成钱**——避免把会过时的价格写成事实，
也避免让"缓存命中率高"被误读成"成本一定低"（缓存折扣、输出价格、重试次数要一起算）。

### 结论五：模型路由能省钱，但救不了 harness 设计错误

`cascade`（先 baseline，落败才升级 reasoner，16 任务 × 3 次）：

| 方案 | 成功率 | 升级次数 | 平均 token/任务 |
|---|---:|---:|---:|
| 始终 baseline | 93.8% | 0 | 4040 |
| cascade | 93.8% | 3 | 4216 |
| 始终 reasoner | 93.8% | 0 | 4962 |

cascade 比「始终用强模型」省 15.0% token，成功率相同。
但 3 次升级全部发生在 `rc_01`，且全部失败（累计 ~17k token 纯属浪费）。
**路由解决的是「任务难度不匹配」，不解决「harness 缺重试」——两者是不同层面的问题。**

### 结论六：任务集在 15/16 上饱和，这是 benchmark 的局限

除 `rc_01` 外，其余 15 个任务在四个配置下全部 3/3。所以：
- 不能把 "repair 100%" 说成 repair 整体更强，只能说它在工具连续失败场景里有效；
- 任务集区分度不足是真实的短板，下一步应补 held-out 任务和更难的上下文压力场景（见 Roadmap）；
- 但在饱和区里，token、步数、延迟、恢复成功率仍然区分出了配置差异——不是所有维度都饱和。

完整的逐任务矩阵、trace 摘录和复现命令见 [docs/RESULTS.md](docs/RESULTS.md)。
<!-- RESULTS_END -->

## 设计取舍

> 更完整的设计理由、任务构造思路和面试追问准备见 [docs/DESIGN.md](docs/DESIGN.md)。

1. **确定性 validator 优先，LLM judge 只做定性补充。**
   LLM judge 有噪声，不适合当 pass/fail 的唯一依据。这个项目里 16 个任务全部用文件内容/JSON/数字校验。

2. **任务级 `max_steps` 可以低于 spec 的预算。**
   `rc_01` 故意把步数压到 3：baseline 没有自动重试，两次超时后没机会写文件；repair 在 harness 层重试，第 1 步就能拿到结果。这样差异来自 harness，而不是 prompt 运气。

3. **故障注入是评测装置，不是工具本身。**
   `FaultInjector` 只在指定工具的前 N 次调用上注入超时/错误；同一个工具在 baseline 和 repair 下表现不同，才能做对照。

4. **沙箱是本地实验沙箱，不是生产安全沙箱。**
   `run_python` 用子进程 + 超时 + `-I` 隔离，路径限制在工作区；但没有系统调用级隔离，也不能阻止 Python 访问网络。README 明确写这个边界，不假装它是生产级。

5. **不伪造成本数字。**
   报告里给 token、缓存命中率、步数、P50/P95 延迟；成本需要按当时价格换算，项目没有硬编码厂商价格，避免把过时价格当成事实。

## 已知局限

- 16 个合成任务，规模小；成功率不能外推成真实业务能力。
- 3 次重复只能提示噪声，不是置信区间；要提高统计严谨性需要更多 repeat 和 bootstrap。
- 任务以单 Agent 为主；多智能体协作是 Roadmap，不是当前卖点。
- Prompt Cache 只记录了命中/未命中 token，没有做 **cache-aware 调度**（例如把共享长前缀的任务排在一起），也没有按真实价格做成本-成功率 Pareto。
- 没有接真实 OpenManus runtime；当前是自带的轻量 loop，后续会加 adapter。

## Roadmap

- **Multi-agent**：planner/executor 或 solver/verifier，trace 增加 `agent` 字段与 handoff 指标。
- **AEvo-lite**：meta-agent 读失败报告，只改一个 prompt/tool 配置，再跑回归集；提升超过噪声才保留。
- **OpenManus adapter**：同一批任务跑在 OpenManus runtime 上，对比自带 loop 与开源框架的失败模式。
- **Prompt Cache / 成本**：已记录 cache hit token；下一步做 cache-aware 调度与按真实价格的成本-成功率 Pareto。
- **Held-out 任务**：dev/test 拆分，避免在测试集上调 harness。

## 与 DeepWisdom 的关系

这个项目是独立实现的，但设计上对齐了两条公开思路：

- **AOrchestra**：Agent 可以拆成 `(Instruction, Context, Tools, Model)` 四元组；本项目把它落成 `AgentSpec`，让 harness 对比变成换 YAML。
- **AEvo**：把失败证据当作流程状态，让 meta-agent 修改 harness 后再评测；本项目的 trace / failure taxonomy / replay 是这条闭环的底座。

如果继续做，下一步是 OpenManus adapter 和 AEvo-lite，而不是再加一个聊天 UI。

## License

MIT

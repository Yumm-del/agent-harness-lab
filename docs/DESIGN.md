# 设计说明

这份文档解释「为什么这样搭」，而不是重复 README 的用法。面试时如果被追问设计取舍，可以从这里挑一条展开。

## 1. 问题定义：Harness 是一个可对照的系统

同一个模型，换 harness 之后成功率会变。harness 至少包含：

- Agent loop：什么时候继续、什么时候停；
- Tool interface：工具 schema、参数校验、错误如何回传给模型；
- Context management：历史消息保留多少、怎么裁剪；
- Control：重试、修复提示、超时、步数预算；
- Routing：什么任务用便宜模型、什么任务升级；
- Observability：trace、指标、失败归因。

如果把这些都混在一个 `agent.py` 里，改一处就无法判断收益来自哪里。本项目的做法是：

1. 把可变部分收敛到 `AgentSpec`；
2. 每次只改一个变量；
3. 用同一批任务、同一个 validator、同一个模型（除模型对比外）跑；
4. 用 repeat 量出噪声，再判断差异。

## 2. 多智能体：planner / executor 是一个可降级的增强

为什么把 `architecture` 作为第五个轴，而不是另写一个 agent 项目：

- 同一批任务、同一个 validator、同一个 executor loop，只多一次 planner 调用，差异才能归因到拓扑；
- 如果另起炉灶，成功率变化可能来自工具、任务或提示词，不可比。

实现：

1. planner 只拿任务 + planner prompt，`tools=[]`，输出 3–6 步计划；
2. executor 的 system prompt = instruction + 操作规则 + `Planner handoff (a hint, not ground truth)`；
3. 计划不伪造一条 assistant 消息，保持标准的 `system → user → assistant/tool` 序列，replay 才能对齐；
4. planner 失败或返回空计划时降级：记录 `llm_error` + `handoff(ok=false)`，executor 按无计划模式继续；
5. trace 的 `llm_call` / `tool_call` 带 `agent` 字段，`handoff` 事件记录 plan、token、延迟；
6. 失败归因新增 `planner_error`。

为什么失败要降级而不是中断：

- 多智能体是增强，不应该因为一次额外调用把一个原本能过的任务拖死；
- 降级行为本身可观测（`handoff ok=false`），不是静默失败。

代价（实测）：

- planner token 只占 5.6%，但总 token +64.7%、P50 +82.7%；成本主要来自计划改变了 executor 的步数和上下文，不是那一次 planner 调用；
- 在这 16 个短任务上没有成功率收益；“计划里说重试”不能替代工具层重试。

局限：

- 只有 planner/executor 一种拓扑；没有 verifier、多轮协商、handoff 失败重试；
- 计划质量没有单独评分；`handoff ok` 只代表产出了非空计划，不代表计划正确。

## 3. 任务设计：小、确定、可解释

任务不是为了模拟真实业务，而是为了让 harness 的差异可观察。

每个任务必须满足：

- **输入确定**：fixtures 写死在代码里，不依赖外部文件或网络；
- **输出确定**：validator 能精确判断，不靠 LLM 打分；
- **边界清楚**：prompt 明确写出输出文件名、字段、类型、排序；
- **失败可归因**：trace 能区分「没读」「读错」「算错」「没写」。

16 个任务覆盖 4 类能力：

| 类别 | 主要观测 |
|---|---|
| file_data | 基本工具使用、格式遵守、类型保持 |
| multi_step | 多步计划、配置驱动、跨工具组合 |
| recovery | 工具失败、脏数据、文件缺失、参数错误 |
| long_horizon | 多步之后还能不能记住目标与中间结果 |

`rc_01` 是唯一一个刻意让 baseline 失败的 task：`max_steps=3` + 前两次 `http_get` 超时。baseline 没有自动重试，第三次成功后已经没有步数写文件；repair 在 harness 层重试，第一步就拿到结果。这样成功率的差异来自 harness 控制策略，而不是 prompt 措辞。

## 4. Trace：先有证据，再有结论

trace 的设计目标是「能回放、能归因」，不是「日志好看」。

每个 `llm_call` 存：

- 完整 `messages`（system、user、assistant tool_calls、tool results）；
- `request_hash`：对 messages 做稳定哈希；
- `response`：content、tool_calls、usage、latency；
- prompt/completion token 与延迟。

每个 `tool_call` 存：

- 工具名、参数；
- `ok`、`attempts`、`failed_attempts`、`attempt_errors`；
- 延迟。

为什么存完整 messages 而不是只存最后答案：

- 失败归因需要知道模型看到了什么；
- 回放需要重建请求；
- `request_hash` 可以检测「代码/提示词/工具变了，历史响应不再适用」。

## 5. 回放：在 LLM 边界录制，在工具边界重跑

回放不是重放日志，而是：

1. 从 trace 按顺序取 LLM 响应；
2. 工具真实执行（文件、mock API、故障注入都会重新发生）；
3. 每一步比较 `request_hash`；
4. 用同一个 validator 重新判定。

这样可以在不花 API 钱的情况下，验证：

- 改 harness 代码后，消息序列有没有变；
- 工具行为是否仍然确定；
- 修复是否让原来失败的 task 通过。

局限：如果工具本身有随机性（真实网络、时间、外部数据库），回放不再完全确定。本项目用 mock API 和固定 fixtures 规避了这一点。

## 6. 失败分类：规则式，但可解释

分类顺序是有意设计的：

1. `llm_error`：API/网络问题，不是模型能力问题；
2. `planner_error`：planner 没产出计划，executor 降级运行后仍失败；根因在交接层，不在 executor 的步数预算；
3. `step_limit`：预算用完，通常是规划/恢复失败的结果；
4. `harness_error`：一次 LLM 调用都没有，说明 harness 自身坏了；
5. `planning_no_tool`：没调用任何工具，说明模型没理解「要动手」；
6. `recovery_loop`：同一工具 + 同一参数重复失败，说明没有从错误中学习；
7. `tool_failure`：有工具错误且没恢复；
8. `missing_artifact` / `format_error` / `execution_wrong_result`：输出层问题。

规则不完美，但它把「模型能力」「工具设计」「harness 控制」「输出格式」拆开了。相比一个笼统的「失败」，这种分类更能指导下一步改什么。

## 7. 指标：成功率高不代表更好

报告里同时看：

- 成功率；
- 每个 repeat 的成功率标准差（噪声提示）；
- 平均步数；
- 平均 token / 总 token；
- planner token 与 planner token 占比（多智能体）；
- handoff 成功 episode / 有交接的 episode（多智能体）；
- P50 / P95 延迟；
- 工具错误率（失败尝试 / 工具调用，可能 >100%）；
- recovery 类任务成功率；
- cascade 升级率。

一个配置可能成功率高 1 个任务，但 token 翻倍、P95 延迟翻倍。面试时应该主动说这个 trade-off，而不是只报成功率。

多智能体的例子更极端：`multi_agent` 成功率与 baseline 相同，但总 token +64.7%、P50 +82.7%。而 planner 调用本身只占 5.6% token——说明成本主要来自计划改变了 executor 的行为，不是多了一次调用。

## 8. Cascade：路由不是「一定成功」

`cascade` 的语义：

1. 先跑 weak spec；
2. validator 失败，才跑 strong spec；
3. 结果记在 `cascade` 下，`escalated=true`，usage 是两次之和。

它回答的问题不是「强模型能不能救」，而是「用弱模型省下的钱，是否值得在少数失败上付强模型的钱」。

局限：

- validator 失败才升级；如果 validator 本身有 bug，cascade 会做出错误路由；
- 强模型也可能失败；
- 当前是串行升级，延迟是两次之和；真实系统可以用异步/预取优化。

## 9. 沙箱边界

`Sandbox` 做三件事：

- 只允许相对路径；
- resolve 后检查是否在 workspace 内；
- `run_python` 用子进程 + 超时 + `-I` 隔离。

它不是生产安全沙箱：

- 没有系统调用级隔离（seccomp/容器）；
- 不能阻止 Python 访问网络；
- Windows 上子进程清理依赖 `subprocess` 的超时机制。

如果要做生产级，需要容器/微虚拟机、网络策略、资源配额、审计日志。README 明确写这个边界，是为了避免把「实验沙箱」说成「安全沙箱」。

## 10. 如果继续做

优先级从高到低：

1. **OpenManus adapter**：把同一批任务跑在真实开源 runtime 上，对比失败模式；这比再加 10 个合成任务更有信息量。
2. **AEvo-lite**：meta-agent 读失败报告，只改一个配置，再跑回归集；提升超过噪声才保留。
3. **Multi-agent v1（已完成）**：planner/executor + `agent` 字段 + `handoff` 指标 + 失败降级；下一步 solver/verifier、多轮 handoff 与长任务集。
4. **Held-out 任务**：dev/test 拆分，避免在测试集上调 harness。
5. **Prompt Cache / 成本**：已记录 cache hit token；下一步做 cache-aware 调度与按真实价格的成本-成功率 Pareto。

## 11. 面试时应该能回答的问题

- 为什么不用 LLM judge 做 pass/fail？
- 为什么 `rc_01` 要把 max_steps 压到 3？
- 为什么 `repair` 在简单任务上反而步数更多？
- `request_hash` 冲突/漂移时怎么办？
- 16 个任务的成功率能不能外推到真实业务？
- cascade 的升级条件有什么风险？
- 如果给你 10 倍预算，你先加任务还是先加模型？为什么？
- 为什么把 planner 的计划放 system prompt，而不是伪造一条 assistant 消息？
- planner 调用失败时为什么降级而不是中断？降级后怎么在 trace 里看出来？
- 多智能体为什么没有提升成功率？成本到底花在哪？
- `handoff ok` 和「计划正确」是一回事吗？

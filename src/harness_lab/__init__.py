"""agent-harness-lab：一个 trace-first 的 Agent Harness 实验场。

核心目标不是「再做一个 Agent」，而是把 Agent 的运行过程变成可评测、
可归因、可回放、可迭代的数据：

  - 每个 episode 记录完整 trace（LLM 请求/响应、工具调用、错误、耗时、token）
  - 用确定性 validator 判断任务是否完成，而不是只看模型自评
  - 把失败按「理解/规划/工具/执行/恢复/上下文」分类
  - 用同一批任务对比不同 harness 配置，看成功率、步数、成本、延迟
"""

__version__ = "0.1.0"

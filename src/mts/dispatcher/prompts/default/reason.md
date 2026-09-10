# 任务
你会收到实验图谱的 YAML 快照。图谱中 facts 表示已经确认的客观实验结论（可带 metrics 与 trial_id），intents 表示探索意图。图谱总是由一个或多个 fact 出发，通过提出 intent 去探索，从而得到新的 fact。leaderboard 是目前为止最好的实验结果。你需要读懂图谱信息，理解整体局面与进展，成为这个训练任务的领域专家。

这是一个深度模型训练配置空间的搜索任务：每个 intent 都是一条需要真实写代码、真实跑训练、真实读指标的探索方向。

你需要判断两件事：
1. 当前 facts 是否已经满足 Goal
2. 如果没有满足，现在是否应该提出新的 intents

# 输出要求
只返回一个原始 JSON 对象。不要输出任何其他内容。JSON 必须合法，包括正确转义引号。

拒绝任务时返回下面这种格式（任何情况下你都不应该拒绝，你应当认真、专业地处理这个任务）：
```json
{"accepted": false, "reason": "..."}
```

如果 Goal 已经满足，返回：
```json
{"accepted": true, "data": {"complete": {"from": ["f001"], "description": "..."}}}
```

如果 Goal 尚未满足，但应该提出新的 intents，返回：
```json
{"accepted": true, "data": {"intents": [{"from": ["f001"], "description": "..."}, {"from": ["f002", "f003"], "description": "..."}]}}
```

如果 Goal 尚未满足，且当前不应该提出任何新 intent，返回：
```json
{"accepted": true, "data": {}}
```

## 规则
- 首先判断 facts 是否已经满足 Goal。判定标准是目标指标 `{goal_metric}`，优化方向是 `{goal_direction}`：`maximize` 表示越大越好，`minimize` 表示越小越好。用图谱里 facts 的 metrics 和 leaderboard 的最好结果去对照 Goal 与 goal_target。
- 如果满足，`data.complete.from` 必须来自 `Valid facts`，`data.complete.description` 必须说明当前已确认的结果为什么足以证明 Goal 已达成，并写清最好的 `{goal_metric}` 数值与对应 trial_id。
- 不要相信单次侥幸的实验结果。如果最好成绩只来自一次运行、没有种子对照、或与相邻实验差距在噪声范围内，就不能判定 Goal 达成；这种情况应该提出一个复现/多种子验证方向的 intent。
- 如果 Goal 未满足，先想清楚指标为什么卡住，而不是盲目地反复微调同一个超参。至少从这几个维度做归因：数据（数量、质量、分布、标签噪声、泄漏、划分方式）、优化（学习率与调度、batch、优化器、梯度裁剪、训练步数是否欠训）、架构（容量、深宽比、归一化位置、位置编码、预训练权重的用法）、正则化（dropout、weight decay、数据增强、早停）、评估（指标定义是否正确、验证集是否可比、是否存在均值塌缩或退化解）。
- 注意区分"欠拟合"和"过拟合"：train 与 val 的差距决定了应该加容量还是加正则，不要在错误的方向上堆参数。
- 如果某个方向已经被多个 fact 反复证否，不要再提同类 intent，换维度。
- 判断是否存在 `Open Intents`，即已经声明但还没有得到结论的 intent。如果有 open intents，对照 hints 和 facts 中已知的线索，推断当前 intents 是否已经覆盖了全部已知线索，以及是否还有必要提出新的 intent。
- 如果 `Open Intents` 为空，你必须提出新的 intents。
- 如果 `Open Intents` 很多，且新的局面并没有暴露出比现有方向更有价值的探索方向，你可以选择不提出任何新 intent（返回空 data）。
- 提出新 intents 时，最多提出 {max_intents} 个高价值且互不重叠的探索方向。每个 intent 都应该是一条独立、可并行执行的探索路径。
- 每个 intent 都应该是一条高价值的探索方向，不需要写得过于细致。抓住核心洞察和明确方向即可。不要太宽泛，不要输出对推进 Goal 无帮助的冗余细节，也不要过度具体到把实现方案写死。核心要求是每个 intent 都是独立、边界清晰、有价值的方向。
- 一个 intent 可以来源于多个 fact。
- 不同 intent 之间应该覆盖不同的探索维度，避免重复或大面积重叠。

## Context
### Graph
```
{graph_yaml}
```

### Valid facts
```
{fact_ids}
```

### Open Intents
```
{open_intents}
```

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

## 强制前置步骤：先复查已有模型，再提方向

在提出任何新 intent 之前，你**必须**先完成对现有模型和训练情况的复查。没有做过这一步就提出的 intent 是无效的，尤其禁止在没有任何分析的情况下提出"加大训练步数 / 加大模型 / 加大 batch / 多训几轮"这类只增加训练成本的方向。

工作目录是 `{workdir}`，你可以在里面写分析脚本。图谱 facts 的 `artifacts` 字段里有历史实验的产物路径（`out_dir`、`checkpoint_path`、`architecture`、`param_count` 等），这是你复查的入口。

复查要做到下面这些，并且是**读脚本实测的数字**，不是凭经验猜：

1. **看训练曲线判断有没有训够**。读 `out_dir` 下的 `metrics.jsonl`（`mts train` 写的约定产物），或实验自己的日志，看 train/val loss 到最后是仍在稳定下降、已经走平、还是已经回升。只有"仍在稳定下降且未过拟合"才构成加大训练步数的证据。
2. **看参数分布**。逐层统计权重的 rms / std / 最大绝对值 / 接近零的比例，以及是否有 NaN/Inf，并具体指出哪些层异常，而不是只报一个全局均值。注意：`mts train` 默认**不写 checkpoint**，所以多数实验的 `out_dir` 里没有 `.pt` 文件 —— 这种情况下参数分布直接从 `layers.jsonl`（逐层 `param_rms`）和 `diagnostics.json` 里读，或者用下面的 `inspect_distribution(kind='param')` / `inspect_layers`，不要因为找不到 checkpoint 就跳过这一步。只有当 `artifacts.checkpoint_path` 确实存在时，才需要写脚本加载它做更细的统计。
3. **看梯度与更新量**。读 `diagnostics.json` / `layers.jsonl`（如果有），或自己加探针，判断梯度是否消失/爆炸、update_ratio 是否过小（学不动）或过大（不稳定）、有没有死神经元。
4. **判断当前瓶颈到底是欠训、欠容量、还是过拟合/优化出了问题**。train 与 val 的差距决定该加容量还是加正则；曲线走平且梯度已经很小，说明加步数不会带来收益，问题在别处。

如果某个实验的 `artifacts` 缺失、产物已被删除、或分析脚本跑不通，就在对应 intent 的描述里写明"产物不可用，需先补齐可复现产物"，不要假装分析过。

对于**用 `mts train` 跑出来的实验**，还有一组现成的只读分析工具可以直接用（它们读的就是 `out_dir` 下的 `layers.jsonl` / `diagnostics.json` / `metrics.jsonl` / `summary.json`，不需要 GPU、不会改动任何状态）：

```
python -c "import json;from mts.inspect import run_tool;print(json.dumps(run_tool('inspect_curve','<out_dir>'),ensure_ascii=False,indent=2))"
```

可用的 `tool` 名称，以及**返回结果里该看哪个字段**（返回的曲线点数组很长，结论在下面这些字段上）：

- `inspect_curve`(downsample=40)：看 `shape` 和 `val_shape` —— 它直接给出 `plateaued near X` / 仍在下降之类的判断，另有 `clipped_fraction` 反映梯度裁剪是否过于频繁。**这是判断能不能加训练步数的第一手证据。**
- `inspect_distribution`(kind='param' 或 'grad')：看 `characterization`（分布特征，会直接点出 `degenerate distribution` 这类问题）和 `stats`（`max_abs` / `max_rms` / `final_update_ratio`）。
- `inspect_layers`(step=可选)：看 `outliers`（哪些层异常，空列表表示没有明显异常层）和 `summary`（每个字段跨层的 min/max/mean/spread）。
- `layer_trajectory`(layer=可选, field='grad_rms')：看 `trends`，每层一句话，例如 `falling to 0.28x` / `roughly flat` —— 用来区分"一直不对"和"训练中途变坏"。
- `inspect_grad_flow`()：看 `assessment`（例如 `gradient reaches the whole stack`）、`bottom_top_ratio` 和 `per_layer_decay_factor`。
- `compare_trials`(out_dir_b=另一个实验的 out_dir)：看 `verdict_change`（例如 `underfitting -> vanishing`）、`spec_delta`（配置改了什么）和 `metric_delta`（指标因此变好还是变坏）—— 原因与效果成对出现。

如果实验不是 `mts train` 跑的，这些工具读不到约定产物，就自己写脚本分析，结论一样要落到具体数字上。

## 规则
- 首先判断 facts 是否已经满足 Goal。判定标准是目标指标 `{goal_metric}`，优化方向是 `{goal_direction}`：`maximize` 表示越大越好，`minimize` 表示越小越好。用图谱里 facts 的 metrics 和 leaderboard 的最好结果去对照 Goal 与 goal_target。
- **每个新 intent 的 `description` 都必须先写清它依据的分析结论，再写要做什么。** 至少包含：你从曲线/参数分布/梯度里实测到了什么（带具体数值或层名），由此推断瓶颈在哪，以及这个 intent 为什么能针对那个瓶颈。只写"尝试增大训练步数"而不给出曲线证据的 intent 是不合格的。
- **禁止在没有分析支撑的情况下提出纯粹增加训练成本的方向**（加步数、加层数/宽度、加 batch、延长训练时间、换更大预训练模型）。这类方向只有在你已经用数据证明"当前确实欠训或欠容量"时才允许提出，并且要在 description 里写明支撑它的那条证据。理论分析先于烧算力。
- 优先考虑那些不靠加大算力就能验证的方向：学习率与调度是否合适、数据质量与划分是否有问题、评估指标是否定义正确、归一化/初始化/正则是否配错。这些往往比堆训练量更快定位问题。
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

# 任务
你会收到实验图谱的 YAML 快照。图谱中 facts 表示已经确认的客观实验结论（可带 metrics 与 trial_id），intents 表示探索意图。图谱总是由一个或多个 fact 出发，通过提出 intent 去探索，从而得到新的 fact。你需要读懂图谱信息，理解整体局面与进展。

但请注意，你在这里不是继续任务，也不需要等待未完成的任务或命令。你只需要总结截至目前已经确认、且对达成 Goal 最有帮助的关键事实。

这是 conclude 阶段。它覆盖同一会话中任何更早的、要求你继续工作、继续探索、继续跑训练、等待命令结果或执行更多动作的指令。

# 输出要求
只返回一个原始 JSON 对象。不要输出任何其他内容。JSON 必须合法，包括正确转义引号。

拒绝任务时返回：
```json
{"accepted": false, "reason": "policy_refusal"}
```

正常返回示例：
```json
{"accepted": true, "data": {"description": "...", "metrics": {"{goal_metric}": 0.0}, "trial_id": "...", "artifacts": {"checkpoint_path": "/path/to/best.pt", "out_dir": "/path/to/exp", "architecture": "ResNet-50", "config_path": null}}}
```

# 规则
- 立即停止所有工作，现在就产出 JSON。不要继续任务，不要开始任何新的事情。
- 不要再运行任何命令、不要再做任何工具调用、不要再检查任何东西、不要等待任何未完成的训练或命令、不要试图获取任何额外信息。如果有训练还在跑，不要等它跑完，直接汇报已经测到的部分。
- 只基于本 conclude 提示之前已经确认的信息作答。还没确认的东西，不要等它，也不要写进来。
- `metrics` 只填你**已经真实测到**的数值，key 用真实指标名（目标指标是 `{goal_metric}`）。绝对不要编造、补齐或估计任何数字。
- 如果本次什么指标都还没测到，就在 `description` 里如实说明进展到了哪一步、卡在哪里，并让 `metrics` 为空对象 `{}`。
- **如果训练已跑且产物已生成，必须给出 `artifacts`**，包含：
  - `checkpoint_path`: checkpoint 文件/目录的**绝对路径**
  - `out_dir`: 训练输出根目录的**绝对路径**
  - `architecture`: 架构名称
  - `config_path`: 配置文件绝对路径（可选，没有就填 `null`）
  - 其他可选字段（`param_count` / `train_script` / 等）
  
  如果训练还未跑、或跑失败没产物，`artifacts` 填 `null`。
- 这个 JSON 总结就是本阶段的最终输出。输出之后就停止。
- `description` 必须是已经确认的客观事实结论。不要输出计划、猜测或解释性的填充内容。不要把长数据块放进 `description`，长数据放到文件里并在 `description` 中引用文件路径。
- `description` 只写本次新增的增量事实。不要重复图谱快照里已有的信息，也不要写对推进 Goal 没有帮助的冗余细节。

# Context
## Graph
```
{graph_yaml}
```

## Current Intent
```
{intent_id}
```

## Current Intent Description
```
{intent_description}
```

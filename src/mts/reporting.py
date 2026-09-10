"""Experiment report generation (docs/09 P3-2).

Generates a human-readable Markdown report from the board state: the fact graph
(what was discovered and why), the trial ledger (ranked by the goal metric), the
knowledge distilled, and a reproducible command trail. This is MTS's analog of
Cairn's `Report` entity — a report is *derived* from the graph, not hand-written.
"""

from __future__ import annotations

from typing import Any

from mts.server.services import Service


def build_report(service: Service, project_id: str) -> str:
    """Render a full Markdown report for a project."""
    proj = service.get_project(project_id)
    if proj is None:
        raise KeyError(f"project not found: {project_id}")

    lines: list[str] = []
    lines.append(f"# 实验报告 · {proj['title']}")
    lines.append("")
    lines.append(_meta_section(proj))
    lines.append("")
    lines.append(_goal_section(proj))
    lines.append("")
    lines.append(_leaderboard_section(service, project_id))
    lines.append("")
    lines.append(_timeline_section(service, project_id))
    lines.append("")
    lines.append(_knowledge_section(service))
    lines.append("")
    lines.append(_reproduce_section(service, project_id))
    return "\n".join(lines).rstrip() + "\n"


def _meta_section(proj: dict[str, Any]) -> str:
    return (
        f"- 项目 ID：`{proj['id']}`\n"
        f"- 状态：{proj['status']}\n"
        f"- 实验总数：{proj.get('trial_count', 0)}\n"
        f"- 最优 val_loss：{proj.get('best_val_loss')}\n"
        f"- 最优实验：`{proj.get('best_trial_id')}`\n"
    )


def _goal_section(proj: dict[str, Any]) -> str:
    return (
        "## 目标\n\n"
        f"- **Origin**：{proj['origin']}\n"
        f"- **Goal**：{proj['goal']}\n"
        f"- 指标：`{proj['goal_metric']}`，方向 `{proj['goal_direction']}`，"
        f"阈值 `{proj['goal_target']}`\n"
    )


def _leaderboard_section(service: Service, project_id: str) -> str:
    """Leaderboard based on facts table (autonomous-agent flow)."""
    proj = service.get_project(project_id)
    if proj is None:
        return "## 排行榜\n\n（项目不存在）"

    facts = service.list_facts(project_id)
    goal_metric = proj["goal_metric"]
    goal_direction = proj["goal_direction"]

    # Filter facts with the goal metric
    ranked = []
    for f in facts:
        if f["id"] in ("origin", "goal"):
            continue
        metrics = f.get("metrics") or {}
        if goal_metric in metrics:
            ranked.append({
                "id": f["id"],
                "description": f["description"],
                "value": metrics[goal_metric],
                "metrics": metrics,
                "trial_id": f.get("trial_id"),
            })

    # Sort by goal metric
    if goal_direction == "maximize":
        ranked.sort(key=lambda x: x["value"], reverse=True)
    else:
        ranked.sort(key=lambda x: x["value"])

    lines = [f"## 排行榜（按 {goal_metric} {'降序' if goal_direction == 'maximize' else '升序'}）", ""]
    if not ranked:
        lines.append("（暂无实验数据）")
        return "\n".join(lines)

    lines.append(f"| 排名 | 事实 | {goal_metric} | 其他指标 |")
    lines.append("|---|---|---|---|")
    for i, fact in enumerate(ranked[:10], start=1):
        other_metrics = {k: v for k, v in fact["metrics"].items() if k != goal_metric}
        other_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                              for k, v in other_metrics.items())
        if not other_str:
            other_str = "-"
        value_str = f"{fact['value']:.4f}" if isinstance(fact['value'], float) else str(fact['value'])
        desc = fact['description'][:50] + "..." if len(fact['description']) > 50 else fact['description']
        lines.append(f"| {i} | {desc} | {value_str} | {other_str} |")

    return "\n".join(lines)


def _timeline_section(service: Service, project_id: str) -> str:
    """The fact graph as a chronological evidence chain."""
    facts = service.list_facts(project_id)
    lines = ["## 探索过程（事实链）", ""]
    if not facts:
        lines.append("（暂无事实）")
        return "\n".join(lines)
    for f in facts:
        trial = f" → `{f['trial_id']}`" if f.get("trial_id") else ""
        lines.append(f"- {f['description']}{trial}")
    return "\n".join(lines)


def _knowledge_section(service: Service) -> str:
    notes = service.list_knowledge()
    lines = ["## 沉淀的经验", ""]
    if not notes:
        lines.append("（暂无知识条目）")
        return "\n".join(lines)
    for n in notes:
        lines.append(f"- **{n['title']}**（`{n['slug']}`，置信度 {n.get('confidence', 0.5):.2f}）")
    return "\n".join(lines)


def _reproduce_section(service: Service, project_id: str) -> str:
    """Reproducible command trail: the exact spec of each trial maps to `mts train`."""
    trials = service.list_trials(project_id)
    lines = ["## 复现命令", ""]
    if not trials:
        lines.append("（暂无实验）")
        return "\n".join(lines)
    lines.append("每个实验的完整配置在 `runs/<id>/spec.json`，可复现：")
    lines.append("")
    lines.append("```bash")
    for t in trials[:5]:  # top-N to keep the report readable
        lines.append(f"mts train --spec {t['spec_path']} --out-dir {t['out_dir']}")
    lines.append("```")
    return "\n".join(lines)

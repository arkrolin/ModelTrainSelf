"""Markdown-backed shared knowledge space.

Lessons are Markdown files under `memory/lessons/` so humans can read and edit
them with any editor and git can diff them. Each lesson carries YAML frontmatter
(title / tags / confidence) plus a structured `trigger` and `evidence` section so
agents can retrieve lessons by matching a situation, and humans can audit where a
claim came from.

File format:

    ---
    title: "Post-norm 对学习率的敏感度显著高于 pre-norm"
    tags: ["norm", "lr", "stability"]
    confidence: 0.7
    ---
    # 标题

    ## Trigger
    - verdict ∈ {diverged, unstable}
    - arch.norm_position = post

    ## Advice
    post-norm 的等效 lr 需降到 pre-norm 的 1/3 …

    ## Evidence
    - t002 (diverged): pre→post 后 38 步发散

    ## Body
    ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Lesson:
    slug: str
    title: str
    tags: list[str] = field(default_factory=list)
    confidence: float = 0.5
    trigger: dict[str, Any] = field(default_factory=dict)
    advice: str = ""
    evidence: list[str] = field(default_factory=list)
    body: str = ""
    path: Path | None = None

    def to_markdown(self) -> str:
        front = {
            "title": self.title,
            "tags": self.tags,
            "confidence": self.confidence,
        }
        lines = ["---", yaml.safe_dump(front, allow_unicode=True, sort_keys=False).rstrip(), "---", ""]
        lines.append(f"# {self.title}\n")
        if self.trigger:
            lines.append("## Trigger")
            for k, v in self.trigger.items():
                lines.append(f"- {k} = {v}")
            lines.append("")
        if self.advice:
            lines.append("## Advice")
            lines.append(self.advice.strip())
            lines.append("")
        if self.evidence:
            lines.append("## Evidence")
            for e in self.evidence:
                lines.append(f"- {e}")
            lines.append("")
        if self.body:
            lines.append("## Body")
            lines.append(self.body.strip())
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    @classmethod
    def from_markdown(cls, slug: str, text: str, path: Path | None = None) -> "Lesson":
        lesson = cls(slug=slug, title=slug)
        lesson.path = path
        # Split frontmatter.
        if text.startswith("---"):
            _, fm, rest = text.split("---", 2)
            try:
                meta = yaml.safe_load(fm) or {}
                lesson.title = meta.get("title", slug)
                lesson.tags = meta.get("tags", [])
                lesson.confidence = float(meta.get("confidence", 0.5))
            except yaml.YAMLError:
                pass
        else:
            rest = text

        lesson.body = rest.strip()
        # Parse sections heuristically.
        sections = _split_sections(rest)
        for heading, content in sections.items():
            if heading == "trigger":
                lesson.trigger = _parse_kv(content)
            elif heading == "advice":
                lesson.advice = content.strip()
            elif heading == "evidence":
                lesson.evidence = [line.lstrip("- ").strip() for line in content.splitlines() if line.strip()]
        return lesson


def _split_sections(text: str) -> dict[str, str]:
    import re

    out: dict[str, str] = {}
    current = None
    for line in text.splitlines():
        m = re.match(r"^##\s+(.+)$", line)
        if m:
            current = m.group(1).lower()
            out[current] = ""
        elif current:
            out[current] += line + "\n"
    return out


def _parse_kv(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip().lstrip("- ").strip()
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


class KnowledgeStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, slug: str) -> Path:
        return self.root / f"{slug}.md"

    def list_slugs(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.md") if p.name != "INDEX.md")

    def load(self, slug: str) -> Lesson | None:
        path = self._path(slug)
        if not path.exists():
            return None
        return Lesson.from_markdown(slug, path.read_text(encoding="utf-8"), path)

    def load_all(self) -> list[Lesson]:
        return [self.load(s) for s in self.list_slugs() if self.load(s) is not None]

    def save(self, lesson: Lesson) -> Path:
        path = self._path(lesson.slug)
        path.write_text(lesson.to_markdown(), encoding="utf-8")
        return path

    def match(self, verdict: str | None = None, spec: dict[str, Any] | None = None,
              limit: int = 5) -> list[Lesson]:
        """Retrieve lessons whose trigger matches the current situation.

        A trigger is a dict of `field = value` where `field` may be a dotted spec
        path (e.g. `arch.norm_position`) or a special key `verdict`. A lesson
        matches when every trigger condition holds.
        """
        scored: list[tuple[float, Lesson]] = []
        for lesson in self.load_all():
            score = _trigger_score(lesson, verdict, spec)
            if score > 0:
                scored.append((score, lesson))
        scored.sort(key=lambda x: (-x[0], -x[1].confidence))
        return [s[1] for s in scored[:limit]]

    def reindex(self) -> Path:
        lessons = self.load_all()
        lines = ["# Knowledge Index\n", f"共 {len(lessons)} 条经验。\n"]
        for lesson in sorted(lessons, key=lambda l: (-l.confidence, l.slug)):
            tags = ", ".join(f"`{t}`" for t in lesson.tags)
            lines.append(f"- **{lesson.slug}** ({lesson.confidence:.1%}) — {lesson.title}  [{tags}]")
        index_path = self.root / "INDEX.md"
        index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return index_path


def _trigger_score(lesson: Lesson, verdict: str | None, spec: dict[str, Any] | None) -> float:
    if not lesson.trigger:
        return 0.0
    total = 0.0
    matched = 0.0
    for key, value in lesson.trigger.items():
        total += 1.0
        if key == "verdict":
            if verdict is not None and str(value) == verdict:
                matched += 1.0
            continue
        # Dotted spec path lookup.
        actual = _lookup(spec or {}, key)
        if actual is not None and str(actual) == str(value):
            matched += 1.0
    if total == 0:
        return 0.0
    return matched / total


def _lookup(spec: dict[str, Any], dotted: str) -> Any:
    node: Any = spec
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node

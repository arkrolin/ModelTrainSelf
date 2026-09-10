"""CLI for the shared Markdown knowledge space."""

from __future__ import annotations

from pathlib import Path

from mts.knowledge.store import KnowledgeStore


def _store(root: str | None) -> KnowledgeStore:
    base = Path(root) if root else Path.cwd() / "memory" / "lessons"
    return KnowledgeStore(base)


def list_lessons(tag: str | None, root: str | None) -> None:
    store = _store(root)
    lessons = store.load_all()
    if tag:
        lessons = [l for l in lessons if tag in l.tags]
    if not lessons:
        print("(no lessons)")
        return
    for lesson in sorted(lessons, key=lambda l: (-l.confidence, l.slug)):
        tags = ", ".join(lesson.tags)
        print(f"{lesson.slug:<32} conf={lesson.confidence:.1%}  {lesson.title}  [{tags}]")


def show_lesson(slug: str, root: str | None) -> None:
    store = _store(root)
    lesson = store.load(slug)
    if lesson is None:
        print(f"lesson not found: {slug}")
        return
    print(lesson.to_markdown())


def reindex(root: str | None) -> None:
    store = _store(root)
    index = store.reindex()
    print(f"wrote {index}")

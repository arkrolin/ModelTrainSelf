"""Agent 活动流：读 claude CLI 自己写的 session transcript，翻成人类可读的进展。

训练任务动辄几十分钟到几小时。在此之前 WebUI 只能看到调度器自己的日志
（"starting worker process ..."），agent 在里面读什么、写什么、跑了哪条命令
全是黑盒，看起来就像卡死了。

**为什么读 transcript 而不是给 CLI 加 `--output-format stream-json`**：
流式方案要改 driver argv，还要重写 `extract_response_text`（输出变成 JSONL，
最终 JSON 得从 result 事件里挖）——那是 conclude 解析和 session 提取都依赖的
执行主路径，为了看进度去动它，风险不对等。transcript 是纯旁路：CLI 本来就在
写，读它不影响任务成败，读坏了也只是看不到进度。额外好处是任务超时/失败后
仍能回看（stdout 那时已经丢了），调度器重启也不断流。

代价是延迟 1~20 秒（一条事件在工具调用结束后才落盘），对小时级任务无所谓。

`TranscriptTail` 按字节偏移增量读，只解析新追加的部分。
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

RING_SIZE = 400
SUMMARY_LIMIT = 220
DETAIL_LIMIT = 2000
# CLI 把 transcript 写在 ~/.claude/projects/<转义后的 cwd>/<session-id>.jsonl。
# cwd 的转义规则是 CLI 内部实现，不去复刻；session-id 由 driver 自己生成，
# 直接拿它 glob 就能唯一命中，也就不依赖那套规则。
TRANSCRIPT_ROOT_ENV = "MTS_TRANSCRIPT_ROOT"


def transcript_root() -> Path:
    override = os.environ.get(TRANSCRIPT_ROOT_ENV)
    if override:
        return Path(override).expanduser()
    return (
        Path(
            os.environ.get("CLAUDE_CONFIG_DIR") or (Path.home() / ".claude")
        ).expanduser()
        / "projects"
    )


def find_transcript(session_id: str, root: Path | None = None) -> Path | None:
    """按 session-id 找 transcript 文件；找不到返回 None（CLI 可能还没建）。"""
    if not session_id:
        return None
    base = root or transcript_root()
    try:
        for candidate in base.glob(f"*/{session_id}.jsonl"):
            return candidate
    except OSError:
        return None
    return None


@dataclass(slots=True)
class ActivityEvent:
    """Agent 干的一件事。`summary` 是给人看的一行字。"""

    at: str
    project_id: str
    worker: str
    phase: str
    kind: str  # thinking | text | tool | tool_result | error
    summary: str
    seq: int = 0
    intent_id: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _shorten_path(path: str, workdir: str | None) -> str:
    """去掉工作目录前缀，长绝对路径在一行摘要里没有信息量。"""
    if workdir and path.startswith(workdir):
        return path[len(workdir) :].lstrip("/") or "."
    return path


# 每个工具挑一个最能说明"在干什么"的入参字段：命令和路径信息量最大，
# 其余退化到 description。
_TOOL_FIELDS: tuple[str, ...] = (
    "command",
    "file_path",
    "path",
    "pattern",
    "query",
    "url",
    "notebook_path",
    "description",
    "prompt",
)
_PATH_FIELDS = frozenset({"file_path", "path", "notebook_path"})


def summarize_tool(name: str, tool_input: Any, workdir: str | None = None) -> str:
    """一行话说明这次工具调用在做什么。"""
    name = name or "tool"
    if not isinstance(tool_input, dict):
        return name
    for field_name in _TOOL_FIELDS:
        value = tool_input.get(field_name)
        if not isinstance(value, str) or not value.strip():
            continue
        if field_name in _PATH_FIELDS:
            value = _shorten_path(value, workdir)
        return f"{name}: {_clip(value, SUMMARY_LIMIT)}"
    return name


def _tool_result_text(block: dict[str, Any]) -> str:
    """tool_result 的 content 可能是字符串，也可能是 content-block 列表。"""
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p)
    return ""


def parse_transcript_line(
    line: str, *, workdir: str | None = None
) -> list[tuple[str, str, str | None]]:
    """把 transcript 的一行解析成若干 `(kind, summary, detail)`。

    只认 assistant / user 两类事件，其余（attachment、ai-title 等 CLI 内部记录）
    对"agent 在干什么"没有信息量，直接跳过。解析失败返回空列表：transcript 是
    旁路，读不动不该让调用方出错。
    """
    line = line.strip()
    if not line:
        return []
    try:
        event = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(event, dict):
        return []

    etype = event.get("type")
    message = event.get("message")
    if etype not in ("assistant", "user") or not isinstance(message, dict):
        return []
    content = message.get("content")
    if isinstance(content, str):
        text = _clip(content, SUMMARY_LIMIT)
        return [("text", text, None)] if text else []
    if not isinstance(content, list):
        return []

    out: list[tuple[str, str, str | None]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text = block.get("text", "").strip()
            if text:
                out.append(
                    ("text", _clip(text, SUMMARY_LIMIT), _clip(text, DETAIL_LIMIT))
                )
        elif btype == "thinking":
            text = (block.get("thinking") or "").strip()
            if text:
                out.append(("thinking", _clip(text, SUMMARY_LIMIT), None))
        elif btype == "tool_use":
            detail = None
            tool_input = block.get("input")
            if isinstance(tool_input, dict):
                with_json = json.dumps(tool_input, ensure_ascii=False, default=str)
                detail = _clip(with_json, DETAIL_LIMIT)
            out.append(
                (
                    "tool",
                    summarize_tool(block.get("name", "tool"), tool_input, workdir),
                    detail,
                )
            )
        elif btype == "tool_result":
            text = _tool_result_text(block)
            if not text.strip():
                continue
            kind = "error" if block.get("is_error") else "tool_result"
            out.append((kind, _clip(text, SUMMARY_LIMIT), _clip(text, DETAIL_LIMIT)))
    return out


class TranscriptTail:
    """按字节偏移增量读一个 transcript，只解析新追加的部分。

    尾部可能读到写了一半的行，所以只在遇到换行时推进 offset，残行留到下次。
    """

    def __init__(self, path: Path, *, workdir: str | None = None):
        self.path = path
        self.workdir = workdir
        self._offset = 0
        self._pending = ""

    def read_new(self) -> list[tuple[str, str, str | None]]:
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size < self._offset:  # 文件被重建，从头再来
            self._offset = 0
            self._pending = ""
        if size == self._offset:
            return []
        try:
            with self.path.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(self._offset)
                chunk = fh.read()
                self._offset = fh.tell()
        except OSError:
            return []

        buffer = self._pending + chunk
        lines = buffer.split("\n")
        self._pending = lines.pop()  # 最后一段没有换行结尾，可能是残行
        out: list[tuple[str, str, str | None]] = []
        for line in lines:
            out.extend(parse_transcript_line(line, workdir=self.workdir))
        return out


@dataclass(slots=True)
class _ProjectFeed:
    events: deque[ActivityEvent] = field(
        default_factory=lambda: deque(maxlen=RING_SIZE)
    )
    seq: int = 0


class ActivityBus:
    """按项目存 agent 活动的环形缓冲，供 WebUI 轮询。

    进程内单例：WebUI 的「启动搜索」把调度循环跑在同进程后台线程里，publish
    与读取天然同进程。CLI `mts dispatch` 下没人读，publish 只是无害空转。
    """

    def __init__(self) -> None:
        self._feeds: dict[str, _ProjectFeed] = defaultdict(_ProjectFeed)
        self._lock = threading.Lock()

    def publish(
        self,
        project_id: str,
        *,
        worker: str,
        phase: str,
        kind: str,
        summary: str,
        detail: str | None = None,
        intent_id: str | None = None,
    ) -> ActivityEvent:
        with self._lock:
            feed = self._feeds[project_id]
            feed.seq += 1
            event = ActivityEvent(
                at=_now(),
                project_id=project_id,
                worker=worker,
                phase=phase,
                kind=kind,
                summary=summary,
                seq=feed.seq,
                intent_id=intent_id,
                detail=detail,
            )
            feed.events.append(event)
        return event

    def events(
        self, project_id: str, *, after_seq: int = 0, limit: int = 200
    ) -> list[ActivityEvent]:
        with self._lock:
            feed = self._feeds.get(project_id)
            if feed is None:
                return []
            picked = [e for e in feed.events if e.seq > after_seq]
        return picked[-limit:]

    def latest_seq(self, project_id: str) -> int:
        with self._lock:
            feed = self._feeds.get(project_id)
            return feed.seq if feed else 0

    def clear(self, project_id: str) -> None:
        with self._lock:
            self._feeds.pop(project_id, None)


_BUS = ActivityBus()


def get_activity_bus() -> ActivityBus:
    return _BUS


class ActivityWatcher:
    """worker 跑的期间，后台线程把它的 transcript 增量喂进 bus。

    整体 best-effort：transcript 找不到、读不动、解析失败都只是看不到进度，
    绝不能影响任务本身。所以循环体整个包在 try 里，异常只记日志。

    用法（配合 with，保证线程一定收尾）：

        with ActivityWatcher(session, project_id=..., worker=..., phase=...):
            result = run_worker_process(...)
    """

    def __init__(
        self,
        session_id: str | None,
        *,
        project_id: str,
        worker: str,
        phase: str,
        intent_id: str | None = None,
        workdir: str | None = None,
        interval: float = 2.0,
        bus: ActivityBus | None = None,
        root: Path | None = None,
    ):
        self.session_id = session_id
        self.project_id = project_id
        self.worker = worker
        self.phase = phase
        self.intent_id = intent_id
        self.workdir = workdir
        self.interval = max(0.5, float(interval))
        self._bus = bus or get_activity_bus()
        self._root = root
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "ActivityWatcher":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def start(self) -> None:
        if not self.session_id or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name=f"activity-{self.project_id}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            # 停之前给一轮时间把尾部事件读完，否则最后几条动作会丢。
            self._thread.join(timeout=self.interval + 1.0)
            self._thread = None

    def _run(self) -> None:
        tail: TranscriptTail | None = None
        while True:
            stopping = self._stop.is_set()
            try:
                if tail is None:
                    path = find_transcript(self.session_id or "", self._root)
                    if path is not None:
                        tail = TranscriptTail(path, workdir=self.workdir)
                if tail is not None:
                    for kind, summary, detail in tail.read_new():
                        self._bus.publish(
                            self.project_id,
                            worker=self.worker,
                            phase=self.phase,
                            kind=kind,
                            summary=summary,
                            detail=detail,
                            intent_id=self.intent_id,
                        )
            except Exception:  # noqa: BLE001 - 旁路观测绝不能拖垮任务
                LOG.debug(
                    "activity watch failed project=%s", self.project_id, exc_info=True
                )
            if stopping:  # 收尾那轮读完再退
                return
            if self._stop.wait(self.interval):
                continue  # 再跑一轮把尾巴收干净，然后从上面的 stopping 分支退出

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(slots=True)
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    cancelled: bool = False
    cancel_reason: str | None = None
    duration_sec: float = 0.0

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.cancelled


@runtime_checkable
class ExecProcess(Protocol):
    """A worker process, regardless of whether it runs inside a container or on the host.

    Local mode uses LocalProcess. Both expose this surface so the task runners,
    heartbeat lease and cancellation stay backend-agnostic.
    """

    def start(self) -> None: ...

    def communicate(self, timeout: float | None) -> ProcessResult: ...

    def kill(self) -> None: ...

    def cancel(self, reason: str) -> None: ...

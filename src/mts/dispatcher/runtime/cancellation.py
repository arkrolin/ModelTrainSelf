from __future__ import annotations

import threading

from mts.dispatcher.runtime.process import ExecProcess


class TaskCancellation:
    """Task cancellation signal.

    A thread-safe "please stop" flag. The dispatcher sets it when a project is
    stopped/deleted; the heartbeat lease kills the attached process on hard failure.
    Combines MTS's existing API (cancel/cancelled/reason) with Cairn's process
    attachment.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: ExecProcess | None = None
        self._reason: str | None = None

    def attach_process(self, process: ExecProcess | None) -> None:
        with self._lock:
            self._process = process
            reason = self._reason
        if process is not None and reason is not None:
            process.cancel(reason)

    def cancel(self, reason: str = "cancelled") -> bool:
        """Signal cancellation. Returns True on first cancel (idempotent)."""
        with self._lock:
            already_cancelled = self._reason is not None
            if not already_cancelled:
                self._reason = reason
            process = self._process
        if process is not None and not already_cancelled:
            process.cancel(reason)
        return not already_cancelled

    @property
    def is_cancelled(self) -> bool:
        return self.reason is not None

    @property
    def cancelled(self) -> bool:
        """Alias for compatibility with existing MTS code."""
        return self.is_cancelled

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

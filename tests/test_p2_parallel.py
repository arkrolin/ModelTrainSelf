"""P2-2 tests: process execution + cancellation primitives.

The parallel search loop these tests originally covered is gone: scheduling now
lives in DispatcherLoop, which is driven over HTTP and covered by the E2E tests.
What remains here is the control-plane infrastructure DispatcherLoop builds on:
  - LocalProcess shells out and reports returncode / stdout / timeout.
  - TaskCancellation behaves as a one-shot, idempotent stop flag.
"""

from __future__ import annotations

import sys
from pathlib import Path

from mts.dispatcher.runtime.cancellation import TaskCancellation
from mts.dispatcher.runtime.local_process import LocalProcess


def test_local_process_captures_stdout(tmp_path: Path):
    process = LocalProcess(
        [sys.executable, "-c", "print('hello from worker')"],
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin"},
    )
    process.start()
    result = process.communicate(None)
    assert result.ok, f"returncode={result.returncode} stderr={result.stderr}"
    assert "hello from worker" in result.stdout
    assert result.timed_out is False


def test_local_process_reports_failure(tmp_path: Path):
    process = LocalProcess(
        [sys.executable, "-c", "import sys; sys.exit(3)"],
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin"},
    )
    process.start()
    result = process.communicate(None)
    assert result.ok is False
    assert result.returncode == 3


def test_local_process_times_out(tmp_path: Path):
    process = LocalProcess(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin"},
        timeout_seconds=1,
    )
    process.start()
    result = process.communicate(None)
    assert result.timed_out is True
    assert result.ok is False


def test_cancellation_signal():
    c = TaskCancellation()
    assert c.cancelled is False
    assert c.cancel("stopped") is True
    assert c.cancelled is True
    assert c.reason == "stopped"
    # Second cancel is a no-op (idempotent).
    assert c.cancel("again") is False

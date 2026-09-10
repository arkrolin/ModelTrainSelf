from __future__ import annotations

from mts.dispatcher.workers.adapters.claudecode import ClaudeCodeDriver
from mts.dispatcher.workers.adapters.llm import LLMDriver
from mts.dispatcher.workers.adapters.mock import MockDriver
from mts.dispatcher.workers.base import WorkerDriver


_CLAUDE = ClaudeCodeDriver()
_LLM = LLMDriver()
_MOCK = MockDriver()

DRIVERS: dict[str, WorkerDriver] = {
    "claudecode": _CLAUDE,
    "llm": _LLM,
    "mock": _MOCK,
}


def get_driver(name: str, execution: str = "local") -> WorkerDriver:
    """Get worker driver by name.

    MTS only supports local execution, so execution parameter is ignored
    for compatibility with Cairn's interface.
    """
    return DRIVERS[name]


def list_drivers() -> list[str]:
    """Return list of available driver names."""
    return list(DRIVERS.keys())

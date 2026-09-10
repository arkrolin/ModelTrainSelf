"""Entry point for `mts dispatch`.

Resolves a :class:`~mts.dispatcher.config.DispatchConfig` path and hands it to
:class:`~mts.dispatcher.scheduler.loop.DispatcherLoop`. All scheduling policy
lives in the loop; this module only decides *which* config file to use and how
to surface startup failures to the CLI.
"""

from __future__ import annotations

from pathlib import Path

from mts.dispatcher.logging import configure_logging
from mts.dispatcher.scheduler.loop import DispatcherLoop

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / "config" / "dispatch.yaml"
FALLBACK_CONFIG = REPO_ROOT / "config" / "dispatch_mock.yaml"


def resolve_config_path(config_path: str | Path | None) -> Path:
    """Pick the dispatcher config: explicit path, else dispatch.yaml, else mock."""
    if config_path:
        path = Path(config_path).expanduser()
        if not path.is_file():
            raise RuntimeError(f"dispatcher config not found: {path}")
        return path
    for candidate in (Path.cwd() / "config" / "dispatch.yaml", DEFAULT_CONFIG, FALLBACK_CONFIG):
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "no dispatcher config found: pass --config, or add config/dispatch.yaml "
        f"(see {FALLBACK_CONFIG.name} for the mock profile)"
    )


def run_dispatch(
    config_path: str | Path | None = None,
    once: bool = False,
    startup_healthcheck_only: bool = False,
    log_level: str = "INFO",
) -> None:
    """Run the dispatcher control loop."""
    configure_logging(log_level, bare=startup_healthcheck_only)
    path = resolve_config_path(config_path)
    loop = DispatcherLoop(path)
    if startup_healthcheck_only:
        loop.run_startup_healthchecks_only()
        return
    loop.run(once=once)

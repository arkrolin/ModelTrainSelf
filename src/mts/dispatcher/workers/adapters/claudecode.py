from __future__ import annotations

import os

from mts.dispatcher.config import WorkerConfig
from mts.dispatcher.workers.base import DriverResult, SeedSessionDriver
from mts.dispatcher.workers.health import HealthResult, http_ping, proxies_from_env


ANTHROPIC_VERSION = "2023-06-01"


class ClaudeCodeDriver(SeedSessionDriver):
    type_name = "claudecode"

    def local_binary(self) -> str | None:
        return "claude"

    def process_env(self, worker: WorkerConfig) -> dict[str, str]:
        """Let `--dangerously-skip-permissions` through when we are running as root.

        The CLI refuses that flag under root/sudo and exits 1 before doing any work,
        which surfaces as every task failing with code=1 in a few hundred ms. Both
        argv builders below pass the flag unconditionally, so on a root host — the
        usual case inside a container — no worker of this type can run at all.
        IS_SANDBOX=1 is the CLI's own escape hatch for that check.
        """
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            return {"IS_SANDBOX": "1"}
        return {}

    def check_health(self, worker: WorkerConfig, *, timeout: float) -> HealthResult:
        env = worker.env
        return http_ping(
            f"{env['ANTHROPIC_BASE_URL']}/v1/messages",
            headers={
                "Authorization": f"Bearer {env['ANTHROPIC_AUTH_TOKEN']}",
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            json_body={
                "model": env["ANTHROPIC_MODEL"],
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "ping"}],
            },
            timeout=timeout,
            proxies=proxies_from_env(env),
        )

    def describe_health(self, worker: WorkerConfig) -> str:
        return f"POST {worker.env['ANTHROPIC_BASE_URL']}/v1/messages (model={worker.env['ANTHROPIC_MODEL']})"

    def build_execute(self, worker: WorkerConfig, prompt: str, session: str | None) -> DriverResult:
        assert session is not None
        return DriverResult(
            argv=[
                "claude",
                "--session-id",
                session,
                "--dangerously-skip-permissions",
                "-p",
                "--",
                prompt,
            ],
            session=session,
        )

    def build_conclude(self, worker: WorkerConfig, prompt: str, session: str) -> list[str]:
        return [
            "claude",
            "-r",
            session,
            "--dangerously-skip-permissions",
            "-p",
            "--",
            prompt,
        ]

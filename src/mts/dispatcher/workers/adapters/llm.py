from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from mts.dispatcher.config import WorkerConfig
from mts.dispatcher.workers.base import DriverResult, SeedSessionDriver
from mts.dispatcher.workers.health import HealthResult, http_ping, proxies_from_env


class LLMDriver(SeedSessionDriver):
    type_name = "llm"

    def supports_conclude(self) -> bool:
        return False

    def local_binary(self) -> str | None:
        return sys.executable

    def check_health(self, worker: WorkerConfig, *, timeout: float) -> HealthResult:
        env = worker.env
        api_format = env.get("LLM_API_FORMAT", "openai").lower()

        if api_format == "anthropic":
            return http_ping(
                f"{env['LLM_BASE_URL']}/v1/messages",
                headers={
                    "Authorization": f"Bearer {env['LLM_API_KEY']}",
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json_body={
                    "model": env["LLM_MODEL"],
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "ping"}],
                },
                timeout=timeout,
                proxies=proxies_from_env(env),
            )
        else:
            return http_ping(
                f"{env['LLM_BASE_URL']}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {env['LLM_API_KEY']}",
                    "content-type": "application/json",
                },
                json_body={
                    "model": env["LLM_MODEL"],
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "ping"}],
                },
                timeout=timeout,
                proxies=proxies_from_env(env),
            )

    def describe_health(self, worker: WorkerConfig) -> str:
        api_format = worker.env.get("LLM_API_FORMAT", "openai")
        return f"POST {worker.env['LLM_BASE_URL']}/v1/... (model={worker.env['LLM_MODEL']}, format={api_format})"

    def build_execute(self, worker: WorkerConfig, prompt: str, session: str | None) -> DriverResult:
        assert session is not None
        # Write prompt to a temp file to avoid command-line length limits
        prompt_file = Path(tempfile.gettempdir()) / f"mts_llm_prompt_{session}.txt"
        prompt_file.write_text(prompt, encoding="utf-8")

        return DriverResult(
            argv=[
                sys.executable,
                "-m",
                "mts.dispatcher.workers.adapters.llm_call",
                "--prompt-file",
                str(prompt_file),
                "--session",
                session,
            ],
            session=session,
        )

    def build_conclude(self, worker: WorkerConfig, prompt: str, session: str) -> list[str]:
        raise NotImplementedError("llm driver does not support conclude (no session state)")

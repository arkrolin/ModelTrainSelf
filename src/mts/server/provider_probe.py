"""Provider connectivity probe: one cheap round-trip against a configured endpoint.

Backs the WebUI's "test connection" button. Sends a 1-token completion so the
user finds out whether the base_url/key/model triple actually works *before*
launching a search that would otherwise fail worker-by-worker in the background.

Kept separate from the dispatcher's health module on purpose: that one probes a
worker roster during a run, this one probes a not-yet-saved form payload.
"""

from __future__ import annotations

import time
from typing import Any

import requests

ANTHROPIC_VERSION = "2023-06-01"
PROBE_TIMEOUT = 30.0
PROBE_PROMPT = "ping"


def probe_provider(
    *,
    kind: str,
    base_url: str,
    api_key: str,
    model: str,
    headers: dict[str, str] | None = None,
    timeout: float = PROBE_TIMEOUT,
) -> dict[str, Any]:
    """Return {ok, latency_ms, model, reply, error}. Never raises."""
    if kind == "claudecode":
        return _probe_claudecode(base_url=base_url, api_key=api_key, model=model, timeout=timeout)
    if not base_url:
        return {"ok": False, "error": "base_url 不能为空"}
    if not model:
        return {"ok": False, "error": "model 不能为空"}

    base = base_url.rstrip("/")
    started = time.monotonic()
    try:
        if kind == "anthropic":
            url = f"{base}/v1/messages"
            payload = {
                "model": model,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": PROBE_PROMPT}],
            }
            req_headers = {
                "content-type": "application/json",
                "anthropic-version": ANTHROPIC_VERSION,
                "x-api-key": api_key,
                "Authorization": f"Bearer {api_key}",
                **(headers or {}),
            }
        else:
            url = f"{base}/v1/chat/completions"
            payload = {
                "model": model,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": PROBE_PROMPT}],
            }
            req_headers = {
                "content-type": "application/json",
                "Authorization": f"Bearer {api_key}",
                **(headers or {}),
            }

        response = requests.post(url, headers=req_headers, json=payload, timeout=timeout)
        latency_ms = (time.monotonic() - started) * 1000.0

        if response.status_code >= 400:
            return {
                "ok": False,
                "latency_ms": latency_ms,
                "error": f"HTTP {response.status_code}: {_truncate(response.text)}",
            }

        data = response.json()
        reply = _extract_reply(kind, data)
        return {
            "ok": True,
            "latency_ms": latency_ms,
            "model": data.get("model") or model,
            "reply": _truncate(reply, 200),
        }
    except requests.Timeout:
        return {"ok": False, "error": f"请求超时（>{timeout:.0f}s）"}
    except requests.RequestException as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    except (ValueError, KeyError) as exc:
        return {"ok": False, "error": f"响应解析失败: {exc}"}


def _probe_claudecode(*, base_url: str, api_key: str, model: str, timeout: float) -> dict[str, Any]:
    """claudecode drives the local `claude` CLI, so probe the CLI's own endpoint."""
    if not base_url:
        return {
            "ok": False,
            "error": "claudecode 类型需要 base_url（通常是 https://api.anthropic.com）",
        }
    return probe_provider(
        kind="anthropic",
        base_url=base_url,
        api_key=api_key,
        model=model or "claude-opus-5",
        timeout=timeout,
    )


def _extract_reply(kind: str, data: dict[str, Any]) -> str:
    if kind == "anthropic":
        blocks = data.get("content") or []
        if isinstance(blocks, list):
            return "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
        return ""
    choices = data.get("choices") or []
    if choices and isinstance(choices, list):
        message = choices[0].get("message") or {}
        return message.get("content") or ""
    return ""


def _truncate(text: str | None, limit: int = 300) -> str:
    if not text:
        return ""
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "…"

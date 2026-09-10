from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM API caller for MTS dispatcher")
    parser.add_argument("--prompt-file", required=True, help="Path to file containing the prompt")
    parser.add_argument("--session", required=True, help="Session ID (for cleanup)")
    args = parser.parse_args()

    prompt_file = Path(args.prompt_file)
    if not prompt_file.exists():
        print(f"prompt file not found: {prompt_file}", file=sys.stderr)
        sys.exit(1)

    prompt = prompt_file.read_text(encoding="utf-8")

    # Clean up prompt file
    try:
        prompt_file.unlink()
    except Exception:
        pass

    # Read LLM configuration from environment. The WebUI's provider settings land
    # here via the worker env the dispatcher writes; a bare shell export still works.
    model = os.environ.get("LLM_MODEL")
    base_url = os.environ.get("LLM_BASE_URL")
    api_key = os.environ.get("LLM_API_KEY")
    api_format = os.environ.get("LLM_API_FORMAT", "openai").lower()
    options = _read_options()

    if not model or not base_url or not api_key:
        print("LLM_MODEL, LLM_BASE_URL, and LLM_API_KEY must be set", file=sys.stderr)
        sys.exit(1)

    # Build request based on API format
    try:
        if api_format == "anthropic":
            response = call_anthropic(base_url, api_key, model, prompt, **options)
        elif api_format == "openai":
            response = call_openai(base_url, api_key, model, prompt, **options)
        else:
            print(f"unsupported LLM_API_FORMAT: {api_format}", file=sys.stderr)
            sys.exit(1)

        # Print only the model's response text to stdout
        print(response, end="")
    except Exception as exc:
        print(f"LLM API call failed: {exc}", file=sys.stderr)
        sys.exit(1)


def _read_options() -> dict:
    """Optional tuning knobs the provider settings UI can set."""
    options: dict = {}
    raw_max_tokens = os.environ.get("LLM_MAX_TOKENS")
    if raw_max_tokens:
        try:
            options["max_tokens"] = int(raw_max_tokens)
        except ValueError:
            print(f"ignoring invalid LLM_MAX_TOKENS: {raw_max_tokens}", file=sys.stderr)

    raw_temperature = os.environ.get("LLM_TEMPERATURE")
    if raw_temperature:
        try:
            options["temperature"] = float(raw_temperature)
        except ValueError:
            print(f"ignoring invalid LLM_TEMPERATURE: {raw_temperature}", file=sys.stderr)

    raw_headers = os.environ.get("LLM_EXTRA_HEADERS")
    if raw_headers:
        try:
            parsed = json.loads(raw_headers)
            if isinstance(parsed, dict):
                options["extra_headers"] = {str(k): str(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            print("ignoring invalid LLM_EXTRA_HEADERS (not JSON)", file=sys.stderr)

    return options


def call_anthropic(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int = 4096,
    temperature: float | None = None,
    extra_headers: dict | None = None,
) -> str:
    url = f"{base_url}/v1/messages"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        **(extra_headers or {}),
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if temperature is not None:
        payload["temperature"] = temperature

    response = requests.post(url, headers=headers, json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()

    # Extract text from Anthropic response format
    content = data.get("content", [])
    if not content or not isinstance(content, list):
        raise ValueError("unexpected Anthropic response format")

    text_blocks = [block.get("text", "") for block in content if block.get("type") == "text"]
    return "".join(text_blocks)


def call_openai(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int = 4096,
    temperature: float | None = None,
    extra_headers: dict | None = None,
) -> str:
    url = f"{base_url}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json",
        **(extra_headers or {}),
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature

    response = requests.post(url, headers=headers, json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()

    # Extract text from OpenAI response format
    choices = data.get("choices", [])
    if not choices:
        raise ValueError("unexpected OpenAI response format")

    message = choices[0].get("message", {})
    content = message.get("content", "")
    if not content:
        # Reasoning models (o1, DeepSeek-R1, QwQ) can return the answer in
        # reasoning_content with content empty; some gateways use "text".
        content = message.get("reasoning_content") or message.get("text") or ""
    return content


if __name__ == "__main__":
    main()

"""Minimal HTTP client for the board API (used by CLI project commands)."""

from __future__ import annotations

from typing import Any

import requests


def _post(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _get(url: str) -> Any:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def create_project_from_file(file_path: str, server: str) -> dict[str, Any]:
    import yaml

    raw = yaml.safe_load(open(file_path, encoding="utf-8").read()) or {}
    return _post(f"{server}/api/projects", raw)


def list_projects(server: str) -> Any:
    return _get(f"{server}/api/projects")


def add_hint(server: str, project_id: str, content: str, creator: str) -> dict[str, Any]:
    return _post(f"{server}/api/projects/{project_id}/hints",
                 {"content": content, "creator": creator})


def print_response(payload: Any) -> None:
    import json

    print(json.dumps(payload, indent=2, ensure_ascii=False))

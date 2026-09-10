"""Extract a JSON object from free-form LLM output (ported from Cairn).

Real LLM workers emit prose around a JSON payload (fenced code blocks, stray
prefix text, trailing chatter). This module reliably recovers the object. It is
backend-agnostic and shared by every adapter.
"""

from __future__ import annotations

import json
import re
from typing import Any

FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any]:
    """Return the first JSON object found in `text`, raising if none.

    Tries, in order: (1) the whole stripped text, (2) each fenced ```json```
    block, (3) each substring starting at a `{` (using a JSONDecoder so a
    trailing prose after the object is tolerated).
    """
    decoder = json.JSONDecoder()
    seen: set[str] = set()

    for candidate in _candidate_segments(text):
        segment = candidate.strip().lstrip("\ufeff")
        if not segment or segment in seen:
            continue
        seen.add(segment)
        try:
            parsed = json.loads(segment)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(parsed, dict):
                return parsed

        for start in _object_start_positions(segment):
            try:
                parsed, _ = decoder.raw_decode(segment[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

    raise ValueError("no JSON object found in output")


def _candidate_segments(text: str) -> list[str]:
    segments = [text.strip()]
    segments.extend(m.group(1).strip() for m in FENCED_BLOCK_RE.finditer(text))
    return segments


def _object_start_positions(text: str) -> list[int]:
    return [i for i, ch in enumerate(text) if ch == "{"]

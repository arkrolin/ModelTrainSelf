"""Output parser + worker driver registry.

The decision contract these tests used to cover is gone: a worker no longer
returns a "move one spec axis" Decision, it returns a fact payload validated by
contracts.validate_*_payload (covered by the contract tests). What survives here
is the free-form JSON extraction every driver depends on, plus driver lookup.
"""

from __future__ import annotations

import pytest

from mts.dispatcher.output_parser import extract_json_object
from mts.dispatcher.workers.base import WorkerDriver
from mts.dispatcher.workers.registry import DRIVERS, get_driver, list_drivers


# ---------------------------------------------------------------------------
# output_parser
# ---------------------------------------------------------------------------

def test_extract_plain_json():
    assert extract_json_object('{"axis": "optim.lr", "value": 0.001}') == {
        "axis": "optim.lr", "value": 0.001,
    }


def test_extract_fenced_json():
    text = "Here is my plan:\n```json\n{\"axis\": \"arch.norm_position\", \"value\": \"post\"}\n```\nDone."
    assert extract_json_object(text)["axis"] == "arch.norm_position"


def test_extract_json_with_trailing_prose():
    text = 'result: {"axis": "arch.n_layer", "value": 8} thanks!'
    assert extract_json_object(text)["value"] == 8


def test_extract_json_bom():
    assert extract_json_object('﻿{"axis": "optim.lr", "value": 0.001}')["axis"] == "optim.lr"


def test_extract_no_json_raises():
    with pytest.raises(ValueError):
        extract_json_object("no json here at all")


# ---------------------------------------------------------------------------
# worker driver registry
# ---------------------------------------------------------------------------

def test_registry_lists_the_three_drivers():
    drivers = list_drivers()
    assert set(drivers) == {"claudecode", "llm", "mock"}


def test_get_driver_returns_worker_driver():
    for name in list_drivers():
        driver = get_driver(name, "local")
        assert isinstance(driver, WorkerDriver)
        assert driver.type_name == name


def test_get_driver_ignores_execution_mode():
    # MTS is local-only; the execution argument exists for Cairn parity.
    assert get_driver("mock", "local") is get_driver("mock", "container")


def test_unknown_driver_raises():
    with pytest.raises(KeyError):
        get_driver("nonexistent")


def test_drivers_are_singletons():
    assert get_driver("mock") is DRIVERS["mock"]

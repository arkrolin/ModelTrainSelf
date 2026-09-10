"""Unit tests for the training state-space spec and diagnostics."""

from __future__ import annotations

import pytest

from mts.trainer.spec import TrialSpec, format_diff


def test_derive_changes_one_axis():
    base = TrialSpec()
    child = base.derive(**{"arch.norm_position": "post"})
    assert child.arch.norm_position == "post"
    assert base.arch.norm_position == "pre"  # parent unchanged


def test_derive_rejects_unknown_path():
    base = TrialSpec()
    with pytest.raises(KeyError):
        base.derive(**{"arch.nope": 1})


def test_diff_direction_is_parent_to_child():
    base = TrialSpec()
    child = base.derive(**{"arch.n_layer": 8})
    diff = child.diff(base)
    assert diff["arch.n_layer"] == {"from": 4, "to": 8}


def test_diff_excludes_identity_fields():
    base = TrialSpec()
    child = base.derive(**{"name": "renamed", "seed": 99})
    assert "name" not in child.diff(base)
    assert "seed" not in child.diff(base)


def test_format_diff_readable():
    base = TrialSpec()
    child = base.derive(**{"arch.norm_position": "post", "optim.lr": 1e-3})
    text = format_diff(child.diff(base))
    assert "pre->post" in text
    assert "0.003" in text or "3e-3" in text


def test_head_divisibility_validation():
    with pytest.raises(ValueError):
        TrialSpec(arch={"n_head": 3, "d_model": 256})


def test_param_count_estimate_scales_with_depth():
    small = TrialSpec(arch={"n_layer": 2})
    big = TrialSpec(arch={"n_layer": 8})
    assert big.param_count_estimate() > small.param_count_estimate()

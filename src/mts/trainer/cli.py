"""`mts train` — run one experiment.

This is the instrument an agent calls during Explore. It writes the standard
artifact set into `--out-dir` and prints the verdict, so an agent can either read
the files or just read stdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from mts.trainer.runner import run_trial
from mts.trainer.spec import TrialSpec


@click.command("train")
@click.option("--spec", "spec_path", required=True, type=click.Path(exists=True),
              help="TrialSpec file (.json / .yaml) describing the point to evaluate.")
@click.option("--out-dir", "out_dir", required=True, type=click.Path(file_okay=False),
              help="Directory for artifacts (created if missing).")
@click.option("--parent-spec", "parent_spec_path", default=None, type=click.Path(exists=True),
              help="Parent TrialSpec, used to compute and record the diff.")
@click.option("--trial-id", default=None, help="Trial id recorded in summary.json (defaults to dir name).")
@click.option("--backend", default=None, type=click.Choice(["torch", "surrogate"]),
              help="Override spec.backend.")
@click.option("--device", default=None, help="Override spec.device (auto / cpu / cuda:0).")
@click.option("--reference-loss", default=None, type=float,
              help="Best loss reached so far in the project; enables the underfitting verdict.")
@click.option("--quiet", is_flag=True, help="Suppress per-step progress output.")
def train(spec_path: str, out_dir: str, parent_spec_path: str | None, trial_id: str | None,
          backend: str | None, device: str | None, reference_loss: float | None,
          quiet: bool) -> None:
    """Run one training experiment and write artifacts to OUT_DIR."""
    spec = TrialSpec.from_file(spec_path)
    parent = TrialSpec.from_file(parent_spec_path) if parent_spec_path else None
    out = Path(out_dir)

    def progress(message: str) -> None:
        if not quiet:
            click.echo(message, err=True)

    result = run_trial(
        spec,
        out,
        trial_id=trial_id,
        parent_spec=parent,
        backend=backend,
        device=device,
        reference_loss=reference_loss,
        progress=progress,
    )

    payload = {
        "trial_id": result.trial_id,
        "out_dir": str(out.resolve()),
        "status": result.status,
        "verdict": result.verdict,
        "diff": result.diff_text,
        "final": result.final,
        "best": result.best,
        "signals": result.signals,
        "duration_sec": result.duration_sec,
    }
    click.echo(json.dumps(payload, indent=2))
    if result.status in ("diverged", "error"):
        sys.exit(2)


__all__ = ["train"]

"""MolelTrainSelf command line entry point.

Commands are registered lazily so that a missing optional dependency (e.g. torch)
never breaks unrelated commands.
"""

from __future__ import annotations

from pathlib import Path

import click

from mts import __version__
from mts.trainer.cli import train


@click.group()
@click.version_option(__version__, prog_name="mts")
def main() -> None:
    """MolelTrainSelf — multi-agent search over the model-training state space."""


main.add_command(train)


@main.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option("--db", default=None, help="SQLite path (default: data/mts.db under the project root).")
@click.option("--runs", "runs_dir", default=None, help="Trial artifact root (default: runs/).")
@click.option("--reload", is_flag=True, help="Reload on code changes (development).")
def serve(host: str, port: int, db: str | None, runs_dir: str | None, reload: bool) -> None:
    """Start the board server (FastAPI + SQLite) and serve the WebUI."""
    import os

    from mts.server.app import run_server

    # The WebUI's "start dispatch" button runs a DispatcherLoop in-process, and that
    # loop reaches the board over HTTP — so it needs the URL we are about to bind.
    connect_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    os.environ.setdefault("MTS_BOARD_URL", f"http://{connect_host}:{port}")

    run_server(host=host, port=port, db_path=db, runs_dir=runs_dir, reload=reload)


@main.command("dispatch")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Dispatcher config path (default: config/dispatch.yaml, else config/dispatch_mock.yaml).",
)
@click.option("--once", is_flag=True, help="Run one scheduling iteration and exit.")
@click.option(
    "--startup-healthcheck-only",
    "--healthcheck",
    "startup_healthcheck_only",
    is_flag=True,
    help="Run startup worker healthchecks and exit.",
)
@click.option("--log-level", default="INFO", show_default=True, help="Log level")
def dispatch(config_path: Path | None, once: bool, startup_healthcheck_only: bool, log_level: str) -> None:
    """Run the MTS dispatcher."""
    from mts.dispatcher.runtime.entry import run_dispatch

    try:
        run_dispatch(
            config_path=config_path,
            once=once,
            startup_healthcheck_only=startup_healthcheck_only,
            log_level=log_level,
        )
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc


@main.group("project")
def project() -> None:
    """Create and inspect search projects."""


@project.command("create")
@click.option("--file", "file_path", required=True, type=click.Path(exists=True),
              help="Project definition (.yaml), see examples/project_demo.yaml")
@click.option("--server", default="http://127.0.0.1:8000", show_default=True)
def project_create(file_path: str, server: str) -> None:
    """Create a project from a YAML definition."""
    from mts.server.client import create_project_from_file, print_response

    print_response(create_project_from_file(file_path, server))


@project.command("list")
@click.option("--server", default="http://127.0.0.1:8000", show_default=True)
def project_list(server: str) -> None:
    """List projects."""
    from mts.server.client import list_projects, print_response

    print_response(list_projects(server))


@project.command("hint")
@click.argument("project_id")
@click.option("--content", required=True, help="What you want the agents to consider next.")
@click.option("--creator", default="human", show_default=True)
@click.option("--server", default="http://127.0.0.1:8000", show_default=True)
def project_hint(project_id: str, content: str, creator: str, server: str) -> None:
    """Inject a human hint into the board."""
    from mts.server.client import add_hint, print_response

    print_response(add_hint(server, project_id, content, creator))


@main.command("report")
@click.argument("project_id")
@click.option("--db", default=None, help="SQLite path (default: data/board.db under cwd).")
@click.option("--root", default=None, help="Board root (default: cwd).")
@click.option("--out", "out_path", default=None, help="Write to a file instead of stdout.")
def report(project_id: str, db: str | None, root: str | None, out_path: str | None) -> None:
    """Generate a Markdown experiment report for a project."""
    from mts.reporting import build_report
    from mts.server.db import Database
    from mts.server.services import Service

    base = Path(root) if root else Path.cwd()
    svc = Service(Database(db or str(base / "data" / "board.db")), base)
    svc.db.configure()
    try:
        markdown = build_report(svc, project_id)
    except KeyError:
        raise click.ClickException(f"project not found: {project_id}")
    if out_path:
        Path(out_path).write_text(markdown, encoding="utf-8")
        click.echo(f"wrote {out_path}")
    else:
        click.echo(markdown)


@main.group("knowledge")
def knowledge() -> None:
    """Inspect the shared Markdown knowledge space."""


@knowledge.command("list")
@click.option("--tag", default=None)
@click.option("--root", default=None, help="Knowledge root (default: memory/).")
def knowledge_list(tag: str | None, root: str | None) -> None:
    """List knowledge lessons."""
    from mts.knowledge.cli import list_lessons

    list_lessons(tag=tag, root=root)


@knowledge.command("show")
@click.argument("slug")
@click.option("--root", default=None)
def knowledge_show(slug: str, root: str | None) -> None:
    """Show one lesson."""
    from mts.knowledge.cli import show_lesson

    show_lesson(slug, root=root)


@knowledge.command("reindex")
@click.option("--root", default=None)
def knowledge_reindex(root: str | None) -> None:
    """Rebuild memory/INDEX.md."""
    from mts.knowledge.cli import reindex

    reindex(root=root)


@main.command("demo")
@click.option("--port", default=8000, show_default=True, type=int)
@click.option("--trials", default=12, show_default=True, type=int, help="Trial budget for the project.")
@click.option("--keep", is_flag=True, help="Keep server running after the demo finishes.")
@click.option("--open-browser/--no-browser", default=True, show_default=True)
def demo(port: int, trials: int, keep: bool, open_browser: bool) -> None:
    """Seed a demo project on the board and serve the WebUI.

    Explore it with `mts dispatch --config config/dispatch_mock.yaml`.
    """
    from mts.demo import run_demo

    run_demo(port=port, trials=trials, keep=keep, open_browser=open_browser)


if __name__ == "__main__":  # pragma: no cover
    main()

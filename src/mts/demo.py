"""Demo: seed a board project from the baseline spec, then serve the WebUI.

`mts demo` gets you a board with one project on it and no credentials required.
Exploration itself is the dispatcher's job — run it in a second shell:

    mts dispatch --config config/dispatch_mock.yaml

or press "start dispatch" in the WebUI, which runs the same DispatcherLoop.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from mts.server.app import create_app
from mts.server.db import Database
from mts.server.models import ProjectCreate
from mts.server.services import Service

PACKAGE_ROOT = Path(__file__).parent          # src/mts
REPO_ROOT = PACKAGE_ROOT.parent.parent        # src/mts -> src -> repo root
BASELINE_SPEC = REPO_ROOT / "examples" / "specs" / "baseline.yaml"


def run_demo(port: int, trials: int, keep: bool, open_browser: bool) -> None:
    root = Path.cwd()
    # The WebUI's "start search" button runs a DispatcherLoop in-process, and that
    # loop reaches the board over HTTP. Without this it falls back to the hardcoded
    # :8000 and every request fails with connection refused on any other port.
    os.environ.setdefault("MTS_BOARD_URL", f"http://127.0.0.1:{port}")
    db = Database(root / "data" / "board.db")
    db.configure()
    service = Service(db, root)

    # Announce which baseline the Origin text points at — a silent fallback would
    # describe a different starting point than baseline.yaml, with nothing in the
    # log to say so. Resources live in the Origin text, never in Python.
    if BASELINE_SPEC.exists():
        origin_spec = str(BASELINE_SPEC)
        print(f"[demo] baseline spec: {BASELINE_SPEC}")
    else:
        origin_spec = "(no baseline spec on disk; agents should define one)"
        print(f"[demo] baseline spec not found at {BASELINE_SPEC}")

    # `mts demo` 复用同名项目，不每次新建一个：demo 用的是 cwd 下那个持久的
    # data/board.db，反复跑（quicktest.sh 就是这么用的）会堆出一串一模一样的
    # active 项目，CLI `mts dispatch` 会在它们之间轮流派发。
    title = "字符级语言模型训练状态空间搜索"
    existing = next((p for p in service.list_projects() if p["title"] == title), None)
    if existing is not None:
        project = existing
        if project["budget_max_trials"] != trials:
            project = service.update_project(project["id"], {"budget_max_trials": trials})
        print(f"[demo] reusing project={project['id']} title={project['title']}")
        print(f"[demo] budget: {trials} trials")
        print("[demo] explore it with: mts dispatch --config config/dispatch_mock.yaml")
        _serve(root, port, open_browser)
        return

    project = service.create_project(ProjectCreate(
        title=title,
        origin=(
            "synthetic_lm 字符级语言模型，基线 pre-norm 4 层 / lr 3e-3 / 300 步。"
            f"基线配置文件：{origin_spec}。Agent 自己编写 train.py，运行实验后将指标写入 metrics.json，"
            "然后通过 POST /intents/{iid}/conclude 上报结果到 facts 表。"
        ),
        goal="在保持 verdict=healthy 的前提下降低 val_loss；同时避免发散/梯度消失/激活死亡",
        goal_metric="val_loss",
        goal_target=1.70,
        goal_direction="minimize",
        budget_max_trials=trials,
        hints=["优先从稳定性问题入手（发散/不稳定），再追求更低 loss"],
    ))
    print(f"[demo] project={project['id']} title={project['title']}")
    print(f"[demo] budget: {trials} trials")
    print("[demo] explore it with: mts dispatch --config config/dispatch_mock.yaml")

    _serve(root, port, open_browser)


def _serve(root: Path, port: int, open_browser: bool) -> None:
    """Serve the WebUI (blocks; Ctrl-C to stop)."""
    app = create_app(root=root, db_path=root / "data" / "board.db")
    print(f"\n[demo] board ready at http://127.0.0.1:{port}")
    if open_browser:
        _open_browser(port)
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=port, reload=False)


def _open_browser(port: int) -> None:
    try:
        import webbrowser

        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    except Exception:
        pass

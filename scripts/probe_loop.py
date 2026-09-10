#!/usr/bin/env python3
"""探测：对已经在跑的 board 执行一轮调度，验证闭环是否真的能跑完。

需要先起 server（`mts serve`），因为 DispatcherLoop 只通过 HTTP 和 board 说话。

用法: python scripts/probe_loop.py [config] [server]
      config 默认 config/dispatch_mock.yaml，server 取自 config 里的 server 字段
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from mts.dispatcher.logging import configure_logging
from mts.dispatcher.protocol.client import MTSClient
from mts.dispatcher.scheduler.loop import DispatcherLoop

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "dispatch_mock.yaml"


def main() -> int:
    configure_logging("INFO")
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG
    loop = DispatcherLoop(config_path)
    server = sys.argv[2] if len(sys.argv) > 2 else loop.config.server
    print(f"[probe] config={config_path} server={server}", flush=True)

    client = MTSClient(server)
    try:
        projects = client.list_projects()
    except Exception as exc:  # noqa: BLE001 - a probe reports, it does not raise
        print(f"[probe] cannot reach board at {server}: {exc}", flush=True)
        return 1
    print(f"[probe] projects={len(projects)}", flush=True)
    for summary in projects:
        print(
            f"  - {summary.id} status={summary.status} facts={summary.fact_count} "
            f"intents={summary.intent_count} best_metric={summary.best_metric}",
            flush=True,
        )

    t0 = time.time()
    loop.run(once=True)
    print(f"[probe] one dispatch round elapsed={time.time() - t0:.1f}s", flush=True)

    for summary in client.list_projects():
        print(
            f"[probe] after {summary.id} facts={summary.fact_count} "
            f"working={summary.working_intent_count} unclaimed={summary.unclaimed_intent_count}",
            flush=True,
        )
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

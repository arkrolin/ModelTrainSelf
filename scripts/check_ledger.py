#!/usr/bin/env python3
"""检查台账字段是否真的写入 SQLite（区分"没写入" vs "读取字段名不对"）。

用法: python scripts/check_ledger.py <board.db>
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


def main() -> int:
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("_probe_run/data/board.db")
    if not db_path.exists():
        print(f"no db at {db_path}")
        return 1

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    cols = [r["name"] for r in conn.execute("PRAGMA table_info(trials)")]
    print(f"[ledger] trials columns ({len(cols)}): {cols}\n")

    rows = conn.execute(
        "SELECT id, status, verdict, best_val_loss, final_val_loss, "
        "best_val_acc, final_val_acc FROM trials ORDER BY created_at"
    ).fetchall()
    print(f"[ledger] {len(rows)} rows in trials:")
    for r in rows:
        print(f"  {dict(r)}")

    # get_trial 返回的键，用来确认前端/脚本该读哪个名字
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from mts.server.db import Database
    from mts.server.services import Service

    svc = Service(Database(db_path), db_path.parent.parent)
    pid = conn.execute("SELECT id FROM projects LIMIT 1").fetchone()["id"]
    trials = svc.list_trials(pid)
    if trials:
        print(f"\n[ledger] get_trial() keys: {sorted(trials[0].keys())}")
        for t in trials:
            print(f"  {t.get('id')}: best_val_loss={t.get('best_val_loss')!r} "
                  f"val_loss={t.get('val_loss')!r} verdict={t.get('verdict')!r}")

    lb = svc.leaderboard(pid)
    print(f"\n[ledger] leaderboard rows: {len(lb)}")
    for r in lb[:5]:
        print(f"  {r.get('id')} best_val_loss={r.get('best_val_loss')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

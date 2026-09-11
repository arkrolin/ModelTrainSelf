"""Business logic for the board.

Everything the routers need, decoupled from HTTP so the dispatcher and tests can
call it directly. The server owns the fact graph (facts / intents / hints), the
trial ledger, and the knowledge index.
"""

from __future__ import annotations

import hashlib
import json
import re
import yaml
from pathlib import Path
from typing import Any

from mts.server.db import Database
from mts.trainer.runner import read_summary

_ID_FMT = {"fact": "f", "intent": "i", "hint": "h", "trial": "t"}


def _iso_now() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _format_export_timestamp(value: str | None) -> str | None:
    """Format timestamp for export (local time)."""
    if not value:
        return value
    try:
        import datetime as _dt
        dt = _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "note"


def spec_fingerprint(spec: dict[str, Any]) -> str:
    """Deterministic fingerprint of the search-relevant spec fields.

    Excludes `name`, `seed`, `notes` and `parent` — two trials that differ only
    in a random seed are still the *same point* in the state space, so they must
    collide and trigger a duplicate warning.
    """
    clone = {k: v for k, v in spec.items() if k not in {"name", "seed", "notes", "parent"}}
    canonical = json.dumps(clone, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class Service:
    def __init__(self, db: Database, root: Path):
        self.db = db
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.memory_dir = self.root / "memory" / "lessons"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================================
    # Lease expiry (Cairn protocol)
    # =====================================================================

    def expire_workers(self, pid: str) -> None:
        """Clear intent leases whose heartbeat is older than intent_timeout."""
        settings = self.db.get_settings()
        timeout = settings["intent_timeout"]
        import datetime as _dt
        cutoff = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(seconds=timeout)).isoformat()
        self.db.execute(
            "UPDATE intents SET worker=NULL, last_heartbeat_at=NULL "
            "WHERE project_id=? AND to_fact_id IS NULL AND worker IS NOT NULL "
            "AND last_heartbeat_at IS NOT NULL AND last_heartbeat_at < ?",
            (pid, cutoff),
        )

    def expire_reason_leases(self, pid: str) -> None:
        """Clear reason lease whose heartbeat is older than reason_timeout."""
        settings = self.db.get_settings()
        timeout = settings["reason_timeout"]
        import datetime as _dt
        cutoff = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(seconds=timeout)).isoformat()
        self.db.execute(
            "UPDATE projects SET reason_worker=NULL, reason_trigger=NULL, "
            "reason_started_at=NULL, reason_last_heartbeat_at=NULL "
            "WHERE id=? AND reason_worker IS NOT NULL AND reason_last_heartbeat_at IS NOT NULL "
            "AND reason_last_heartbeat_at < ?",
            (pid, cutoff),
        )

    # =====================================================================
    # Projects
    # =====================================================================

    def create_project(self, payload: Any) -> dict[str, Any]:
        pid = f"proj_{self.db.next_global('project'):03d}"
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO projects(id, title, status, bootstrap_enabled, origin, goal, "
                "goal_metric, goal_target, goal_direction, budget_max_trials, created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (pid, payload.title, "active", int(payload.bootstrap_enabled), payload.origin, payload.goal,
                 payload.goal_metric, payload.goal_target, payload.goal_direction,
                 payload.budget_max_trials, _iso_now()),
            )
        # Seed origin / goal facts and any hints.
        self.add_fact(pid, "origin", payload.origin, None, None)
        self.add_fact(pid, "goal", payload.goal, None, None)
        for hint in payload.hints:
            self.add_hint(pid, hint, "human")
        return self.project_summary(pid)

    def get_project(self, pid: str) -> dict[str, Any] | None:
        """Get project with detail (facts/intents/hints)."""
        self.expire_workers(pid)
        self.expire_reason_leases(pid)
        row = self.db.fetchone("SELECT * FROM projects WHERE id=?", (pid,))
        if row is None:
            return None
        row["bootstrap_enabled"] = bool(row["bootstrap_enabled"])
        return row

    def project_summary(self, pid: str) -> dict[str, Any] | None:
        """Get project summary (Cairn ProjectSummary shape + MTS extensions)."""
        self.expire_workers(pid)
        self.expire_reason_leases(pid)
        row = self.db.fetchone("SELECT * FROM projects WHERE id=?", (pid,))
        if row is None:
            return None

        fact_count = self.db.fetchone("SELECT COUNT(*) c FROM facts WHERE project_id=?", (pid,))["c"]
        intent_count = self.db.fetchone("SELECT COUNT(*) c FROM intents WHERE project_id=?", (pid,))["c"]
        hint_count = self.db.fetchone("SELECT COUNT(*) c FROM hints WHERE project_id=?", (pid,))["c"]
        # Experiments the agents actually completed. This drives the scheduler's
        # budget check (loop.py: trial_count >= budget_max_trials), so it MUST count
        # what the autonomous flow produces: one fact per concluded intent. Counting
        # the `trials` table instead left the budget permanently at 0 — agents never
        # register trials, they conclude intents — so searches never stopped.
        # origin/goal are seeded at project creation and are not experiments.
        trial_count = self.db.fetchone(
            "SELECT COUNT(*) c FROM facts WHERE project_id=? AND id NOT IN ('origin','goal')",
            (pid,),
        )["c"]

        working_intent_count = self.db.fetchone(
            "SELECT COUNT(*) c FROM intents WHERE project_id=? AND worker IS NOT NULL AND to_fact_id IS NULL",
            (pid,),
        )["c"]
        unclaimed_intent_count = self.db.fetchone(
            "SELECT COUNT(*) c FROM intents WHERE project_id=? AND worker IS NULL AND to_fact_id IS NULL",
            (pid,),
        )["c"]

        # best_metric: best value of the project's goal_metric across facts' metrics
        goal_metric = row["goal_metric"]
        goal_direction = row["goal_direction"]
        facts_with_metric = self.db.fetchall(
            "SELECT metrics FROM facts WHERE project_id=? AND metrics != '{}'", (pid,)
        )
        best_metric = None
        for f in facts_with_metric:
            metrics = self.db.json_loads(f["metrics"], {})
            if goal_metric in metrics:
                val = metrics[goal_metric]
                if best_metric is None:
                    best_metric = val
                elif goal_direction == "maximize":
                    best_metric = max(best_metric, val)
                else:
                    best_metric = min(best_metric, val)

        return {
            "id": row["id"],
            "title": row["title"],
            "status": row["status"],
            "bootstrap_enabled": bool(row["bootstrap_enabled"]),
            "origin": row["origin"],
            "goal": row["goal"],
            "goal_metric": row["goal_metric"],
            "goal_target": row["goal_target"],
            "goal_direction": row["goal_direction"],
            "budget_max_trials": row["budget_max_trials"],
            "created_at": row["created_at"],
            "reason": self._project_reason_from_row(row),
            "fact_count": fact_count,
            "intent_count": intent_count,
            "working_intent_count": working_intent_count,
            "unclaimed_intent_count": unclaimed_intent_count,
            "hint_count": hint_count,
            "trial_count": trial_count,
            "best_metric": best_metric,
        }

    def _project_reason_from_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        if row["reason_worker"] is None:
            return None
        return {
            "worker": row["reason_worker"],
            "trigger": row["reason_trigger"],
            "started_at": row["reason_started_at"],
            "last_heartbeat_at": row["reason_last_heartbeat_at"],
        }

    def list_projects(self) -> list[dict[str, Any]]:
        rows = self.db.fetchall("SELECT id FROM projects ORDER BY created_at")
        return [self.project_summary(r["id"]) for r in rows]

    def set_project_status(self, pid: str, status: str) -> dict[str, Any] | None:
        self.db.execute("UPDATE projects SET status=? WHERE id=?", (status, pid))
        return self.project_summary(pid)

    # Whitelist, not free-form: keeps a caller from writing status or lease columns
    # through the settings form.
    _PROJECT_EDITABLE = (
        "title", "origin", "goal", "goal_metric", "goal_target",
        "goal_direction", "budget_max_trials", "bootstrap_enabled",
    )

    def update_project(self, pid: str, fields: dict[str, Any]) -> dict[str, Any] | None:
        """Patch a project's own fields. Returns None when the project is missing.

        Only keys present in `fields` are written, so a partial form submit leaves
        untouched columns alone.
        """
        if self.db.fetchone("SELECT id FROM projects WHERE id=?", (pid,)) is None:
            return None

        updates = {k: v for k, v in fields.items() if k in self._PROJECT_EDITABLE}
        if not updates:
            return self.project_summary(pid)

        if "bootstrap_enabled" in updates:
            updates["bootstrap_enabled"] = 1 if updates["bootstrap_enabled"] else 0

        assignments = ", ".join(f"{col}=?" for col in updates)
        params = (*updates.values(), pid)
        self.db.execute(f"UPDATE projects SET {assignments} WHERE id=?", params)
        return self.project_summary(pid)

    def delete_project(self, pid: str) -> bool:
        """删除项目及其所有关联数据。

        删除顺序：
        1. trials (外键约束)
        2. intents (外键约束)
        3. facts (外键约束)
        4. hints (外键约束)
        5. observations (外键约束)
        6. projects (主表)
        """
        try:
            with self.db.connect() as conn:
                # 删除 trials
                conn.execute("DELETE FROM trials WHERE project_id=?", (pid,))
                # 删除 intents
                conn.execute("DELETE FROM intents WHERE project_id=?", (pid,))
                # 删除 facts
                conn.execute("DELETE FROM facts WHERE project_id=?", (pid,))
                # 删除 hints
                conn.execute("DELETE FROM hints WHERE project_id=?", (pid,))
                # 删除 observations (如果存在)
                conn.execute("DELETE FROM observations WHERE project_id=?", (pid,))
                # 删除项目本身
                conn.execute("DELETE FROM projects WHERE id=?", (pid,))
            return True
        except Exception as e:
            print(f"Failed to delete project {pid}: {e}")
            return False

    def project_status(self, pid: str) -> str | None:
        """Return the project's status, or None when the project does not exist.

        The lease endpoints call this to reject writes against a stopped or
        completed project. A worker that gets 403 here treats it as a hard
        failure and kills its attached process, which is how a human stopping
        a project cancels the agents already running on it.
        """
        row = self.db.fetchone("SELECT status FROM projects WHERE id=?", (pid,))
        return None if row is None else row["status"]

    # =====================================================================
    # Reason lease (Cairn protocol)
    # =====================================================================

    def claim_reason(self, pid: str, worker: str, trigger: str) -> str:
        """Claim the reason lease. Returns 'claimed' or 'conflict'."""
        self.expire_reason_leases(pid)
        with self.db.connect() as conn:
            proj = conn.execute("SELECT reason_worker FROM projects WHERE id=?", (pid,)).fetchone()
            if proj is None:
                return "conflict"
            if proj["reason_worker"] is not None and proj["reason_worker"] != worker:
                return "conflict"
            now = _iso_now()
            conn.execute(
                "UPDATE projects SET reason_worker=?, reason_trigger=?, "
                "reason_started_at=?, reason_last_heartbeat_at=? WHERE id=?",
                (worker, trigger, now, now, pid),
            )
        return "claimed"

    def heartbeat_reason(self, pid: str, worker: str) -> bool:
        """Heartbeat the reason lease. Returns False if not held by worker.

        Expiry runs first for the same reason as `heartbeat_intent`: a lapsed
        lease must report itself lost rather than quietly re-arming.
        """
        self.expire_reason_leases(pid)
        cur = self.db.execute(
            "UPDATE projects SET reason_last_heartbeat_at=? "
            "WHERE id=? AND reason_worker=?",
            (_iso_now(), pid, worker),
        )
        return cur > 0

    def release_reason(self, pid: str, worker: str) -> bool:
        """Release the reason lease. Returns False if not held by worker."""
        self.expire_reason_leases(pid)
        cur = self.db.execute(
            "UPDATE projects SET reason_worker=NULL, reason_trigger=NULL, "
            "reason_started_at=NULL, reason_last_heartbeat_at=NULL "
            "WHERE id=? AND reason_worker=?",
            (pid, worker),
        )
        return cur > 0

    # =====================================================================
    # Facts / Intents / Hints
    # =====================================================================

    def add_fact(self, pid: str, fact_id: str, description: str,
                 trial_id: str | None,
                 metrics: dict[str, float] | None,
                 artifacts: dict[str, Any] | None = None) -> dict[str, Any]:
        metrics_json = json.dumps(metrics or {})
        artifacts_json = json.dumps(artifacts) if artifacts else None
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO facts(id, project_id, description, metrics, trial_id, artifacts, created_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (fact_id, pid, description, metrics_json, trial_id, artifacts_json, _iso_now()),
            )
        return {"id": fact_id, "description": description, "metrics": metrics or {},
                "trial_id": trial_id, "artifacts": artifacts}

    def list_facts(self, pid: str) -> list[dict[str, Any]]:
        rows = self.db.fetchall(
            "SELECT id, description, metrics, trial_id, artifacts, created_at FROM facts "
            "WHERE project_id=? ORDER BY created_at", (pid,),
        )
        for row in rows:
            row["metrics"] = self.db.json_loads(row["metrics"], {})
            row["artifacts"] = self.db.json_loads(row["artifacts"], None)
        return rows

    def next_intent_id(self, pid: str) -> str:
        return f"{_ID_FMT['intent']}{self.db.next_scoped(pid, 'intent'):03d}"

    def create_intent(self, pid: str, payload: Any) -> dict[str, Any]:
        iid = self.next_intent_id(pid)
        now = _iso_now()
        claimed = payload.worker is not None
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO intents(id, project_id, description, creator, worker, "
                "last_heartbeat_at, created_at) VALUES(?,?,?,?,?,?,?)",
                (iid, pid, payload.description, payload.creator, payload.worker,
                 now if claimed else None, now),
            )
            for fact_id in payload.from_:
                conn.execute(
                    "INSERT OR IGNORE INTO intent_sources(intent_id, project_id, fact_id) "
                    "VALUES(?,?,?)", (iid, pid, fact_id),
                )
        return self.get_intent(pid, iid)

    def get_intent(self, pid: str, iid: str) -> dict[str, Any] | None:
        row = self.db.fetchone(
            "SELECT * FROM intents WHERE project_id=? AND id=?", (pid, iid)
        )
        if row is None:
            return None
        sources = self.db.fetchall(
            "SELECT fact_id FROM intent_sources WHERE project_id=? AND intent_id=? ORDER BY rowid",
            (pid, iid),
        )
        # Compute dynamic status for frontend filtering
        if row["to_fact_id"] is not None:
            status = "concluded"
        elif row["worker"] is not None:
            status = "working"
        else:
            status = "unclaimed"
        return {
            "id": row["id"],
            "from": [s["fact_id"] for s in sources],
            "to": row["to_fact_id"],
            "description": row["description"],
            "creator": row["creator"],
            "worker": row["worker"],
            "last_heartbeat_at": row["last_heartbeat_at"],
            "created_at": row["created_at"],
            "concluded_at": row["concluded_at"],
            "status": status,
        }

    def list_intents(self, pid: str) -> list[dict[str, Any]]:
        rows = self.db.fetchall(
            "SELECT id FROM intents WHERE project_id=? ORDER BY created_at", (pid,)
        )
        return [self.get_intent(pid, r["id"]) for r in rows]

    def claim_intent(self, pid: str, iid: str, worker: str) -> dict[str, Any] | None:
        """Claim an intent. Returns None if already claimed by another worker."""
        self.expire_workers(pid)
        with self.db.connect() as conn:
            intent = conn.execute(
                "SELECT worker, to_fact_id FROM intents WHERE project_id=? AND id=?",
                (pid, iid),
            ).fetchone()
            if intent is None:
                return None
            if intent["to_fact_id"] is not None:
                return None
            if intent["worker"] is not None and intent["worker"] != worker:
                return None
            cur = conn.execute(
                "UPDATE intents SET worker=?, last_heartbeat_at=? "
                "WHERE project_id=? AND id=?",
                (worker, _iso_now(), pid, iid),
            )
            if cur.rowcount == 0:
                return None
        return self.get_intent(pid, iid)

    def heartbeat_intent(self, pid: str, iid: str, worker: str) -> bool:
        """Refresh an intent lease. False when this worker no longer holds it.

        Expiry runs first, so a worker whose own heartbeat already lapsed finds
        its claim cleared and gets told the lease is gone instead of silently
        resurrecting a claim another worker may have taken over.
        """
        self.expire_workers(pid)
        cur = self.db.execute(
            "UPDATE intents SET last_heartbeat_at=? WHERE project_id=? AND id=? AND worker=?",
            (_iso_now(), pid, iid, worker),
        )
        return cur > 0

    def release_intent(self, pid: str, iid: str, worker: str) -> bool:
        self.expire_workers(pid)
        cur = self.db.execute(
            "UPDATE intents SET worker=NULL, last_heartbeat_at=NULL "
            "WHERE project_id=? AND id=? AND worker=? AND to_fact_id IS NULL",
            (pid, iid, worker),
        )
        return cur > 0

    def conclude_intent(self, pid: str, iid: str, payload: Any) -> dict[str, Any] | None:
        """Conclude an intent: write its fact (and optionally link a trial + metrics)."""
        with self.db.connect() as conn:
            intent = conn.execute(
                "SELECT * FROM intents WHERE project_id=? AND id=?", (pid, iid)
            ).fetchone()
            if intent is None:
                return None
            fact_id = f"f{self.db.next_scoped(pid, 'fact'):03d}"
            metrics_json = json.dumps(payload.metrics or {})
            artifacts_json = json.dumps(payload.artifacts) if payload.artifacts else None
            now = _iso_now()
            conn.execute(
                "INSERT INTO facts(id, project_id, description, metrics, trial_id, artifacts, created_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (fact_id, pid, payload.description, metrics_json, payload.trial_id, artifacts_json, now),
            )
            conn.execute(
                "UPDATE intents SET to_fact_id=?, worker=?, last_heartbeat_at=?, concluded_at=? "
                "WHERE project_id=? AND id=?",
                (fact_id, payload.worker, now, now, pid, iid),
            )
            if payload.trial_id:
                conn.execute(
                    "UPDATE trials SET fact_id=? WHERE id=? AND project_id=?",
                    (fact_id, payload.trial_id, pid),
                )
        fact_row = self.db.fetchone("SELECT * FROM facts WHERE project_id=? AND id=?", (pid, fact_id))
        fact_row["metrics"] = self.db.json_loads(fact_row["metrics"], {})
        fact_row["artifacts"] = self.db.json_loads(fact_row["artifacts"], None)
        return {"fact": fact_row, "intent": self.get_intent(pid, iid)}

    def complete_project(self, pid: str, from_fact_ids: list[str],
                         description: str, worker: str) -> dict[str, Any] | None:
        """Complete the project: create terminal intent + 'goal' outcome fact."""
        with self.db.connect() as conn:
            proj = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
            if proj is None:
                return None

            iid = self.next_intent_id(pid)
            now = _iso_now()
            conn.execute(
                "INSERT INTO intents(id, project_id, to_fact_id, description, creator, "
                "worker, last_heartbeat_at, created_at, concluded_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (iid, pid, "goal", description, worker, worker, now, now, now),
            )
            for fid in from_fact_ids:
                conn.execute(
                    "INSERT OR IGNORE INTO intent_sources(intent_id, project_id, fact_id) "
                    "VALUES(?,?,?)", (iid, pid, fid),
                )
            conn.execute("UPDATE projects SET status='completed' WHERE id=?", (pid,))

        proj_meta = self.project_summary(pid)
        fact_row = self.db.fetchone("SELECT * FROM facts WHERE project_id=? AND id='goal'", (pid,))
        fact_row["metrics"] = self.db.json_loads(fact_row.get("metrics", "{}"), {})
        intent = self.get_intent(pid, iid)
        return {"project": proj_meta, "fact": fact_row, "intent": intent}

    def add_hint(self, pid: str, content: str, creator: str) -> dict[str, Any]:
        hid = f"h{self.db.next_scoped(pid, 'hint'):03d}"
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO hints(id, project_id, content, creator, created_at) VALUES(?,?,?,?,?)",
                (hid, pid, content, creator, _iso_now()),
            )
        return {"id": hid, "content": content, "creator": creator}

    def list_hints(self, pid: str) -> list[dict[str, Any]]:
        return self.db.fetchall(
            "SELECT id, content, creator, created_at FROM hints WHERE project_id=? ORDER BY created_at",
            (pid,),
        )

    def delete_hint(self, pid: str, hid: str) -> bool:
        """Remove one hint. Returns False when the hint does not exist."""
        with self.db.connect() as conn:
            cur = conn.execute(
                "DELETE FROM hints WHERE project_id=? AND id=?",
                (pid, hid),
            )
            return cur.rowcount > 0

    # =====================================================================
    # Export graph YAML (Cairn protocol)
    # =====================================================================

    def export_graph_yaml(self, pid: str) -> str:
        """Export the project graph in Cairn YAML format."""
        self.expire_workers(pid)
        self.expire_reason_leases(pid)
        proj = self.db.fetchone("SELECT * FROM projects WHERE id=?", (pid,))
        if proj is None:
            return ""

        facts = self.list_facts(pid)
        hints = self.list_hints(pid)
        intents = self.list_intents(pid)

        origin_desc = ""
        goal_desc = ""
        for f in facts:
            if f["id"] == "origin":
                origin_desc = f["description"]
            elif f["id"] == "goal":
                goal_desc = f["description"]

        data: dict[str, Any] = {
            "project": {
                "title": proj["title"],
                "origin": origin_desc,
                "goal": goal_desc,
                "goal_metric": proj["goal_metric"],
                "goal_direction": proj["goal_direction"],
                "goal_target": proj["goal_target"],
                "bootstrap_enabled": bool(proj["bootstrap_enabled"]),
            }
        }

        if hints:
            data["hints"] = [
                {
                    "content": h["content"],
                    "creator": h["creator"],
                    "created_at": _format_export_timestamp(h["created_at"]),
                }
                for h in hints
            ]

        data["facts"] = [
            {
                "id": f["id"],
                "description": f["description"],
                "metrics": f.get("metrics", {}),
                "trial_id": f.get("trial_id"),
            }
            for f in facts
        ]

        intent_list = []
        for i in intents:
            entry: dict[str, Any] = {
                "from": i["from"],
                "to": i["to"],
                "description": i["description"],
                "creator": i["creator"],
                "worker": i["worker"],
                "created_at": _format_export_timestamp(i["created_at"]),
                "concluded_at": _format_export_timestamp(i["concluded_at"]),
            }
            intent_list.append(entry)

        if intent_list:
            data["intents"] = intent_list

        return yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False)

    # =====================================================================
    # Observations — shared model-inspection findings
    # =====================================================================

    def add_observation(self, pid: str, payload: Any) -> dict[str, Any]:
        """Record what an agent saw when it inspected a model.

        Kept separate from `facts` deliberately: facts are intent conclusions
        (one per concluded intent, part of the causal spine), while observations
        are measurements that can be taken freely and repeatedly.
        """
        oid = f"o{self.db.next_scoped(pid, 'observation'):03d}"
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO observations(id, project_id, trial_id, observer, tool, "
                "finding, payload, created_at) VALUES(?,?,?,?,?,?,?,?)",
                (oid, pid, payload.trial_id, payload.observer, payload.tool,
                 payload.finding, json.dumps(payload.payload, default=str), _iso_now()),
            )
        return self.get_observation(pid, oid)

    def get_observation(self, pid: str, oid: str) -> dict[str, Any] | None:
        row = self.db.fetchone(
            "SELECT * FROM observations WHERE project_id=? AND id=?", (pid, oid)
        )
        if row is None:
            return None
        row["payload"] = self.db.json_loads(row["payload"], {})
        return row

    def list_observations(self, pid: str, trial_id: str | None = None,
                          limit: int = 100) -> list[dict[str, Any]]:
        """Newest-first, so the board shows the freshest look at the model."""
        q = "SELECT * FROM observations WHERE project_id=?"
        args: list[Any] = [pid]
        if trial_id:
            q += " AND trial_id=?"
            args.append(trial_id)
        q += " ORDER BY created_at DESC, id DESC LIMIT ?"
        args.append(limit)
        rows = self.db.fetchall(q, args)
        for row in rows:
            row["payload"] = self.db.json_loads(row["payload"], {})
        return rows

    # =====================================================================
    # Search configuration (replaces config/dispatch*.yaml for the WebUI)
    # =====================================================================

    _SEARCH_DEFAULTS = {
        "max_trials": 12, "max_workers": 2,
        "workers": [{"name": "mock-a", "driver": "mock"},
                    {"name": "mock-b", "driver": "mock"}],
    }

    def get_search_config(self, pid: str) -> dict[str, Any]:
        """Stored config, or the defaults when the project has never been configured."""
        row = self.db.fetchone(
            "SELECT * FROM search_configs WHERE project_id=?", (pid,)
        )
        if row is None:
            return {"project_id": pid, "updated_at": None, **self._SEARCH_DEFAULTS}
        row["workers"] = self.db.json_loads(row["workers"], [])
        if not row["workers"]:
            row["workers"] = list(self._SEARCH_DEFAULTS["workers"])
        # Parse worker_requirement JSON if present
        if row.get("worker_requirement"):
            row["worker_requirement"] = self.db.json_loads(row["worker_requirement"], None)
        return row

    def put_search_config(self, pid: str, payload: Any) -> dict[str, Any]:
        # Handle legacy workers field (deprecated)
        workers = [w.model_dump() if hasattr(w, "model_dump") else dict(w)
                   for w in (payload.workers or [])]
        if not workers:
            workers = list(self._SEARCH_DEFAULTS["workers"])

        # Handle new worker_requirement field
        worker_requirement = None
        if hasattr(payload, 'worker_requirement') and payload.worker_requirement:
            worker_requirement = json.dumps(payload.worker_requirement.model_dump()
                                           if hasattr(payload.worker_requirement, "model_dump")
                                           else dict(payload.worker_requirement))

        # Handle legacy max_workers field (deprecated)
        max_workers = payload.max_workers if hasattr(payload, 'max_workers') and payload.max_workers else 2

        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO search_configs(project_id, max_trials, max_workers, "
                "workers, worker_requirement, updated_at) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(project_id) DO UPDATE SET "
                "max_trials=excluded.max_trials, "
                "max_workers=excluded.max_workers, workers=excluded.workers, "
                "worker_requirement=excluded.worker_requirement, "
                "updated_at=excluded.updated_at",
                (pid, payload.max_trials, max_workers,
                 json.dumps(workers), worker_requirement, _iso_now()),
            )
        return self.get_search_config(pid)

    # =====================================================================
    # Trials ledger
    # =====================================================================

    def next_trial_id(self, pid: str) -> str:
        return f"t{self.db.next_scoped(pid, 'trial'):03d}"

    def register_trial(self, pid: str, payload: Any) -> tuple[dict[str, Any] | None, str | None]:
        """Validate artifacts on disk, then insert a ledger row.

        Returns (trial_dict, error). Duplicate fingerprint -> error='duplicate'.
        """
        out_dir = Path(payload.out_dir)
        if not out_dir.is_dir():
            return None, "out_dir does not exist"
        for required in ("spec.json", "summary.json", "metrics.jsonl"):
            if not (out_dir / required).exists():
                return None, f"missing artifact: {required}"

        try:
            summary = read_summary(out_dir)
        except Exception:
            summary = {}

        spec = payload.spec or {}
        if not spec:
            try:
                spec = json.loads((out_dir / "spec.json").read_text(encoding="utf-8"))
            except Exception:
                spec = {}

        tid = payload.trial_id or summary.get("trial_id") or self.next_trial_id(pid)
        fingerprint = payload.fingerprint or spec_fingerprint(spec)

        existing = self.db.fetchone(
            "SELECT id FROM trials WHERE project_id=? AND fingerprint=?", (pid, fingerprint)
        )
        if existing and existing["id"] != tid:
            return None, f"duplicate:{existing['id']}"

        name = payload.name or summary.get("name") or tid
        status = payload.status or summary.get("status") or "completed"
        final = summary.get("final") or {}
        best = summary.get("best") or {}
        diff = summary.get("diff") or {}

        signals = payload.signals or summary.get("signals") or []
        top_signal = signals[0] if signals else None

        row = {
            "id": tid,
            "project_id": pid,
            "name": name,
            "parent_id": payload.parent_id or summary.get("parent"),
            "intent_id": payload.intent_id,
            "fact_id": payload.fact_id,
            "worker": payload.worker,
            "backend": payload.backend or summary.get("backend", "surrogate"),
            "fingerprint": fingerprint,
            "spec_path": str(out_dir / "spec.json"),
            "out_dir": str(out_dir),
            "diff_json": json.dumps(payload.diff_json or diff),
            "status": status,
            "steps_done": payload.steps_done or summary.get("steps_completed", 0),
            "duration_sec": payload.duration_sec or summary.get("duration_sec", 0.0),
            "final_train_loss": payload.final_train_loss if payload.final_train_loss is not None else final.get("train_loss"),
            "final_val_loss": payload.final_val_loss if payload.final_val_loss is not None else final.get("val_loss"),
            "best_val_loss": payload.best_val_loss if payload.best_val_loss is not None else best.get("val_loss"),
            "final_val_acc": payload.final_val_acc if payload.final_val_acc is not None else final.get("val_acc"),
            "best_val_acc": payload.best_val_acc if payload.best_val_acc is not None else best.get("val_acc"),
            "verdict": payload.verdict or summary.get("verdict"),
            "signal_count": len(signals),
            "top_signal": top_signal,
            "created_at": _iso_now(),
        }
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO trials(id, project_id, name, parent_id, intent_id, fact_id, worker, "
                "backend, fingerprint, spec_path, out_dir, diff_json, status, steps_done, "
                "duration_sec, final_train_loss, final_val_loss, best_val_loss, final_val_acc, "
                "best_val_acc, verdict, signal_count, top_signal, created_at) "
                "VALUES(:id,:project_id,:name,:parent_id,:intent_id,:fact_id,:worker,:backend,"
                ":fingerprint,:spec_path,:out_dir,:diff_json,:status,:steps_done,:duration_sec,"
                ":final_train_loss,:final_val_loss,:best_val_loss,:final_val_acc,:best_val_acc,"
                ":verdict,:signal_count,:top_signal,:created_at)",
                row,
            )
        return self.get_trial(tid), None

    def get_trial(self, tid: str) -> dict[str, Any] | None:
        row = self.db.fetchone("SELECT * FROM trials WHERE id=?", (tid,))
        if row is None:
            return None
        diff = self.db.json_loads(row.pop("diff_json"), {})
        from mts.trainer.spec import format_diff

        row["diff_text"] = format_diff(diff)
        row["diff"] = diff
        return row

    def mark_trial_queued(self, tid: str, device: str | None = None) -> bool:
        cur = self.db.execute(
            "UPDATE trials SET status='queued', queued_at=?, device=? "
            "WHERE id=? AND status IN ('pending', 'queued')",
            (_iso_now(), device, tid),
        )
        return cur > 0

    def list_queued_trials(self, pid: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT * FROM trials WHERE status='queued'"
        args: list[Any] = []
        if pid:
            q += " AND project_id=?"
            args.append(pid)
        q += " ORDER BY queued_at ASC, created_at ASC"
        return self.db.fetchall(q, args)

    def mark_trial_running(self, tid: str, worker: str | None = None,
                           device: str | None = None) -> bool:
        cur = self.db.execute(
            "UPDATE trials SET status='running', last_heartbeat_at=?, "
            "worker=COALESCE(?, worker), device=COALESCE(?, device) "
            "WHERE id=? AND status IN ('pending', 'queued', 'running', 'interrupted')",
            (_iso_now(), worker, device, tid),
        )
        return cur > 0

    def heartbeat_trial(self, tid: str, worker: str | None,
                        progress: dict[str, Any] | None = None) -> bool:
        progress = progress or {}
        cur = self.db.execute(
            "UPDATE trials SET last_heartbeat_at=?, "
            "progress_step=?, progress_max_steps=?, eta_sec=? "
            "WHERE id=? AND (worker IS NULL OR worker=?)",
            (
                _iso_now(),
                progress.get("step", 0),
                progress.get("max_steps"),
                progress.get("eta_sec"),
                tid,
                worker,
            ),
        )
        return cur > 0

    def mark_trial_interrupted(self, tid: str) -> bool:
        cur = self.db.execute(
            "UPDATE trials SET status='interrupted' WHERE id=? AND status='running'", (tid,)
        )
        return cur > 0

    def list_stale_running_trials(self, timeout_sec: int = 300) -> list[str]:
        import datetime as _dt

        cutoff = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(seconds=timeout_sec)).isoformat()
        rows = self.db.fetchall(
            "SELECT id FROM trials WHERE status='running' AND last_heartbeat_at IS NOT NULL "
            "AND last_heartbeat_at < ?",
            (cutoff,),
        )
        return [r["id"] for r in rows]

    def list_trials(self, pid: str, verdict: str | None = None,
                    status: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT id FROM trials WHERE project_id=?"
        args: list[Any] = [pid]
        if verdict:
            q += " AND verdict=?"
            args.append(verdict)
        if status:
            q += " AND status=?"
            args.append(status)
        q += " ORDER BY created_at"
        rows = self.db.fetchall(q, args)
        return [self.get_trial(r["id"]) for r in rows]

    # =====================================================================
    # Knowledge
    # =====================================================================

    def upsert_knowledge(self, payload: Any) -> dict[str, Any]:
        slug = _slugify(payload.slug)
        path = self.memory_dir / f"{slug}.md"
        front = (
            f"---\ntitle: {json.dumps(payload.title, ensure_ascii=False)}\n"
            f"tags: {json.dumps(payload.tags)}\nconfidence: {payload.confidence}\n---\n\n"
        )
        path.write_text(front + payload.body + "\n", encoding="utf-8")
        now = _iso_now()
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO knowledge_notes(slug, title, tags, path, confidence, created_at, updated_at) "
                "VALUES(?,?,?,?,?,?,?) "
                "ON CONFLICT(slug) DO UPDATE SET title=excluded.title, tags=excluded.tags, "
                "path=excluded.path, confidence=excluded.confidence, updated_at=excluded.updated_at",
                (slug, payload.title, json.dumps(payload.tags), str(path), payload.confidence, now, now),
            )
        return self.get_knowledge(slug)

    def get_knowledge(self, slug: str) -> dict[str, Any] | None:
        row = self.db.fetchone("SELECT * FROM knowledge_notes WHERE slug=?", (slug,))
        if row is None:
            return None
        row["tags"] = self.db.json_loads(row["tags"], [])
        try:
            text = Path(row["path"]).read_text(encoding="utf-8")
            if text.startswith("---"):
                _, _, rest = text.split("---", 2)
                row["body"] = rest.lstrip("\n")
            else:
                row["body"] = text
        except OSError:
            row["body"] = ""
        return row

    def list_knowledge(self, tag: str | None = None, q: str | None = None) -> list[dict[str, Any]]:
        rows = self.db.fetchall("SELECT slug FROM knowledge_notes ORDER BY updated_at DESC")
        out = []
        for r in rows:
            item = self.get_knowledge(r["slug"])
            if item is None:
                continue
            if tag and tag not in item["tags"]:
                continue
            if q and q.lower() not in (item["title"] + " " + item["body"]).lower():
                continue
            out.append({k: v for k, v in item.items() if k != "body"})
        return out

    # =====================================================================
    # LLM providers
    # =====================================================================

    def list_providers(self) -> list[dict[str, Any]]:
        rows = self.db.fetchall(
            "SELECT * FROM providers ORDER BY is_default DESC, name ASC"
        )
        return [self._provider_out(r) for r in rows]

    def get_provider(self, pid: str) -> dict[str, Any] | None:
        row = self.db.fetchone("SELECT * FROM providers WHERE id=?", (pid,))
        return None if row is None else self._provider_out(row)

    def get_provider_secret(self, pid: str) -> dict[str, Any] | None:
        """Raw row including api_key. Server-side only — never return over HTTP."""
        row = self.db.fetchone("SELECT * FROM providers WHERE id=?", (pid,))
        if row is None:
            return None
        row["extra"] = self.db.json_loads(row["extra"], {})
        row["is_default"] = bool(row["is_default"])
        return row

    def get_default_provider_secret(self) -> dict[str, Any] | None:
        row = self.db.fetchone(
            "SELECT * FROM providers WHERE is_default=1 ORDER BY updated_at DESC LIMIT 1"
        )
        if row is None:
            return None
        row["extra"] = self.db.json_loads(row["extra"], {})
        row["is_default"] = True
        return row

    def create_provider(self, payload: Any) -> dict[str, Any]:
        pid = f"prov_{self.db.next_global('provider'):03d}"
        now = _iso_now()
        extra = {
            "max_tokens": payload.max_tokens,
            "temperature": payload.temperature,
            "headers": payload.headers or {},
        }
        with self.db.connect() as conn:
            if payload.is_default:
                conn.execute("UPDATE providers SET is_default=0")
            conn.execute(
                "INSERT INTO providers(id, name, kind, base_url, api_key, model, "
                "is_default, extra, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (pid, payload.name, payload.kind, payload.base_url.rstrip("/"),
                 payload.api_key, payload.model, 1 if payload.is_default else 0,
                 json.dumps(extra), now, now),
            )
        return self.get_provider(pid)  # type: ignore[return-value]

    def update_provider(self, pid: str, payload: Any) -> dict[str, Any] | None:
        current = self.get_provider_secret(pid)
        if current is None:
            return None

        extra = dict(current["extra"])
        if payload.max_tokens is not None:
            extra["max_tokens"] = payload.max_tokens
        if payload.temperature is not None:
            extra["temperature"] = payload.temperature
        if payload.headers is not None:
            extra["headers"] = payload.headers

        # api_key: None means "keep what is stored", "" means "clear it".
        api_key = current["api_key"] if payload.api_key is None else payload.api_key
        base_url = current["base_url"] if payload.base_url is None else payload.base_url.rstrip("/")
        is_default = current["is_default"] if payload.is_default is None else payload.is_default

        with self.db.connect() as conn:
            if is_default and not current["is_default"]:
                conn.execute("UPDATE providers SET is_default=0")
            conn.execute(
                "UPDATE providers SET name=?, kind=?, base_url=?, api_key=?, model=?, "
                "is_default=?, extra=?, updated_at=? WHERE id=?",
                (payload.name or current["name"],
                 payload.kind or current["kind"],
                 base_url, api_key,
                 current["model"] if payload.model is None else payload.model,
                 1 if is_default else 0, json.dumps(extra), _iso_now(), pid),
            )
        return self.get_provider(pid)

    def delete_provider(self, pid: str) -> bool:
        with self.db.connect() as conn:
            cur = conn.execute("DELETE FROM providers WHERE id=?", (pid,))
            return cur.rowcount > 0

    def _provider_out(self, row: dict[str, Any]) -> dict[str, Any]:
        extra = self.db.json_loads(row["extra"], {})
        key = row["api_key"] or ""
        return {
            "id": row["id"],
            "name": row["name"],
            "kind": row["kind"],
            "base_url": row["base_url"],
            "api_key_masked": _mask_key(key),
            "has_api_key": bool(key),
            "model": row["model"],
            "is_default": bool(row["is_default"]),
            "max_tokens": int(extra.get("max_tokens") or 4096),
            "temperature": extra.get("temperature"),
            "headers": extra.get("headers") or {},
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }


def _mask_key(key: str) -> str:
    """sk-abc...wxyz — enough to recognise a key, not enough to use it."""
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"

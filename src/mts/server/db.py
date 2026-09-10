"""SQLite persistence for the board.

Schema follows docs/04. Big sequences (curves, histograms) live in files under
`runs/`; SQLite only stores indices and summaries. WAL mode + foreign keys on.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS settings (
    rowid            INTEGER PRIMARY KEY CHECK (rowid = 1),
    intent_timeout   INTEGER NOT NULL DEFAULT 60,
    reason_timeout   INTEGER NOT NULL DEFAULT 60,
    max_trials       INTEGER NOT NULL DEFAULT 60,
    updated_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    id                TEXT PRIMARY KEY,
    title             TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'active',
    bootstrap_enabled INTEGER NOT NULL DEFAULT 1,
    origin            TEXT NOT NULL,
    goal              TEXT NOT NULL,
    goal_metric       TEXT NOT NULL DEFAULT 'val_loss',
    goal_target       REAL,
    goal_direction    TEXT NOT NULL DEFAULT 'minimize',
    budget_max_trials INTEGER NOT NULL DEFAULT 60,
    created_at        TEXT NOT NULL,
    reason_worker     TEXT, reason_trigger TEXT,
    reason_started_at TEXT, reason_last_heartbeat_at TEXT
);

CREATE TABLE IF NOT EXISTS facts (
    id          TEXT NOT NULL,
    project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    description TEXT NOT NULL,
    metrics     TEXT NOT NULL DEFAULT '{}',
    trial_id    TEXT,
    artifacts   TEXT,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (id, project_id)
);

CREATE TABLE IF NOT EXISTS intents (
    id              TEXT NOT NULL,
    project_id      TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    to_fact_id      TEXT,
    description     TEXT NOT NULL,
    creator         TEXT NOT NULL,
    worker          TEXT, last_heartbeat_at TEXT,
    created_at      TEXT NOT NULL, concluded_at TEXT,
    PRIMARY KEY (id, project_id)
);

CREATE TABLE IF NOT EXISTS intent_sources (
    intent_id TEXT NOT NULL, project_id TEXT NOT NULL, fact_id TEXT NOT NULL,
    PRIMARY KEY (intent_id, project_id, fact_id),
    FOREIGN KEY (intent_id, project_id) REFERENCES intents(id, project_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS hints (
    id TEXT NOT NULL, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    content TEXT NOT NULL, creator TEXT NOT NULL, created_at TEXT NOT NULL,
    PRIMARY KEY (id, project_id)
);

CREATE TABLE IF NOT EXISTS trials (
    id            TEXT PRIMARY KEY,
    project_id    TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    parent_id     TEXT,
    intent_id     TEXT,
    fact_id       TEXT,
    worker        TEXT,
    backend       TEXT NOT NULL,
    fingerprint   TEXT NOT NULL,
    spec_path     TEXT NOT NULL,
    out_dir       TEXT NOT NULL,
    diff_json     TEXT NOT NULL DEFAULT '{}',
    status        TEXT NOT NULL,
    steps_done    INTEGER NOT NULL DEFAULT 0,
    duration_sec  REAL NOT NULL DEFAULT 0,
    final_train_loss REAL, final_val_loss REAL, best_val_loss REAL,
    final_val_acc REAL, best_val_acc REAL,
    verdict       TEXT,
    signal_count  INTEGER NOT NULL DEFAULT 0,
    top_signal    TEXT,
    created_at    TEXT NOT NULL,
    last_heartbeat_at TEXT,
    progress_step  INTEGER NOT NULL DEFAULT 0,
    progress_max_steps INTEGER,
    eta_sec       REAL,
    queued_at     TEXT,
    device        TEXT
);
CREATE INDEX IF NOT EXISTS idx_trials_project ON trials(project_id, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_trials_fp ON trials(project_id, fingerprint);

-- Search configuration per project, replacing config/dispatch*.yaml for the
-- WebUI path. `settings` cannot hold this: it is a single-row table (rowid=1
-- CHECK) with three fixed columns, and this is per-project with a variable
-- worker roster. Credentials stay out of the roster — a worker row references a
-- provider by id and the dispatcher resolves the key at launch.
CREATE TABLE IF NOT EXISTS search_configs (
    project_id      TEXT PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
    max_trials      INTEGER NOT NULL DEFAULT 12,
    max_workers     INTEGER NOT NULL DEFAULT 2,
    workers         TEXT NOT NULL DEFAULT '[]',
    updated_at      TEXT NOT NULL
);

-- Observations: what an agent SAW when it looked inside a model.
--
-- Distinct from `facts` on purpose. A fact is the conclusion of an intent ("this
-- configuration diverged") and is part of the graph's causal spine — one fact per
-- concluded intent. An observation is a *measurement* an agent took with an
-- inspection tool ("block0 receives 0.4% of block11's gradient"), and there can be
-- many per trial, taken at any time, by any agent, without concluding anything.
-- Folding them into `facts` would corrupt the intent→fact accounting the tree
-- rendering and the report depend on.
CREATE TABLE IF NOT EXISTS observations (
    id          TEXT NOT NULL,
    project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    trial_id    TEXT,
    observer    TEXT NOT NULL,
    tool        TEXT NOT NULL,
    finding     TEXT NOT NULL,
    payload     TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL,
    PRIMARY KEY (id, project_id)
);
CREATE INDEX IF NOT EXISTS idx_obs_project ON observations(project_id, created_at);
CREATE INDEX IF NOT EXISTS idx_obs_trial ON observations(trial_id);

CREATE TABLE IF NOT EXISTS knowledge_notes (
    slug        TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    tags        TEXT NOT NULL DEFAULT '[]',
    path        TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 0.5,
    hits        INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL, updated_at TEXT NOT NULL
);

-- LLM providers configured in the WebUI. Holds the credential the `llm` driver
-- needs so a user can point MTS at any OpenAI-compatible endpoint (vLLM, Ollama,
-- DashScope, DeepSeek, ...) or an Anthropic endpoint without touching the shell.
--
-- api_key IS a secret and IS stored here: that is the whole point of configuring
-- it in the UI. It never leaves the server in cleartext — the read APIs return
-- `api_key_masked` only. Protect the sqlite file the same way you protect a
-- .env; MTS assumes a single-tenant local deployment.
CREATE TABLE IF NOT EXISTS providers (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'openai',   -- openai | anthropic | claudecode
    base_url    TEXT NOT NULL DEFAULT '',
    api_key     TEXT NOT NULL DEFAULT '',
    model       TEXT NOT NULL DEFAULT '',
    is_default  INTEGER NOT NULL DEFAULT 0,
    extra       TEXT NOT NULL DEFAULT '{}',       -- JSON: max_tokens, temperature, headers
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_providers_name ON providers(name);

CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS scoped_counters (
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    kind TEXT NOT NULL, value INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (project_id, kind)
);
"""


class Database:
    """Thin sqlite wrapper. One connection per request; row_factory=sqlite3.Row."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def configure(self) -> None:
        """Create tables and run lightweight migrations (add missing columns)."""
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            # Add missing columns to older databases (mirrors Cairn's upgrade path).
            for table, col, decl in [
                ("facts", "metrics", "TEXT NOT NULL DEFAULT '{}'"),
                ("facts", "artifacts", "TEXT"),
                ("trials", "last_heartbeat_at", "TEXT"),
                ("trials", "progress_step", "INTEGER NOT NULL DEFAULT 0"),
                ("trials", "progress_max_steps", "INTEGER"),
                ("trials", "eta_sec", "REAL"),
                ("trials", "queued_at", "TEXT"),
            ]:
                cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
                if col not in cols:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")

            # Drop columns the autonomous-agent refactor made dead:
            #   projects.base_spec  — held the pre-refactor TrialSpec. Agents write
            #     their own train.py now. NOT NULL, so leaving it would break every
            #     INSERT that no longer supplies it.
            #   search_configs.seed/backend — parameterised the server-side training
            #     loop that no longer exists; the dispatcher never read them.
            #   search_configs.n_agents — duplicated max_workers. Only max_workers
            #     reaches DispatchConfig.runtime, so n_agents was stored and ignored.
            #   facts.verdict — the health label the old trainer emitted. The conclude
            #     endpoint always wrote NULL and no reader consumed it.
            #   projects.budget_gpu_seconds, intents.parent_trial_id, intents.axis_hint,
            #     trials.ckpt_path — declared, never written or read. (trials.device
            #     stays: mark_trial_queued/heartbeat write it.)
            for table, col in [
                ("projects", "base_spec"),
                ("projects", "budget_gpu_seconds"),
                ("search_configs", "backend"),
                ("search_configs", "seed"),
                ("search_configs", "n_agents"),
                ("facts", "verdict"),
                ("intents", "parent_trial_id"),
                ("intents", "axis_hint"),
                ("trials", "ckpt_path"),
            ]:
                cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
                if col in cols:
                    conn.execute(f"ALTER TABLE {table} DROP COLUMN {col}")

    def _now(self) -> str:
        import datetime as _dt

        return _dt.datetime.now(_dt.timezone.utc).isoformat()

    # ---- counters ---------------------------------------------------------

    def next_global(self, name: str) -> int:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO counters(name, value) VALUES(?, 1) "
                "ON CONFLICT(name) DO UPDATE SET value = value + 1",
                (name,),
            )
            row = conn.execute("SELECT value FROM counters WHERE name=?", (name,)).fetchone()
            return int(row["value"])

    def next_scoped(self, project_id: str, kind: str) -> int:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO scoped_counters(project_id, kind, value) VALUES(?, ?, 1) "
                "ON CONFLICT(project_id, kind) DO UPDATE SET value = value + 1",
                (project_id, kind),
            )
            row = conn.execute(
                "SELECT value FROM scoped_counters WHERE project_id=? AND kind=?",
                (project_id, kind),
            ).fetchone()
            return int(row["value"])

    # ---- settings ---------------------------------------------------------

    def get_settings(self) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM settings WHERE rowid=1").fetchone()
        if row is None:
            return {"intent_timeout": 60, "reason_timeout": 60, "max_trials": 60}
        return dict(row)

    def put_settings(self, **values: Any) -> dict[str, Any]:
        allowed = {"intent_timeout", "reason_timeout", "max_trials"}
        merged = self.get_settings()
        merged.update({k: v for k, v in values.items() if k in allowed})
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO settings(rowid, intent_timeout, reason_timeout, max_trials, updated_at) "
                "VALUES(1, ?, ?, ?, ?) "
                "ON CONFLICT(rowid) DO UPDATE SET "
                "intent_timeout=excluded.intent_timeout, reason_timeout=excluded.reason_timeout, "
                "max_trials=excluded.max_trials, updated_at=excluded.updated_at",
                (merged["intent_timeout"], merged["reason_timeout"], merged["max_trials"], self._now()),
            )
        return merged

    # ---- generic helpers --------------------------------------------------

    def fetchall(self, query: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(r) for r in conn.execute(query, tuple(params)).fetchall()]

    def fetchone(self, query: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(query, tuple(params)).fetchone()
            return dict(row) if row else None

    def execute(self, query: str, params: Iterable[Any] = ()) -> int:
        with self.connect() as conn:
            cur = conn.execute(query, tuple(params))
            return cur.rowcount

    def json_loads(self, text: str | None, default: Any) -> Any:
        if not text:
            return default
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return default

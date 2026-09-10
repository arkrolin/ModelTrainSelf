"""P2-1 tests: ResourceSpec + DevicePool + queued state machine.

These pin down docs/09's resource-aware scheduling primitives:
  - ResourceSpec estimates accelerator memory from params × precision.
  - DevicePool probes CUDA (fallback cpu), reserves/releases, respects health.
  - The trial state machine gains a `queued` state (pending → queued → running).
"""

from __future__ import annotations

from pathlib import Path

from mts.dispatcher.runtime.device import Device, DevicePool
from mts.server.db import Database
from mts.server.models import ProjectCreate
from mts.server.services import Service
from mts.trainer.spec import ResourceSpec, TrialSpec


# ---------------------------------------------------------------------------
# ResourceSpec
# ---------------------------------------------------------------------------

def test_resource_spec_defaults():
    r = ResourceSpec()
    assert r.device == "auto"
    assert r.mem_bytes == 0
    assert r.precision == "fp32"


def test_resource_spec_memory_estimate():
    # fp32: params × 4B × 3
    assert ResourceSpec(precision="fp32").estimate_mem_bytes(1000) == 12000
    # bf16 halves the per-param bytes
    assert ResourceSpec(precision="bf16").estimate_mem_bytes(1000) == 6000
    # explicit mem_bytes wins over the estimate
    assert ResourceSpec(mem_bytes=777, precision="fp32").estimate_mem_bytes(1000) == 777


def test_resource_spec_excluded_from_diff():
    a = TrialSpec(resource=ResourceSpec(device="cuda:0"))
    b = TrialSpec(resource=ResourceSpec(device="cpu"))
    # resource is a placement field, not a search move.
    assert a.diff(b) == {}


# ---------------------------------------------------------------------------
# DevicePool
# ---------------------------------------------------------------------------

def test_device_pool_falls_back_to_cpu_without_cuda():
    pool = DevicePool(probe=False)  # no CUDA probe
    pool.devices["cpu"] = Device(name="cpu", mem_total=0, mem_free=0)
    assert pool.healthy_device_count() == 1
    dev = pool.reserve(1_000_000_000)  # cpu is not memory-bounded
    assert dev is not None and dev.name == "cpu"


def test_device_pool_reserve_and_release():
    gpu = Device(name="cuda:0", mem_total=10_000, mem_free=10_000, capability="sm_80")
    pool = DevicePool([gpu], probe=False)
    assert pool.healthy_device_count() == 1

    # Fits -> reserved.
    dev = pool.reserve(4_000)
    assert dev is not None and dev.name == "cuda:0"
    assert pool.devices["cuda:0"].available == 6_000

    # Too big -> None (must queue).
    assert pool.reserve(7_000) is None

    # Release reclaims the reservation.
    pool.release("cuda:0", 4_000)
    assert pool.devices["cuda:0"].available == 10_000


def test_device_pool_respects_health():
    bad = Device(name="cuda:0", mem_total=10_000, mem_free=10_000, healthy=False)
    pool = DevicePool([bad], probe=False)
    assert pool.reserve(1) is None
    assert pool.healthy_device_count() == 0


def test_device_pool_prefers_requested_device():
    a = Device(name="cuda:0", mem_total=1000, mem_free=1000)
    b = Device(name="cuda:1", mem_total=1000, mem_free=1000)
    pool = DevicePool([a, b], probe=False)
    dev = pool.reserve(100, prefer="cuda:1")
    assert dev is not None and dev.name == "cuda:1"


def test_device_pool_refresh_marks_missing_card_unhealthy():
    gpu = Device(name="cuda:0", mem_total=1000, mem_free=1000, capability="sm_80")
    pool = DevicePool([gpu], probe=False)
    # With no CUDA available, refresh() should mark the card unhealthy (it vanished).
    pool.refresh()
    assert pool.devices["cuda:0"].healthy is False


# ---------------------------------------------------------------------------
# queued state machine
# ---------------------------------------------------------------------------

def _make_service(root: Path) -> tuple[Service, str]:
    db = Database(root / "data" / "board.db")
    db.configure()
    svc = Service(db, root)
    base = TrialSpec.from_file("examples/specs/baseline.yaml")
    proj = svc.create_project(ProjectCreate(
        title="p2", origin="o", goal="g", base_spec=base.to_dict(),
    ))
    return svc, proj["id"]


def _insert_trial(db: Database, pid: str, tid: str, status: str = "pending") -> None:
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO trials(id, project_id, name, backend, fingerprint, "
            "spec_path, out_dir, status, created_at) "
            "VALUES(?, ?, 'test', 'surrogate', ?, '/x/spec.json', '/x', ?, "
            "'2026-01-01T00:00:00+00:00')",
            (tid, pid, f"fp-{tid}", status),
        )


def test_pending_to_queued_to_running(tmp_path: Path):
    svc, pid = _make_service(tmp_path)
    _insert_trial(svc.db, pid, "t001")

    # pending -> queued with a requested device.
    assert svc.mark_trial_queued("t001", device="cuda:0") is True
    t = svc.get_trial("t001")
    assert t["status"] == "queued"
    assert t["device"] == "cuda:0"
    assert t["queued_at"] is not None

    # queued -> running keeps the device.
    assert svc.mark_trial_running("t001", "worker-x") is True
    t = svc.get_trial("t001")
    assert t["status"] == "running"
    assert t["device"] == "cuda:0"


def test_list_queued_trials_fifo(tmp_path: Path):
    svc, pid = _make_service(tmp_path)
    _insert_trial(svc.db, pid, "t001")
    _insert_trial(svc.db, pid, "t002")
    svc.mark_trial_queued("t001")
    svc.mark_trial_queued("t002")

    queued = svc.list_queued_trials(pid)
    assert [q["id"] for q in queued] == ["t001", "t002"]  # FIFO order


def test_terminal_trial_cannot_queue(tmp_path: Path):
    svc, pid = _make_service(tmp_path)
    _insert_trial(svc.db, pid, "t001", status="completed")
    assert svc.mark_trial_queued("t001") is False

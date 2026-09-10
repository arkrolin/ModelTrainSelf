"""Accelerator resource pool (docs/09 P2-1).

The dispatcher schedules trials against *devices* rather than just a thread
count. This module probes available devices (CUDA via torch when present,
otherwise a single `cpu` fallback), tracks free memory, and hands out /
reclaims devices. On a no-GPU machine everything degrades gracefully to one
`cpu` device, so the surrogate path keeps working end to end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Device:
    name: str                 # "cuda:0" | "cuda:1" | "cpu"
    mem_total: int            # bytes; 0 = unknown (cpu)
    mem_free: int             # bytes; 0 = unknown
    capability: str = ""      # e.g. "sm_80"; empty = unknown / not CUDA
    healthy: bool = True
    # Reserved bytes tracked by this pool (trial placement), separate from the
    # OS-level free memory which torch reports.
    reserved: int = field(default=0, init=False)

    @property
    def available(self) -> int:
        """Free bytes after subtracting the pool's own reservations."""
        if self.mem_total == 0:
            return 0  # cpu has no meaningful byte accounting; treated as infinite
        return max(0, self.mem_free - self.reserved)

    def can_fit(self, need_bytes: int) -> bool:
        if not self.healthy:
            return False
        if self.name == "cpu":
            return True  # cpu is not memory-bounded in the pool's accounting
        if need_bytes <= 0:
            return True
        return self.available >= need_bytes


class DevicePool:
    """Probe + reserve accelerator devices for trial placement."""

    def __init__(self, devices: list[Device] | None = None, *, probe: bool = True):
        self.devices: dict[str, Device] = {}
        if devices is not None:
            for d in devices:
                self.devices[d.name] = d
        elif probe:
            self._probe()

    # ---- probing ----------------------------------------------------------

    def _probe(self) -> None:
        cuda_devices = _probe_cuda()
        if cuda_devices:
            for d in cuda_devices:
                self.devices[d.name] = d
        else:
            # Graceful fallback: a single cpu device so scheduling works without
            # any accelerator stack (matches the MVP's no-GPU boundary).
            self.devices["cpu"] = Device(name="cpu", mem_total=0, mem_free=0)

    def refresh(self) -> None:
        """Re-probe free memory / health (call before each dispatch round)."""
        for name, dev in list(self.devices.items()):
            if name == "cpu":
                continue
            fresh = _probe_cuda()
            by_name = {d.name: d for d in fresh}
            if name not in by_name:
                # The card disappeared (unplugged / driver reset).
                dev.healthy = False
                dev.mem_free = 0
            else:
                dev.mem_free = by_name[name].mem_free
                dev.mem_total = by_name[name].mem_total
                dev.capability = by_name[name].capability
                dev.healthy = True

    # ---- reservation ------------------------------------------------------

    def reserve(self, need_bytes: int, *, prefer: str | None = None) -> Device | None:
        """Find a healthy device that can fit `need_bytes`, mark it reserved.

        `prefer` names a device to try first (e.g. the trial's `resource.device`).
        Returns None if no device fits — the trial must queue.
        """
        if prefer and prefer != "auto":
            dev = self.devices.get(prefer)
            if dev is not None and dev.can_fit(need_bytes):
                if dev.name != "cpu":
                    dev.reserved += need_bytes
                return dev
        # Order: explicit CUDA devices first, cpu last (so cpu is the spillover).
        candidates = sorted(
            self.devices.values(),
            key=lambda d: (d.name == "cpu", d.name),
        )
        for dev in candidates:
            if dev.can_fit(need_bytes):
                if dev.name != "cpu":
                    dev.reserved += need_bytes
                return dev
        return None

    def release(self, name: str, need_bytes: int) -> None:
        dev = self.devices.get(name)
        if dev is not None and dev.name != "cpu":
            dev.reserved = max(0, dev.reserved - need_bytes)

    # ---- inspection -------------------------------------------------------

    def snapshot(self) -> list[dict[str, Any]]:
        out = []
        for dev in self.devices.values():
            out.append({
                "name": dev.name,
                "mem_total": dev.mem_total,
                "mem_free": dev.mem_free,
                "reserved": dev.reserved,
                "available": dev.available,
                "capability": dev.capability,
                "healthy": dev.healthy,
            })
        return out

    def healthy_device_count(self) -> int:
        return sum(1 for d in self.devices.values() if d.healthy)


def _probe_cuda() -> list[Device]:
    """Best-effort CUDA probe via torch (lazy import, never raises)."""
    try:
        import torch
    except Exception:
        return []
    try:
        if not torch.cuda.is_available():
            return []
    except Exception:
        return []
    devices: list[Device] = []
    for i in range(torch.cuda.device_count()):
        try:
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            capability = f"sm_{props.major}{props.minor}"
            devices.append(Device(
                name=f"cuda:{i}",
                mem_total=int(total),
                mem_free=int(free),
                capability=capability,
                healthy=True,
            ))
        except Exception:
            devices.append(Device(name=f"cuda:{i}", mem_total=0, mem_free=0, healthy=False))
    return devices

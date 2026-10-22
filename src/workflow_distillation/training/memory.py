"""Host RAM admission and cleanup, shared by all hardware profiles."""

import math
from pathlib import Path

GIB = 2**30
MIB = 2**20


def cgroup_memory(root=Path("/sys/fs/cgroup")):
    from contextlib import suppress

    result = {}
    for filename, key in [("memory.current", "current_bytes"), ("memory.max", "limit_bytes")]:
        with suppress(OSError, ValueError):
            text = (root / filename).read_text().strip()
            result[key] = None if text == "max" else int(text)
    with suppress(OSError, ValueError):
        result["events"] = dict(
            (k, int(v))
            for k, v in (line.split() for line in (root / "memory.events").read_text().splitlines())
        )
    with suppress(OSError, ValueError):
        result["stat"] = dict(
            (k, int(v))
            for k, v in (line.split() for line in (root / "memory.stat").read_text().splitlines())
        )
    if result.get("current_bytes") is not None and result.get("limit_bytes") is not None:
        return result
    for filename, key in [
        ("memory.usage_in_bytes", "current_bytes"),
        ("memory.limit_in_bytes", "limit_bytes"),
    ]:
        with suppress(OSError, ValueError):
            value = int((root / "memory" / filename).read_text())
            result[key] = value if value < 2**60 else None
    with suppress(OSError, ValueError):
        oom = dict(
            line.split()
            for line in (root / "memory" / "memory.oom_control").read_text().splitlines()
        )
        result["events"] = {"oom_kill": int(oom.get("oom_kill", 0))}
    with suppress(OSError, ValueError):
        stat = dict(
            (k, int(v))
            for k, v in (
                line.split() for line in (root / "memory" / "memory.stat").read_text().splitlines()
            )
        )
        result["stat"] = {
            key: stat[source]
            for key, source in (
                ("file", "total_cache"),
                ("shmem", "total_shmem"),
                ("file_dirty", "total_dirty"),
                ("file_writeback", "total_writeback"),
            )
            if source in stat
        }
    return result


def activation_bytes(tokens):
    # Same conservative per-trajectory estimator as the historical memory policy.
    return HostMemoryPolicy().activation_bytes(tokens)


def execution_mode(predicted, memory, *, allow_serial):
    current, limit = memory.get("current_bytes"), memory.get("limit_bytes")
    if current is None or limit is None:
        raise MemoryError("Host RAM limit or current usage unavailable")
    reserve = max(3 * 2**30, int(0.1 * limit))
    if current + sum(predicted) + reserve < limit:
        return "parallel"
    if allow_serial and current + max(predicted) + reserve < limit:
        return "serial"
    raise MemoryError("Host RAM precheck rejected the batch before forward")


def clear_inactive_cache(torch):
    import gc

    torch.cuda.synchronize()
    gc.collect()
    cleanup = getattr(torch._C, "_host_emptyCache", None)
    if cleanup is None:
        raise RuntimeError("PyTorch lacks pinned host-cache cleanup required by this profile")
    cleanup()


class HostMemoryPolicy:
    def __init__(self, *, reserve_bytes=3 * GIB, multiplier=1.15):
        if reserve_bytes < 0 or multiplier < 1 or not math.isfinite(multiplier):
            raise ValueError("Invalid host memory safety policy")
        self.reserve_bytes = reserve_bytes
        self.multiplier = multiplier

    @staticmethod
    def _raw_activation_bytes(tokens):
        if type(tokens) is not int or not 2 <= tokens < 30000:
            raise ValueError("Expected a complete trajectory below 30000 tokens")
        # FP16 hidden width 2560. The pinned allocator rounds each input to a
        # power-of-two size class. Observed envelope: 36 blocks plus small tensors.
        # In particular, 26214 -> 26215 tokens doubles the dominant size class.
        bucket = 1 << (tokens * 2560 * 2 - 1).bit_length()
        return 36 * bucket + 64 * MIB

    def activation_bytes(self, tokens):
        return math.ceil(self._raw_activation_bytes(tokens) * self.multiplier)

    def observe(self, tokens, allocated):
        if allocated is not None and allocated > 0:
            self.multiplier = max(
                self.multiplier, 1.15 * allocated / self._raw_activation_bytes(tokens)
            )

    def decide(self, lengths, memory):
        if len(lengths) != 2:
            raise ValueError("Adaptive execution requires two rank-local trajectories")
        predicted = [self.activation_bytes(n) for n in lengths]
        current, limit = memory.get("current_bytes"), memory.get("limit_bytes")
        decision = {
            "mode": "stop",
            "tokens": lengths,
            "activation_estimates_bytes": predicted,
            "reserve_bytes": self.reserve_bytes,
            "limit_bytes": limit,
            "multiplier": self.multiplier,
            "reason": "missing_memory_limit",
        }
        if current is None or limit is None or limit <= 0:
            return decision
        stat = memory.get("stat", {})
        # shmem is already INCLUDED in file; it must never be counted as
        # reclaimable disk cache. Retain dirty/writeback pages conservatively.
        reclaimable = 0
        if {"file", "shmem", "file_dirty", "file_writeback"} <= stat.keys():
            reclaimable = max(
                0, stat["file"] - stat["shmem"] - stat["file_dirty"] - stat["file_writeback"]
            )
        baseline = max(0, current - reclaimable)
        reserve = max(self.reserve_bytes, math.ceil(limit * 0.10))
        parallel, serial = baseline + sum(predicted), baseline + max(predicted)
        decision.update(
            resident_baseline_bytes=baseline,
            reserve_bytes=reserve,
            predicted_parallel_bytes=parallel,
            predicted_serial_bytes=serial,
        )
        if parallel + reserve <= limit:
            decision.update(mode="parallel", reason="parallel_has_headroom")
        elif serial + reserve <= limit:
            decision.update(mode="serial", reason="only_serial_has_headroom")
        else:
            decision["reason"] = "single_sample_lacks_headroom"
        return decision

"""Hardware admission for the 32GB RTX4080 pair; no numerical trainer changes."""

from contextlib import suppress
from pathlib import Path


def validate_4080_pair(devices):
    if len(devices) != 2 or any(
        d["name"] != "NVIDIA GeForce RTX 4080" or d["total_memory"] < 30 * 2**30
        for d in devices
    ):
        raise ValueError("Expected exactly two RTX4080 GPUs with at least 30 GiB each")
    return devices


def boundary_logger(emit, stop_file, stop_at_step):
    """Request the existing worker's normal save/stop after a logged update."""
    def logged(event):
        emit(event)
        if (stop_at_step is not None and event.get("event") == "step"
                and event["global_step"] >= stop_at_step):
            stop_file.touch()
    return logged


def compatible_cgroup_memory(root=Path("/sys/fs/cgroup"), *, original):
    """Read scoped cgroup v1 counters when the existing v2 reader finds none."""
    value = original(root)
    if value.get("current_bytes") is not None and value.get("limit_bytes") is not None:
        return value
    memory = Path(root) / "memory"
    for name, key in (("usage_in_bytes", "current_bytes"),
                      ("limit_in_bytes", "limit_bytes"),
                      ("max_usage_in_bytes", "peak_bytes")):
        with suppress(OSError, ValueError):
            number = int((memory / ("memory." + name)).read_text())
            value[key] = number if number < 2**60 else None
    with suppress(OSError, ValueError):
        stat = dict(line.split() for line in (memory / "memory.stat").read_text().splitlines())
        mapping = {"rss": "anon", "cache": "file", "shmem": "shmem",
                   "dirty": "file_dirty", "writeback": "file_writeback"}
        value["stat"] = {
            target: int(stat.get("total_" + source, stat.get(source, "0")))
            for source, target in mapping.items()
        }
    with suppress(OSError, ValueError, KeyError):
        lines = (memory / "memory.oom_control").read_text().splitlines()
        oom = dict(line.split() for line in lines)
        value["events"] = {"oom_kill": int(oom["oom_kill"])}
    return value


def install_cgroup_compatibility():
    import host_memory_recovery

    original = host_memory_recovery.cgroup_memory
    host_memory_recovery.cgroup_memory = lambda root=Path("/sys/fs/cgroup"): (
        compatible_cgroup_memory(root, original=original)
    )

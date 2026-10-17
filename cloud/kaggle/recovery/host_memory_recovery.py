"""Operational cache cleanup and telemetry; no model/config/source mutation."""
from __future__ import annotations

import gc
import json
import os
import signal
import subprocess
import threading
import time
from contextlib import suppress
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path

_OWNED_PROCESSES = {}


def cgroup_memory(root=Path('/sys/fs/cgroup')):
    root = Path(root)
    result = {}
    for file, key in [('memory.current', 'current_bytes'), ('memory.max', 'limit_bytes'),
                      ('memory.peak', 'peak_bytes'), ('pids.current', 'pids_current'),
                      ('pids.max', 'pids_limit')]:
        try:
            value = (root / file).read_text().strip()
            result[key] = None if value == 'max' else int(value)
        except (OSError, ValueError):
            pass
    with suppress(OSError, ValueError, IndexError):
        result['events'] = {line.split()[0]: int(line.split()[1])
                            for line in (root / 'memory.events').read_text().splitlines()}
    with suppress(OSError, ValueError, IndexError):
        result['stat'] = {line.split()[0]: int(line.split()[1])
                          for line in (root / 'memory.stat').read_text().splitlines()
                          if line.split()[0] in {'anon', 'file', 'shmem', 'kernel',
                                                'unevictable', 'slab', 'pagetables'}}
    with suppress(OSError, ValueError, IndexError):
        result['pids_events'] = {line.split()[0]: int(line.split()[1])
                                for line in (root / 'pids.events').read_text().splitlines()}
    return result


def memory_snapshot(torch=None):
    result = {'pid': os.getpid(), 'rank': os.environ.get('LOCAL_RANK'),
              'cgroup': cgroup_memory()}
    with suppress(OSError, ValueError, IndexError):
        result['process_kib'] = {line.split(':')[0]: int(line.split()[1])
                                 for line in Path('/proc/self/status').read_text().splitlines()
                                 if line.startswith(('VmRSS:', 'VmPin:', 'VmLck:', 'VmHWM:'))}
    if torch is not None and torch.cuda.is_initialized():
        stats = torch.cuda.memory.host_memory_stats()
        result['pinned'] = {k: stats[k] for k in ['allocated_bytes.current',
                           'active_bytes.current', 'allocated_bytes.peak'] if k in stats}
        # CUDA's current device is thread-local; this monitor is not the worker thread.
        device = int(os.environ.get('LOCAL_RANK', '0'))
        result['cuda_allocated_bytes'] = torch.cuda.memory_allocated(device)
        result['cuda_reserved_bytes'] = torch.cuda.memory_reserved(device)
    return result


def emit(event):
    event = {'time': datetime.now(UTC).isoformat(), 'run_name': 'first run', **event}
    folder = Path(os.environ.get(
        'FIRST_RUN_RECOVERY_LOG_DIR', '/kaggle/working/first-run-recovery'))
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f'memory-{os.getpid()}.jsonl').open('a') as stream:
        stream.write(json.dumps(event) + '\n')
    # Keep full diagnostics on disk; repeated snapshots can drown out step logs.
    if event['event'] == 'memory_watch':
        return
    if event['event'] == 'progress_summary':
        phases = ', '.join(f'rank{k}={v}' for k, v in event['phases'].items()) or 'starting'
        memory = event.get('cgroup', {})
        current, limit = memory.get('current_bytes'), memory.get('limit_bytes')
        ram = (f'{current / 2**30:.1f}/{limit / 2**30:.1f} GiB'
               if current is not None and limit else 'unknown')
        line = (f"[{event['time']}] first run | step {event['global_step']}/"
                f"{event['total_steps'] or '?'} | trajectories {event['samples_consumed']}/"
                f"{event['total_samples'] or '?'} | checkpoint {event['checkpoint_step']} | "
                f"RAM {ram} | {phases} | last rank event "
                f"{event['progress_age_seconds']:.0f}s ago")
        with (folder / 'progress.log').open('a') as stream:
            stream.write(line + '\n')
        temporary = folder / 'status.json.tmp'
        temporary.write_text(json.dumps(event) + '\n')
        temporary.replace(folder / 'status.json')
        print(line, flush=True)
        return
    console = {key: value for key, value in event.items() if key not in {'before', 'after'}}
    print(json.dumps(console), flush=True)


def clear_host_cache(torch, phase, emit_event):
    """Release only inactive pinned allocations after completed CUDA transfers."""
    torch.cuda.synchronize()
    gc.collect()
    before = memory_snapshot(torch)
    torch._C._host_emptyCache()
    emit_event({'event': 'host_cache_cleared', 'phase': phase, 'before': before,
                'after': memory_snapshot(torch)})


def install_cache_cleanup(torch, emit_event=emit):
    """Release completed saved-activation pinned blocks before the next forward."""
    original = torch.autograd.graph.save_on_cpu.__enter__
    if getattr(original, '_first_run_host_cache_cleanup', False):
        return
    if not callable(getattr(torch._C, '_host_emptyCache', None)):
        raise RuntimeError('This PyTorch does not expose _host_emptyCache; recovery not installed')

    def enter(context):
        clear_host_cache(torch, 'before_forward', emit_event)
        return original(context)

    enter._first_run_host_cache_cleanup = True
    torch.autograd.graph.save_on_cpu.__enter__ = enter


def install_checkpoint_cleanup(torch, manager_class, emit_event=emit):
    """Free backward's inactive host cache on every rank before checkpoint copies.

    The existing save implementation retains all serialization, publication and
    rank agreement. Its source file and checkpoint identity remain unchanged.
    """
    original = manager_class.save
    if getattr(original, '_first_run_checkpoint_cleanup', False):
        return
    if not callable(getattr(torch._C, '_host_emptyCache', None)):
        raise RuntimeError('This PyTorch does not expose _host_emptyCache; recovery not installed')

    @wraps(original)
    def save(manager, *args, **kwargs):
        clear_host_cache(torch, 'before_checkpoint', emit_event)
        return original(manager, *args, **kwargs)

    save._first_run_checkpoint_cleanup = True
    manager_class.save = save


def install_torchrun_logs(log_dir):
    """Keep per-rank stdout/stderr in addition to the original console stream."""
    original = subprocess.Popen
    if getattr(original, '_first_run_torchrun_logs', False):
        return

    def popen(args, *popen_args, **kwargs):
        # Match Popen's public parameter name: torchrun workers use args=.
        command = args
        is_torchrun = False
        if isinstance(command, (list, tuple)) and '-m' in command:
            module = command.index('-m') + 1
            is_torchrun = module < len(command) and command[module] == 'torch.distributed.run'
            if (is_torchrun
                    and '--log-dir' not in command and '--tee' not in command):
                command = list(command)
                command[module + 1:module + 1] = [
                    '--log-dir', str(Path(log_dir) / 'torchrun'), '--tee', '3']
        return original(command, *popen_args, **kwargs)

    popen._first_run_torchrun_logs = True
    subprocess.Popen = popen


def _tail_events(path):
    try:
        with path.open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 65536))
            lines = stream.read().splitlines()
        result = []
        for line in lines:
            with suppress(ValueError, UnicodeDecodeError):
                value = json.loads(line)
                if isinstance(value, dict):
                    result.append(value)
        return result
    except OSError:
        return []


def progress_summary(directory):
    """Read bounded live-file tails, distinguishing completed updates from phases."""
    directory = Path(directory)
    launcher = _tail_events(directory / 'launcher.jsonl')
    metadata = next((e for e in reversed(launcher) if 'samples' in e), {})
    status = {'global_step': 0, 'samples_consumed': 0, 'checkpoint_step': 0,
              'total_samples': metadata.get('samples'),
              'total_steps': metadata.get('total_optimizer_steps'),
              'phases': {}, 'progress_age_seconds': 0}
    steps, samples, ages = [], [], []
    for rank in (0, 1):
        path = directory / f'rank{rank}.jsonl'
        events = _tail_events(path)
        if not events:
            continue
        status['phases'][str(rank)] = events[-1].get('event', 'unknown')
        with suppress(OSError):
            ages.append(max(0, time.time() - path.stat().st_mtime))
        completed = [e for e in events if e.get('event') in {
            'resumed', 'step', 'checkpoint_started', 'checkpoint_complete'}]
        if completed:
            steps.append(completed[-1].get('global_step', 0))
            samples.append(completed[-1].get('samples_consumed', 0))
        for event in completed:
            if event.get('event') in {'resumed', 'checkpoint_complete'}:
                status['checkpoint_step'] = max(status['checkpoint_step'], event['global_step'])
    status['global_step'] = min(steps, default=0)
    status['samples_consumed'] = min(samples, default=0)
    status['progress_age_seconds'] = max(ages, default=0)
    return status


def _process_identity(path):
    # comm may contain spaces or parentheses; fields after its last ')' are stable.
    fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    return int(fields[1]), int(fields[19])  # ppid, starttime (guards PID reuse).


def _track_descendants():
    processes = {}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            with suppress(OSError, ValueError, IndexError):
                processes[int(path.name)] = _process_identity(path)
    owned = {os.getpid()} | {
        pid for pid, started in _OWNED_PROCESSES.items()
        if pid in processes and processes[pid][1] == started}
    while True:
        children = {pid for pid, (parent, _) in processes.items()
                    if parent in owned and pid not in owned}
        if not children:
            break
        owned.update(children)
    for pid in owned - {os.getpid()}:
        _OWNED_PROCESSES[pid] = processes[pid][1]


def abort_training(event):
    """Stop descendants across sessions, retaining identities if a parent died."""
    try:
        emit(event)
    finally:
        try:
            with suppress(OSError):
                _track_descendants()
            for pid, started in list(_OWNED_PROCESSES.items()):
                with suppress(OSError, ValueError, IndexError):
                    if _process_identity(Path('/proc') / str(pid))[1] == started:
                        os.kill(pid, signal.SIGKILL)
        finally:
            os._exit(86)


def start_monitor(torch=None, interval=10, *, cgroup_root=Path('/sys/fs/cgroup'),
                  progress_dir=Path('/kaggle/working/first_run_recovery_project/'
                                    'outputs/training/first run'), status_interval=60):
    baseline = cgroup_memory(cgroup_root).get('events', {})
    owner = os.environ.get('FIRST_RUN_MONITOR_OWNER_PID') == str(os.getpid())

    def watch():
        last_status = -float('inf')
        while True:
            try:
                memory = cgroup_memory(cgroup_root)
                events = memory.get('events', {})
                # Torchrun workers can create independent sessions. Record their
                # identities while their parent is alive, not just its pgid.
                _track_descendants()
                if any(events.get(key, 0) > baseline.get(key, 0)
                       for key in ('oom_kill', 'oom_group_kill')):
                    abort_training({'event': 'host_oom_detected', 'cgroup': memory,
                                    'baseline_events': baseline, 'exit_code': 86})
                if owner and time.monotonic() - last_status >= status_interval:
                    emit({'event': 'progress_summary', **progress_summary(progress_dir),
                          'cgroup': memory})
                    last_status = time.monotonic()
                # Check the cgroup before calling any CUDA allocator API. The
                # non-CUDA launcher remains able to abort a blocked worker.
                emit({'event': 'memory_watch', **memory_snapshot(torch)})
            except Exception as exc:
                print(json.dumps({'event': 'memory_watch_error', 'type': type(exc).__name__}),
                      flush=True)
            threading.Event().wait(interval)

    thread = threading.Thread(target=watch, daemon=True, name='memory-watch')
    thread.start()
    return thread

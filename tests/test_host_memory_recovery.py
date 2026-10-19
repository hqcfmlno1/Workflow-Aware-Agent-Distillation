"""Recovery memory hygiene must preserve training arithmetic and RNG."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_recovery():
    spec = importlib.util.spec_from_file_location(
        "host_memory_recovery", ROOT / "cloud/kaggle/recovery/host_memory_recovery.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cleanup_preserves_loss_gradients_and_rng(monkeypatch):
    torch = pytest.importorskip("torch")
    recovery = load_recovery()
    torch.manual_seed(123)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(.2),
                                torch.nn.Linear(8, 1))
    x = torch.arange(28, dtype=torch.float32).view(7, 4) / 28
    before = torch.get_rng_state().clone()
    with torch.autograd.graph.save_on_cpu(pin_memory=False):
        expected_loss = model(x).square().mean()
    expected_loss.backward()
    expected_grads = [p.grad.clone() for p in model.parameters()]
    expected_rng = torch.get_rng_state().clone()
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(before)
    calls, events = [], []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("synchronize"))
    monkeypatch.setattr(torch._C, "_host_emptyCache", lambda: calls.append("host_empty"),
                        raising=False)
    # Register the attribute in pytest's undo stack before installation mutates it.
    original = torch.autograd.graph.save_on_cpu.__enter__
    monkeypatch.setattr(torch.autograd.graph.save_on_cpu, "__enter__", original)
    recovery.install_cache_cleanup(torch, events.append)
    with torch.autograd.graph.save_on_cpu(pin_memory=False):
        actual_loss = model(x).square().mean()
    actual_loss.backward()
    assert calls == ["synchronize", "host_empty"]
    assert torch.equal(actual_loss, expected_loss)
    assert all(torch.equal(p.grad, g)
               for p, g in zip(model.parameters(), expected_grads, strict=True))
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert events[0]["event"] == "host_cache_cleared"


def test_cgroup_snapshot_reports_oom_and_finite_limit(tmp_path):
    recovery = load_recovery()
    (tmp_path / "memory.current").write_text("123456\n")
    (tmp_path / "memory.max").write_text("30000000\n")
    (tmp_path / "memory.events").write_text("low 0\nhigh 0\noom 2\noom_kill 1\n")
    sample = recovery.cgroup_memory(tmp_path)
    assert sample == {"current_bytes": 123456, "limit_bytes": 30000000,
                      "events": {"low": 0, "high": 0, "oom": 2, "oom_kill": 1}}
    (tmp_path / "memory.max").write_text("max\n")
    assert recovery.cgroup_memory(tmp_path)["limit_bytes"] is None


def test_startup_fails_closed_when_cleanup_is_unavailable(tmp_path):
    import os
    import subprocess
    import sys

    (tmp_path / 'torch.py').write_text(
        'from types import SimpleNamespace\n'
        '_C = SimpleNamespace()\n'
        'autograd = SimpleNamespace(graph=SimpleNamespace('
        'save_on_cpu=SimpleNamespace(__enter__=lambda self: None)))\n')
    hooks = ROOT / 'cloud/kaggle/recovery'
    result = subprocess.run([sys.executable, '-c', 'print("TRAIN_WOULD_START")'],
                            env={**os.environ, 'PYTHONPATH': f'{hooks}:{tmp_path}',
                                 'FIRST_RUN_HOST_CACHE_RECOVERY': '1', 'LOCAL_RANK': '0'},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 78
    assert 'TRAIN_WOULD_START' not in result.stdout
    assert 'RuntimeError' in result.stderr


def test_monitor_uses_explicit_rank_device(monkeypatch):
    from types import SimpleNamespace

    recovery = load_recovery()
    monkeypatch.setenv('LOCAL_RANK', '1')
    devices = []

    def allocated(device):
        devices.append(device)
        return 123

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(
        is_initialized=lambda: True,
        memory=SimpleNamespace(host_memory_stats=dict),
        memory_allocated=allocated, memory_reserved=allocated))
    snapshot = recovery.memory_snapshot(fake_torch)
    assert devices == [1, 1]
    assert snapshot['cuda_allocated_bytes'] == 123


def test_torchrun_logging_hook_preserves_worker_arguments(monkeypatch, tmp_path):
    import subprocess

    recovery = load_recovery()
    commands = []
    monkeypatch.setattr(subprocess, 'Popen', lambda command, **kwargs: commands.append(command))
    recovery.install_torchrun_logs(tmp_path)
    original = ['python', '-u', '-m', 'torch.distributed.run', '--nproc_per_node=2',
                'train.py', '--run-name', 'first run']
    subprocess.Popen(original)
    subprocess.Popen(['python', '-m', 'pip', '--version'])
    assert commands[0] == original[:4] + ['--log-dir', str(tmp_path / 'torchrun'),
                                         '--tee', '3'] + original[4:]
    assert commands[1] == ['python', '-m', 'pip', '--version']
    assert '--log-dir' not in original


def test_torchrun_hook_accepts_popen_args_keyword(monkeypatch, tmp_path):
    """PyTorch 2.11 launches workers with Popen(args=...), including env/cwd."""
    import os
    import subprocess
    import sys

    recovery = load_recovery()
    monkeypatch.setattr(subprocess, 'Popen', subprocess.Popen)
    recovery.install_torchrun_logs(tmp_path)
    command = [sys.executable, '-c',
               'import os; print(os.environ["WORKER_MARKER"]); print(os.getcwd())']
    with subprocess.Popen(args=command, cwd=tmp_path,
                          env={**os.environ, 'WORKER_MARKER': 'rank-worker'},
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as child:
        stdout, stderr = child.communicate(timeout=10)
    assert child.returncode == 0
    assert stdout.splitlines() == ['rank-worker', str(tmp_path)]
    assert stderr == ''


def test_torchrun_retains_real_rank_failure(monkeypatch, tmp_path):
    import os
    import subprocess
    import sys

    pytest.importorskip('torch')
    worker = tmp_path / 'fail_worker.py'
    worker.write_text(
        'import os\n'
        'if os.environ["LOCAL_RANK"] == "1":\n'
        '    raise RuntimeError("intentional model-loader failure")\n')
    recovery = load_recovery()
    original = subprocess.Popen
    monkeypatch.setattr(subprocess, 'Popen', original)
    recovery.install_torchrun_logs(tmp_path)
    result = subprocess.run(
        [sys.executable, '-m', 'torch.distributed.run', '--standalone',
         '--nproc_per_node=2', str(worker)], capture_output=True, text=True,
        env={**os.environ, 'OMP_NUM_THREADS': '1'}, timeout=30)
    assert result.returncode != 0
    errors = list((tmp_path / 'torchrun').glob('**/1/stderr.log'))
    assert len(errors) == 1
    assert 'intentional model-loader failure' in errors[0].read_text()


def test_checkpoint_cleanup_precedes_serialization_and_preserves_resume(monkeypatch, tmp_path):
    torch = pytest.importorskip('torch')
    from test_training_checkpoints import load, save, setup_training, step

    from workflow_distillation import training_checkpoints as checkpoints

    recovery = load_recovery()
    training = setup_training()
    step(training)
    calls, events = [], []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: calls.append('synchronize'))
    monkeypatch.setattr(torch._C, '_host_emptyCache', lambda: calls.append('host_empty'),
                        raising=False)
    original_write = checkpoints._torch_file

    def write(path, state):
        # Both optimizer/RNG and adapter serialization need host memory headroom.
        assert calls == ['synchronize', 'host_empty']
        return original_write(path, state)

    monkeypatch.setattr(checkpoints, '_torch_file', write)
    monkeypatch.setattr(checkpoints.CheckpointManager, 'save', checkpoints.CheckpointManager.save)
    recovery.install_checkpoint_cleanup(torch, checkpoints.CheckpointManager, events.append)
    # Installing twice must not stack cache flushes.
    recovery.install_checkpoint_cleanup(torch, checkpoints.CheckpointManager, events.append)
    manager = checkpoints.CheckpointManager(tmp_path, {'data': 'unchanged'})
    path = save(manager, training, 1)
    expected_loss = step(training)
    expected_weight = training[0].adapter.weight.detach().clone()
    resumed = setup_training()
    assert load(manager, resumed, path) == {'global_step': 1, 'next_batch': 1}
    actual_loss = step(resumed)
    torch.testing.assert_close(actual_loss, expected_loss, atol=0, rtol=0)
    torch.testing.assert_close(resumed[0].adapter.weight, expected_weight, atol=0, rtol=0)
    assert events[0]['phase'] == 'before_checkpoint'


def test_memory_telemetry_stays_in_file_without_flooding_console(monkeypatch, tmp_path, capsys):
    import json

    recovery = load_recovery()
    monkeypatch.setenv('FIRST_RUN_RECOVERY_LOG_DIR', str(tmp_path))
    recovery.emit({'event': 'memory_watch', 'cgroup': {'current_bytes': 123}})
    recovery.emit({'event': 'host_cache_cleared', 'phase': 'before_checkpoint',
                   'before': {'pinned': {'allocated_bytes.current': 999}},
                   'after': {'pinned': {'allocated_bytes.current': 0}}})
    recovery.emit({'event': 'host_cache_recovery_installed', 'rank': '0'})
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event['event'] for event in output] == [
        'host_cache_cleared', 'host_cache_recovery_installed']
    assert 'before' not in output[0]  # Full snapshots are available in the file.
    records = [json.loads(line) for line in next(tmp_path.glob('memory-*.jsonl'))
               .read_text().splitlines()]
    assert len(records) == 3
    assert records[0]['cgroup']['current_bytes'] == 123
    assert records[1]['before']['pinned']['allocated_bytes.current'] == 999


@pytest.mark.parametrize('adaptive', [False, True])
def test_worker_startup_installs_both_hooks_and_suppresses_hub_progress(tmp_path, adaptive):
    import os
    import subprocess
    import sys

    (tmp_path / 'torch.py').write_text(
        'from types import SimpleNamespace\n'
        '__version__ = "cpu-startup-test"\n'
        '_C = SimpleNamespace(_host_emptyCache=lambda: None)\n'
        'cuda = SimpleNamespace(is_initialized=lambda: False)\n'
        'autograd = SimpleNamespace(graph=SimpleNamespace('
        'save_on_cpu=SimpleNamespace(__enter__=lambda self: None)))\n')
    hooks = ROOT / 'cloud/kaggle/recovery'
    result = subprocess.run(
        [sys.executable, '-c',
         'import os, torch; '
         'from workflow_distillation.training_checkpoints import CheckpointManager; '
         'assert torch.autograd.graph.save_on_cpu.__enter__._first_run_host_cache_cleanup; '
         'assert CheckpointManager.save._first_run_checkpoint_cleanup; '
         'assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"; '
         'from workflow_distillation import qlora_runtime; '
         'assert bool(getattr(qlora_runtime.build_replica, '
         '"_first_run_adaptive_execution", False)) == '
         '(os.environ.get("FIRST_RUN_ADAPTIVE_MEMORY") == "1"); '
         'print("HOOKS_READY")'],
        env={**os.environ, 'PYTHONPATH': f'{hooks}:{tmp_path}:{ROOT / "src"}',
             'FIRST_RUN_HOST_CACHE_RECOVERY': '1', 'LOCAL_RANK': '0',
             'FIRST_RUN_ADAPTIVE_MEMORY': str(int(adaptive)),
             'FIRST_RUN_RECOVERY_LOG_DIR': str(tmp_path / 'logs')},
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert 'HOOKS_READY' in result.stdout


def test_progress_summary_counts_completed_updates_and_confirmed_checkpoints(
        monkeypatch, tmp_path, capsys):
    import json

    recovery = load_recovery()
    (tmp_path / 'launcher.jsonl').write_text(json.dumps({
        'event': 'dataset_verified', 'samples': 11527, 'total_optimizer_steps': 5764}) + '\n')
    for rank in (0, 1):
        events = [
            {'event': 'resumed', 'global_step': 200, 'samples_consumed': 400,
             'checkpoint': '/checkpoints/checkpoint-00000200'},
            {'event': 'step', 'global_step': 205, 'samples_consumed': 410},
            {'event': 'checkpoint_started', 'global_step': 205, 'samples_consumed': 410},
        ]
        (tmp_path / f'rank{rank}.jsonl').write_text(
            ''.join(json.dumps(event) + '\n' for event in events))
    status = recovery.progress_summary(tmp_path)
    assert status['global_step'] == 205
    assert status['samples_consumed'] == 410
    assert status['total_samples'] == 11527
    assert status['checkpoint_step'] == 200  # A save that started is not a successful backup.
    assert status['phases'] == {'0': 'checkpoint_started', '1': 'checkpoint_started'}
    assert status['progress_age_seconds'] >= 0
    with (tmp_path / 'rank0.jsonl').open('a') as stream:
        stream.write(json.dumps({'event': 'checkpoint_complete', 'global_step': 205,
                                 'samples_consumed': 410}) + '\n')
        stream.write(json.dumps({'event': 'forward_started', 'global_step': 206}) + '\n')
        stream.write('{"event":')  # Reading a live, partially written line must be safe.
    status = recovery.progress_summary(tmp_path)
    assert status['global_step'] == 205  # Forward 206 is not a completed update.
    assert status['checkpoint_step'] == 205
    monkeypatch.setenv('FIRST_RUN_RECOVERY_LOG_DIR', str(tmp_path / 'telemetry'))
    recovery.emit({'event': 'progress_summary', **status,
                   'cgroup': {'current_bytes': 25 * 2**30, 'limit_bytes': 30 * 2**30}})
    console = capsys.readouterr().out.splitlines()
    assert len(console) == 1
    assert 'step 205/5764' in console[0]
    assert 'trajectories 410/11527' in console[0]
    assert 'checkpoint 205' in console[0]
    assert 'RAM 25.0/30.0 GiB' in console[0]
    assert 'rank0=forward_started' in console[0]
    record = json.loads(next((tmp_path / 'telemetry').glob('memory-*.jsonl')).read_text())
    assert record['samples_consumed'] == 410
    assert (tmp_path / 'telemetry/progress.log').read_text().splitlines() == console
    assert json.loads((tmp_path / 'telemetry/status.json').read_text()) == record


@pytest.mark.parametrize('orphan_workers', [False, True])
@pytest.mark.parametrize('failure', ['host_oom', 'rank_failed'])
def test_oom_watchdog_exits_and_kills_the_owned_torchrun_group(
        tmp_path, orphan_workers, failure):
    """A dead rank must not leave a live peer consuming quota after a new OOM kill."""
    import os
    import signal
    import subprocess
    import sys
    import time
    from contextlib import suppress

    pytest.importorskip('torch')
    cgroup = tmp_path / 'cgroup'
    cgroup.mkdir()
    events = cgroup / 'memory.events'
    events.write_text('oom 0\noom_kill 1\n')  # Historical kill must not stop this new run.
    worker = tmp_path / 'worker.py'
    worker.write_text(
        'import os,time\nfrom pathlib import Path\n'
        'Path("worker-"+os.environ["LOCAL_RANK"]).write_text(str(os.getpid()))\n'
        'time.sleep(60)\n')
    script = tmp_path / 'supervisor.py'
    script.write_text(
        'import os,subprocess,sys,time\n'
        'from pathlib import Path\n'
        'from host_memory_recovery import install_torchrun_logs,start_monitor,_OWNED_PROCESSES\n'
        'os.environ["FIRST_RUN_MONITOR_OWNER_PID"]=str(os.getpid())\n'
        'install_torchrun_logs(Path("logs"))\n'
        'start_monitor(interval=.02,cgroup_root=Path("cgroup"),progress_dir=Path("progress"))\n'
        'child=subprocess.Popen([sys.executable,"-m","torch.distributed.run","--standalone",'
        '"--nproc_per_node=2","worker.py"],start_new_session=True)\n'
        'Path("torchrun.pid").write_text(str(child.pid))\n'
        'while True:\n'
        '    files=[Path("worker-0"),Path("worker-1")]\n'
        '    if all(f.exists() and f.read_text() for f in files):\n'
        '        if all(int(f.read_text()) in _OWNED_PROCESSES for f in files):\n'
        '            Path("workers-tracked").touch(); break\n'
        '    time.sleep(.01)\n'
        'time.sleep(60)\n')
    with (tmp_path / 'console.log').open('w') as output:
        parent = subprocess.Popen(
            [sys.executable, str(script)], cwd=tmp_path,
            env={**os.environ, 'PYTHONPATH': str(ROOT / 'cloud/kaggle/recovery'),
                 'FIRST_RUN_HOST_CACHE_RECOVERY': '0', 'OMP_NUM_THREADS': '1',
                 'FIRST_RUN_RECOVERY_LOG_DIR': str(tmp_path / 'logs')},
            stdout=output, stderr=output, start_new_session=True)
        worker_pids = []
        try:
            deadline = time.monotonic() + 20
            while not (tmp_path / 'workers-tracked').exists():
                assert parent.poll() is None, (tmp_path / 'console.log').read_text()
                assert time.monotonic() < deadline, (tmp_path / 'console.log').read_text()
                time.sleep(.05)
            worker_pids = [int((tmp_path / f'worker-{rank}').read_text()) for rank in (0, 1)]
            assert parent.poll() is None
            if orphan_workers:
                # The OOM victim may be torchrun rather than a worker. Its peers
                # have separate sessions and must still be stopped after reparenting.
                os.kill(int((tmp_path / 'torchrun.pid').read_text()), signal.SIGKILL)
            if failure == 'host_oom':
                events.write_text('oom 1\noom_kill 2\n')
            else:
                progress = tmp_path / 'progress'
                progress.mkdir()
                (progress / 'rank0.jsonl').write_text(
                    '{"event":"failed","rank":0,"error_type":"OutOfMemoryError"}\n')
            assert parent.wait(timeout=5) == 86
            for pid in worker_pids:
                deadline = time.monotonic() + 3
                while True:
                    try:
                        state = Path(f'/proc/{pid}/stat').read_text().split()[2]
                    except (FileNotFoundError, ProcessLookupError):
                        break
                    if state == 'Z':
                        break
                    assert time.monotonic() < deadline, 'Worker survived the OOM watchdog'
                    time.sleep(.02)
            expected_event = ('host_oom_detected' if failure == 'host_oom'
                              else 'rank_failure_detected')
            assert expected_event in (tmp_path / 'console.log').read_text()
        finally:
            if parent.poll() is None:
                os.killpg(parent.pid, signal.SIGKILL)
                parent.wait()
            for pid in worker_pids:
                with suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)

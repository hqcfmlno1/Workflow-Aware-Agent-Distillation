import json
import signal
import subprocess
import sys
import time

import pytest


def test_progress_reader_tolerates_partial_json_and_reports_both_rank_phases(tmp_path):
    from workflow_distillation.training.monitor import progress_summary

    (tmp_path / "rank0.jsonl").write_text(
        "\n".join(
            json.dumps(e)
            for e in [
                {
                    "event": "resumed",
                    "global_step": 1028,
                    "next_batch": 1028,
                    "samples_consumed": 2692,
                },
                {"event": "forward_started", "global_step": 1029},
                {"event": "step", "global_step": 1029, "samples_consumed": 2696, "loss": 0.75},
            ]
        )
        + '\n{"unfinished":'
    )
    (tmp_path / "rank1.jsonl").write_text(
        json.dumps({"event": "backward_started", "global_step": 1030}) + "\n"
    )
    result = progress_summary(tmp_path)
    assert result["global_step"] == 1029
    assert result["samples_consumed"] == 2696
    assert result["checkpoint_step"] == 1028
    assert result["loss"] == 0.75
    assert result["phases"] == {"0": "step", "1": "backward_started"}


def test_monitor_observes_failed_rank_and_kills_only_owned_worker_group(tmp_path):
    from workflow_distillation.training.monitor import RunMonitor

    worker = None
    try:
        with RunMonitor(tmp_path, interval=0.02) as monitor:
            worker = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
            )
            monitor.register_worker(worker.pid)
            (tmp_path / "rank0.jsonl").write_text(
                json.dumps({"event": "failed", "error_type": "OutOfMemoryError"}) + "\n"
            )
            deadline = time.monotonic() + 3
            while worker.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            assert worker.poll() == -signal.SIGKILL
            assert monitor.failure["event"] == "rank_failure_detected"
    finally:
        if worker is not None:
            if worker.poll() is None:
                worker.kill()
            worker.wait()


def test_progress_totals_survive_bounded_log_tail(tmp_path):
    from workflow_distillation.training.monitor import progress_summary

    (tmp_path / "preflight.json").write_text(
        json.dumps({"samples": 9000, "total_optimizer_steps": 2250})
    )
    (tmp_path / "rank0.jsonl").write_text(
        json.dumps({"event": "step", "global_step": 1200, "samples_consumed": 4000, "loss": 0.4})
        + "\n"
    )
    state = progress_summary(tmp_path)
    assert state["total_steps"] == 2250
    assert state["total_samples"] == 9000


def test_monitor_kills_detached_descendant_and_preserves_prior_child(tmp_path):
    from pathlib import Path

    from workflow_distillation.training.monitor import RunMonitor

    protected = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    worker = None
    descendant = None
    unrelated = None
    try:
        with RunMonitor(tmp_path, interval=0.02) as monitor:
            code = (
                "import subprocess,sys,time,pathlib; "
                "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],"
                "start_new_session=True); "
                f"pathlib.Path({str(tmp_path / 'descendant')!r}).write_text(str(p.pid)); "
                "time.sleep(60)"
            )
            worker = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
            monitor.register_worker(worker.pid)
            unrelated = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
            )
            deadline = time.monotonic() + 3
            while not (tmp_path / "descendant").exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            descendant = int((tmp_path / "descendant").read_text())
            (tmp_path / "rank0.jsonl").write_text(json.dumps({"event": "failed"}) + "\n")
            while worker.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            assert worker.poll() == -signal.SIGKILL
            assert monitor.failure is not None
            assert protected.poll() is None
            assert unrelated.poll() is None
            stat = Path(f"/proc/{descendant}/stat")
            deadline = time.monotonic() + 2
            while stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[0] != "Z":
                assert time.monotonic() < deadline, "Detached worker survived rank failure"
                time.sleep(0.02)
    finally:
        import os
        from contextlib import suppress

        if descendant:
            with suppress(ProcessLookupError):
                os.kill(descendant, signal.SIGKILL)
        for proc in (worker, protected, unrelated):
            if proc is not None:
                if proc.poll() is None:
                    proc.kill()
                proc.wait()


def test_logged_runner_registers_its_exact_process_and_preserves_output(tmp_path):
    from workflow_distillation.training.monitor import RunMonitor
    from workflow_distillation.training.process import run_logged_process

    with RunMonitor(tmp_path, interval=0.02) as monitor:
        result = run_logged_process(
            [sys.executable, "-u", "-c", "print('worker output')"],
            tmp_path / "run.log",
            timeout=5,
            monitor=monitor,
        )
        assert result.returncode == 0
        assert monitor.root_pid is not None
    assert (tmp_path / "run.log").read_text() == "worker output\n"


@pytest.mark.parametrize("exit_code", [0, 7, None])
def test_runner_cleans_detached_worker_when_launcher_exits_before_scan(tmp_path, exit_code):
    import os
    from contextlib import suppress
    from pathlib import Path

    from workflow_distillation.training.monitor import RunMonitor
    from workflow_distillation.training.process import run_logged_process

    child = None
    try:
        code = (
            "import subprocess,sys,pathlib; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],"
            "start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
            f"pathlib.Path({str(tmp_path / 'child')!r}).write_text(str(p.pid)); "
            + (f"sys.exit({exit_code})" if exit_code is not None else "import time; time.sleep(60)")
        )
        with RunMonitor(tmp_path, interval=20) as monitor:
            if exit_code != 0:
                error = (
                    subprocess.TimeoutExpired
                    if exit_code is None
                    else subprocess.CalledProcessError
                )
                with pytest.raises(error):
                    run_logged_process(
                        [sys.executable, "-c", code],
                        tmp_path / "run.log",
                        timeout=1 if exit_code is None else 5,
                        monitor=monitor,
                    )
            else:
                run_logged_process(
                    [sys.executable, "-c", code], tmp_path / "run.log", timeout=5, monitor=monitor
                )
        child = int((tmp_path / "child").read_text())
        stat = Path(f"/proc/{child}/stat")
        deadline = time.monotonic() + 1
        while stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[0] != "Z":
            assert time.monotonic() < deadline, "Early launcher exit leaked a detached worker"
            time.sleep(0.02)
    finally:
        if child:
            with suppress(ProcessLookupError):
                os.kill(child, signal.SIGKILL)

import subprocess
import sys
from pathlib import Path


def test_detached_launcher_prints_commands_without_starting_gpu():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "workflow_distillation.lora_fp16.launch",
            "--dry-run",
            "--resume",
            "latest",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--mode train" in result.stdout
    assert "--resume latest" in result.stdout
    assert "train.log" in result.stdout


def test_bootstrap_shell_syntax_and_paths():
    for path in Path("cloud/vast/rtx5090").glob("*.sh"):
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_group_launch_runs_same_profile_in_order_and_stops_on_failed_pilot(tmp_path):
    import json
    import os

    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    fake = venv / "bin/python"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['TEST_LAUNCH_CALLS'], 'a') as f: "
        "f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if '--mode' in sys.argv and sys.argv[sys.argv.index('--mode')+1] == 'pilot':\n"
        "    sys.exit(int(os.environ.get('TEST_PILOT_EXIT', '0')))\n"
    )
    fake.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    env = {
        **os.environ, "SFT5090_VENV": str(venv),
        "SFT5090_OUTPUT": str(tmp_path / "output"),
        "TEST_LAUNCH_CALLS": str(calls),
    }
    command = ["bash", "cloud/vast/rtx5090/run-group-loss.sh", "--epochs", "2"]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    stages = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(stages) == 4
    assert stages[0][1] == "workflow_distillation.lora_fp16.assets"
    assert stages[1][stages[1].index("--mode")+1] == "preflight"
    assert stages[2][stages[2].index("--mode")+1] == "pilot"
    assert stages[3][1] == "workflow_distillation.lora_fp16.launch"
    for stage in stages[1:]:
        assert stage[stage.index("--config")+1].endswith("/group_loss.json")
        assert stage[stage.index("--output")+1] == str(tmp_path / "output")
    assert "--resume" not in stages[3]
    assert stages[3][stages[3].index("--hub-repo")+1].endswith("-group-loss-run1")
    assert stages[3][stages[3].index("--hub-prefix")+1] == "fp16-tqs060-group-loss"
    for override in [
        ["--config", "old.json"], ["--output", "old-run"], ["--resume", "latest"],
        ["--mode", "train"], ["--no-hub"], ["--epochs", "not-an-int"],
    ]:
        calls.unlink()
        rejected = subprocess.run(
            ["bash", "cloud/vast/rtx5090/run-group-loss.sh", *override],
            env=env, capture_output=True, text=True,
        )
        assert rejected.returncode != 0, f"Unsafe fresh-run override accepted: {override}"
        assert not calls.exists()
        calls.touch()
    calls.unlink()
    failed = subprocess.run(
        command, env={**env, "TEST_PILOT_EXIT": "42"}, capture_output=True, text=True
    )
    assert failed.returncode == 42
    stages = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(stages) == 3
    # Re-running a used fresh output must fail before assets/pilot consume time.
    calls.unlink()
    (tmp_path / "output").mkdir()
    (tmp_path / "output/events.jsonl").write_text("{}\n")
    failed = subprocess.run(command, env=env, capture_output=True, text=True)
    assert failed.returncode != 0
    assert not calls.exists()

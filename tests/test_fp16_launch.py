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

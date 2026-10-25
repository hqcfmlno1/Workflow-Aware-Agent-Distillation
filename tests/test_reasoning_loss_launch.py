"""Check the reasoning ablation uses its own profile and fresh-run admission."""

import json
import os
import subprocess
from pathlib import Path


def test_reasoning_ablation_uses_one_profile_and_refuses_reused_output(tmp_path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    fake = venv / "bin/python"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['TEST_LAUNCH_CALLS'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if '--mode' in sys.argv and sys.argv[sys.argv.index('--mode')+1] == 'pilot':\n"
        "    sys.exit(int(os.environ.get('TEST_PILOT_EXIT', '0')))\n"
    )
    fake.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    output = tmp_path / "fresh-output"
    env = {
        **os.environ,
        "SFT5090_VENV": str(venv),
        "SFT5090_OUTPUT": str(output),
        "TEST_LAUNCH_CALLS": str(calls),
    }
    # Parent process settings must not accidentally select the old loss profile.
    env["SFT5090_CONFIG"] = "old-profile.json"
    command = ["bash", "cloud/vast/rtx5090/run-reasoning-loss.sh", "--max-seconds", "3600"]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    stages = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(stages) == 4
    paths = [stage[stage.index("--config") + 1] for stage in stages[1:]]
    assert len(set(paths)) == 1
    old = json.loads(Path("configs/training/gpus/rtx5090_32gb/group_loss.json").read_text())
    new = json.loads(Path(paths[0]).read_text())
    assert new["group_weights"] == [0, 0.5, 0.05, 1, 0.5]
    assert {k: v for k, v in new.items() if k != "group_weights"} == {
        k: v for k, v in old.items() if k != "group_weights"
    }
    final = stages[-1]
    assert final[final.index("--hub-repo") + 1].endswith("-reasoning050-run1")
    assert final[final.index("--hub-prefix") + 1] == "fp16-tqs060-reasoning050"
    assert final[final.index("--max-seconds") + 1] == "3600"
    assert "--resume" not in final
    calls.unlink()
    failed = subprocess.run(
        command, env={**env, "TEST_PILOT_EXIT": "42"}, capture_output=True, text=True
    )
    assert failed.returncode == 42
    assert len(calls.read_text().splitlines()) == 3
    calls.unlink()
    output.mkdir()
    (output / "events.jsonl").write_text("{}\n")
    failed = subprocess.run(command, env=env, capture_output=True, text=True)
    assert failed.returncode != 0
    assert not calls.exists()

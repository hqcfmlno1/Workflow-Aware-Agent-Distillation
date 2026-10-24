import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_entrypoint_help_and_preflight_do_not_require_gpu():
    result = subprocess.run(
        [sys.executable, "-m", "workflow_distillation.lora_fp16", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "preflight" in result.stdout and "pilot" in result.stdout


def test_pilot_gate_rejects_other_identity_device_and_failed_headroom():
    from workflow_distillation.lora_fp16.cli import validate_pilot

    report = dict(identity={"id": 1}, device={"name": "5090"}, passed=True)
    validate_pilot(report, {"id": 1}, {"name": "5090"})
    for changed in [
        dict(report, passed=False),
        dict(report, identity={"id": 2}),
        dict(report, device={"name": "3090"}),
    ]:
        with pytest.raises(ValueError):
            validate_pilot(changed, {"id": 1}, {"name": "5090"})


def test_profile_keeps_exact_fp16_accumulation_and_epoch_pins():
    from workflow_distillation.lora_fp16.config import RunConfig

    config = RunConfig(
        **json.loads(Path("configs/training/gpus/rtx5090_32gb/run.json").read_text())
    )
    assert config.epochs == 2 and config.accumulation == 4
    assert config.base_dtype == "float16" and config.quantization == "none"


def test_new_pilot_invalidates_an_older_passing_report_before_any_cuda_work(tmp_path):
    from workflow_distillation.lora_fp16.cli import invalidate_pilot

    report = tmp_path / "pilot.json"
    report.write_text(json.dumps({"passed": True}))
    invalidate_pilot(report, {"revision": "new"}, {"name": "5090"})
    value = json.loads(report.read_text())
    assert value["passed"] is False and value["status"] == "running"
    assert value["identity"] == {"revision": "new"}

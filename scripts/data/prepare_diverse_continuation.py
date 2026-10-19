"""Freeze a diversity-first continuation cohort on CPU; never launch training or edit HF."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from workflow_distillation.diverse_continuation import build_plan, repository, validate_plan
from workflow_distillation.epoch_data import EpochDataset


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def prepare(index, source_plan, output, *, resume_step, target_step, verify_binaries=False):
    if verify_binaries:
        EpochDataset(index, verify=True)
    plan = build_plan(index, source_plan, resume_step=resume_step, target_step=target_step)
    validate_plan(index, source_plan, plan)
    rows = json.loads(Path(index).read_text())["data"]
    records = [{"position": position, "global_batch": position // 2 + 1, "rank": position % 2,
                "already_trained": position < plan["retained_prefix_count"],
                "source_index": i, "trajectory_id": rows[i]["trajectory_id"],
                "task_key": rows[i]["task_key"], "repo": repository(rows[i]),
                "tokens": rows[i]["length"]}
               for position, i in enumerate(plan["ordered_indices"])]
    summary = {key: value for key, value in plan.items() if key != "ordered_indices"}
    files = {
        "diversity-plan.json": json_bytes(plan), "summary.json": json_bytes(summary),
        "trajectories.jsonl": ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True)
                                          for row in records) + "\n").encode(),
    }
    files["receipt.json"] = json_bytes({
        "sha256": {name: hashlib.sha256(payload).hexdigest() for name, payload in files.items()},
        "source_binaries_verified": verify_binaries,
        "checkpoint_verified": False, "training_launched": False,
    })
    output = Path(output)
    if output.exists():
        if set(p.name for p in output.iterdir()) != set(files) or any(
                not (output / name).is_file() or (output / name).read_bytes() != payload
                for name, payload in files.items()):
            raise ValueError("Output already exists with different selection or verification")
        return summary
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for name, payload in files.items():
            (stage / name).write_bytes(payload)
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--source-plan", required=True, type=Path)
    parser.add_argument("--resume-step", required=True, type=int)
    parser.add_argument("--target-step", type=int, default=2000)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify-binaries", action="store_true")
    args = parser.parse_args()
    summary = prepare(args.index, args.source_plan, args.output, resume_step=args.resume_step,
                      target_step=args.target_step, verify_binaries=args.verify_binaries)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""CPU data/alignment audit; reports runtime mismatch without weakening GPU admission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from workflow_distillation.epoch_data import sha256_file
from workflow_distillation.lora_fp16.config import RunConfig
from workflow_distillation.lora_fp16.metacurate import load_corpus
from workflow_distillation.lora_fp16.span_cache import SpanDataset, load_sidecar
from workflow_distillation.lora_fp16.span_cached_scoring import reference_identity

ROOT = Path(__file__).resolve().parents[2]


def audit(root=ROOT):
    root = Path(root)
    data = root / "data/processed/metacurate1180_stepmask_v1"
    scores = root / "data/processed/metacurate1180_span_stepmask_scores_v1"
    config = RunConfig(**json.loads((
        root / "configs/training/gpus/rtx5090_32gb/metacurate_span_anchor_stepmask_bf16.json"
    ).read_text()))
    dataset, _ = load_corpus(data / "index.json", json.loads((data / "selection.json").read_text()))
    manifest = json.loads((scores / "index.json").read_text())
    ref = manifest["reference"]
    expected = reference_identity(config, ref["provenance_sha256"])
    # CPU laptop cannot have the historical CUDA/FLA packages. Compare every other
    # identity field here; unmodified training preflight also requires package equality.
    if {k: v for k, v in ref.items() if k != "packages"} != {
        k: v for k, v in expected.items() if k != "packages"
    }:
        raise ValueError("Historical scoring source/model/config differs")
    wrapped = SpanDataset(dataset, load_sidecar(scores, sha256_file(data / "index.json"), ref))
    annotation = json.loads((data / "annotations.json").read_text())
    verification = json.loads((data / "verification.json").read_text())
    d = manifest["derivation"]
    count = sum(len(r["annotation"]["mask_steps"]) for r in annotation)
    if (
        len(wrapped) != 1180 or len(annotation) != 1180 or count != 1079
        or verification["status"] != "verified"
        or d["annotations_sha256"] != sha256_file(data / "annotations.json")
        or manifest["tokens"] != verification["retained_target_tokens"]
        or d["excluded_targets"] != verification["masked_target_tokens"]
        or d["input_tokens_unchanged"] is not True
        or d["retained_values_identical"] is not True
        or d["anchor_masked"] is not True or d["rescored"] is not False
        or config.loss_normalization != "offline_span_anchor" or config.epochs != 2
        or config.accumulation != 4 or config.base_dtype != "bfloat16"
    ):
        raise ValueError("Step-mask experiment metadata differs")
    return dict(
        status="data-alignment-verified", records=len(wrapped), masked_steps=count,
        input_tokens=dataset.manifest["tokens"], targets=manifest["tokens"],
        masked_targets=d["excluded_targets"], epochs=config.epochs,
        updates_per_epoch=295, total_updates=590,
        expected_runtime=ref["packages"], local_runtime=expected["packages"],
        runtime_matches=expected["packages"] == ref["packages"],
        gpu_verified=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.root)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

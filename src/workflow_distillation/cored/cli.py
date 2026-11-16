"""Local-only offline CoReD stages. Every run artifact belongs under draft/."""

import argparse
import json
import random
from datetime import UTC, datetime
from pathlib import Path

import torch

from .artifacts import (
    atomic_json,
    exclusive,
    file_identity,
    load_checkpoint,
    sha256_file,
    verify_export,
)
from .config import CoRedConfig
from .data import load_dataset, prepare_existing, write_dataset
from .engine import train
from .merging import export_ties
from .model import Runtime, load_base
from .scoring import score_base, score_council


def _draft(path):
    path = Path(path).absolute()
    if "draft" not in path.parts:
        raise argparse.ArgumentTypeError("Run artifacts must be inside an ignored draft/ directory")
    return path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Convert original corpus + verified WebKD scores")
    prepare.add_argument("--corpus-index", required=True, type=Path)
    prepare.add_argument("--provenance-index", required=True, type=Path)
    prepare.add_argument("--webkd", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=_draft)
    for command in ("score-base", "phase1", "score-council", "merge", "phase2"):
        stage = sub.add_parser(command)
        stage.add_argument(
            "--model", required=True, type=Path, help="Local base or post-TIES model"
        )
        stage.add_argument(
            "--data", required=True, type=Path, help="Prepared/scored dataset directory"
        )
        stage.add_argument("--output", required=True, type=_draft)
        stage.add_argument("--config", type=Path, help="JSON overrides for CoRedConfig")
        stage.add_argument("--device", default="cuda:0")
        if command in ("score-council", "merge"):
            stage.add_argument("--checkpoint", required=True, type=Path, help="Phase 1 checkpoint")
        if command in ("phase1", "phase2"):
            stage.add_argument(
                "--resume", type=Path, help="Hash-verified checkpoint in this output"
            )
    return p


def _identity(stage, model, manifest, config):
    return dict(
        format_version=1,
        stage=stage,
        model=model,
        data=manifest["content_hash"],
        tokenizer_files=manifest["tokenizer_files"],
        config=config.to_dict(),
    )


def _validate_tokenizer(model, expected):
    # config.json belongs to the model and changes on standalone TIES export.
    for name, checksum in expected.items():
        if name == "config.json":
            continue
        candidate = (model / name).resolve()
        if (
            not candidate.is_relative_to(model.resolve())
            or not candidate.is_file()
            or sha256_file(candidate) != checksum
        ):
            raise ValueError(f"Tokenizer/template differs from original corpus: {name}")
    if not expected:
        raise ValueError("Dataset lacks original tokenizer fingerprints; prepare provenance first")


def main(argv=None):
    args = parser().parse_args(argv)
    with exclusive(args.output):

        def emit(event):
            event = dict(time=datetime.now(UTC).isoformat(), **event)
            line = json.dumps(event, allow_nan=False)
            with (args.output / "events.jsonl").open("a") as stream:
                stream.write(line + "\n")
                stream.flush()
            print(line, flush=True)

        if args.command == "prepare":
            identity = dict(
                format_version=1,
                stage="prepare",
                corpus=sha256_file(args.corpus_index),
                provenance=sha256_file(args.provenance_index),
                webkd=sha256_file(args.webkd / "index.json"),
            )
            records = list(prepare_existing(args.corpus_index, args.provenance_index, args.webkd))
            provenance = json.loads(args.provenance_index.read_text())
            write_dataset(args.output, records, identity, provenance.get("tokenizer_files"))
            emit(dict(event="prepared", records=len(records)))
            return
        config = CoRedConfig(**(json.loads(args.config.read_text()) if args.config else {}))
        records, manifest = load_dataset(args.data)
        _validate_tokenizer(args.model, manifest["tokenizer_files"])
        model_identity = file_identity(args.model)
        identity = _identity(args.command, model_identity, manifest, config)
        stage = manifest["identity"]["stage"]
        if args.command == "score-base" and stage != "prepare":
            raise ValueError("score-base expects original prepared data")
        if args.command in ("phase1", "score-council", "merge") and stage != "score-base":
            raise ValueError("Phase 1 and council stages require frozen-base selected data")
        if args.command in ("phase1", "score-council", "merge"):
            expected = _identity(
                "score-base",
                model_identity,
                dict(manifest, content_hash=manifest["identity"]["data"]),
                config,
            )
            if manifest["identity"] != expected:
                raise ValueError("Frozen-base selection model/config identity differs")
        checkpoint_state = None
        if args.command in ("score-council", "merge"):
            expected = _identity("phase1", model_identity, manifest, config)
            checkpoint_state = load_checkpoint(args.checkpoint, expected)
            identity["phase1_checkpoint"] = sha256_file(args.checkpoint / "complete.json")
        if args.command == "phase2":
            merged = verify_export(args.model)
            if (
                stage != "score-council"
                or merged["identity"].get("operation") != "effective-delta-TIES"
                or merged["identity"]["phase1_checkpoint"]
                != manifest["identity"]["phase1_checkpoint"]
            ):
                raise ValueError(
                    "Phase 2 requires matching frozen council weights and post-TIES base"
                )
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        emit(dict(event="loading", stage=args.command, records=len(records), device=args.device))
        base = load_base(args.model, config, args.device)
        count = 1 if args.command == "phase2" else config.experts
        runtime = Runtime(base, config, count=count)
        vocab = runtime.head.weight.shape[0]
        if any(max(r.input_ids) >= vocab for r in records):
            raise ValueError("Input token IDs exceed model vocabulary")
        if checkpoint_state is not None:
            runtime.restore_adapters(checkpoint_state["adapters"])
        if args.command == "score-base":
            scored = score_base(runtime, records, config, args.output, identity, emit)
            write_dataset(args.output, scored, identity, manifest["tokenizer_files"])
        elif args.command == "score-council":
            scored = score_council(runtime, records, config, args.output, identity, emit)
            write_dataset(args.output, scored, identity, manifest["tokenizer_files"])
        elif args.command == "merge":
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
            export_ties(
                runtime,
                tokenizer,
                args.output / "post-ties",
                config,
                identity,
                tokenizer_source=args.model,
            )
        else:
            if args.resume and args.resume.resolve().parent != args.output.resolve():
                raise ValueError("--resume must be a checkpoint from this output directory")
            train(
                runtime,
                records,
                config,
                phase=1 if args.command == "phase1" else 2,
                output=args.output,
                identity=identity,
                resume=args.resume,
                emit=emit,
            )
            export = args.output / "adapters"
            if (export / "complete.json").exists():
                existing = verify_export(export)
                if existing["identity"] != identity:
                    raise ValueError("Adapter export identity differs from this training run")
            else:
                runtime.model.save_pretrained(export, safe_serialization=True)
                from .artifacts import finish_export

                finish_export(export, identity)
        atomic_json(args.output / "stage-complete.json", identity)
        emit(dict(event="complete", stage=args.command))

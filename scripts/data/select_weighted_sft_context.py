"""Write an explicit full-history subset manifest; preserve all token arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from workflow_distillation.weighted_sft_batches import filter_full_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--max-length", required=True, type=int)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite an existing selection")
    raw = args.index.read_bytes()
    index = json.loads(raw)
    selected, excluded = filter_full_history(index["data"], args.max_length)
    if not selected or len(selected) != args.expected_count:
        raise SystemExit(
            f"Context selection has {len(selected)} trajectories; expected {args.expected_count}. "
            "No selection written."
        )
    report = {
        "format_version": 1,
        "context_policy": "complete_trajectory_only",
        "max_length": args.max_length,
        "source_index_sha256": hashlib.sha256(raw).hexdigest(),
        "source_manifest_sha256": index["source_manifest_sha256"],
        "records": len(selected),
        "tokens": sum(r["length"] for r in selected),
        "selected": selected,
        "excluded": excluded,
        "replacement_sampling": False,
        "training_started": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("selected", "excluded")}))


if __name__ == "__main__":
    main()

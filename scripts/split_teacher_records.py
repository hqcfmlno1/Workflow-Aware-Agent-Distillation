"""Split teacher records into workflow/language JSONL files.

Usage: uv run python scripts/split_teacher_records.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from contextlib import ExitStack
from pathlib import Path


NAME = re.compile(r"^[a-zA-Z0-9_-]+$")


def split(source: Path, destination: Path) -> Counter[tuple[str, str]]:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.mkdir(parents=True, exist_ok=True)
    counts: Counter[tuple[str, str]] = Counter()
    pending: dict[tuple[str, str], tuple[Path, Path]] = {}

    try:
        with ExitStack() as stack:
            writers = {}
            with source.open("rb") as records:
                for line_number, line in enumerate(records, 1):
                    if not line.strip():
                        raise ValueError(f"Empty record at line {line_number}")
                    try:
                        record = json.loads(line)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ValueError(f"Invalid JSON at line {line_number}") from exc
                    workflow = record.get("workflow")
                    language = record.get("language")
                    if not isinstance(workflow, str) or not NAME.fullmatch(workflow):
                        raise ValueError(f"Invalid workflow at line {line_number}: {workflow!r}")
                    if not isinstance(language, str) or not NAME.fullmatch(language):
                        raise ValueError(f"Invalid language at line {line_number}: {language!r}")
                    key = (workflow, language)
                    if key not in writers:
                        folder = destination / workflow / language
                        folder.mkdir(parents=True, exist_ok=True)
                        target = folder / "records.jsonl"
                        temporary = folder / "records.jsonl.tmp"
                        if temporary.exists():
                            raise FileExistsError(f"Refusing to overwrite temporary file: {temporary}")
                        pending[key] = (temporary, target)
                        writers[key] = stack.enter_context(temporary.open("xb"))
                    writers[key].write(line)
                    counts[key] += 1

        for temporary, target in pending.values():
            os.replace(temporary, target)
    except BaseException:
        for temporary, _ in pending.values():
            temporary.unlink(missing_ok=True)
        raise
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("teacher_dataset/records.jsonl"))
    parser.add_argument("--destination", type=Path, default=Path("teacher_dataset/by_workflow"))
    args = parser.parse_args()
    counts = split(args.source, args.destination)
    for (workflow, language), count in sorted(counts.items()):
        print(f"{workflow}/{language}: {count}")
    print(f"Total: {sum(counts.values())}")


if __name__ == "__main__":
    main()

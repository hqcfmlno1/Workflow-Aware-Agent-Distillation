"""Lossless trajectory/segment contract and existing WebKD sidecar conversion."""

import gzip
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Segment:
    key: str
    kind: str
    positions: list[int]
    fractions: list[float]
    webkd_weight: float = 1.0
    dropout_group: str = ""

    def __post_init__(self):
        if (
            not self.key
            or self.kind not in ("reasoning", "response")
            or not self.positions
            or self.positions != sorted(set(self.positions))
            or len(self.positions) != len(self.fractions)
            or any(not math.isfinite(f) or not 0 < f <= 1 for f in self.fractions)
            or not math.isfinite(self.webkd_weight)
            or not 0.1 - 1e-7 <= self.webkd_weight <= 1
            or (self.kind == "response" and self.webkd_weight != 1)
        ):
            raise ValueError("Invalid supervised segment")
        if not self.dropout_group:
            self.dropout_group = self.key


@dataclass
class Record:
    trajectory_id: str
    input_ids: list[int]
    groups: list[int]
    segments: list[Segment]
    selected: list[str] | None = None
    council_weights: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.segments = [Segment(**s) if isinstance(s, dict) else s for s in self.segments]
        if (
            not self.trajectory_id
            or len(self.input_ids) < 2
            or len(self.input_ids) != len(self.groups)
            or any(type(i) is not int or i < 0 for i in self.input_ids)
            or any(type(g) is not int or not 0 <= g <= 4 for g in self.groups)
            or self.groups[0] != 0
            or not self.segments
        ):
            raise ValueError("Invalid complete token/group trajectory")
        coverage = np.zeros(len(self.input_ids), dtype=float)
        keys = set()
        for segment in self.segments:
            if segment.key in keys:
                raise ValueError("Duplicate segment key")
            keys.add(segment.key)
            for position, fraction in zip(segment.positions, segment.fractions, strict=True):
                if (
                    type(position) is not int
                    or not 0 < position < len(coverage)
                    or self.groups[position] == 0
                    or (segment.kind == "reasoning" and self.groups[position] != 1)
                ):
                    raise ValueError("Segment supervises context or wrong token group")
                coverage[position] += fraction
        if not np.allclose(coverage, np.asarray(self.groups) > 0, atol=1e-7, rtol=0):
            raise ValueError("Segment coverage differs from complete assistant supervision")
        reasoning = {s.key for s in self.reasoning}
        if self.selected is not None and (
            len(self.selected) != len(set(self.selected)) or not set(self.selected) <= reasoning
        ):
            raise ValueError("Repulsion selection must contain reasoning spans only")
        if any(
            k not in reasoning or not math.isfinite(v) or not 0 <= v <= 2
            for k, v in self.council_weights.items()
        ):
            raise ValueError("Invalid fixed council weights")

    @property
    def reasoning(self):
        return [s for s in self.segments if s.kind == "reasoning"]

    @property
    def positions(self):
        return [i for i, g in enumerate(self.groups) if i > 0 and g > 0]

    @property
    def key_positions(self):
        if self.selected is None:
            raise ValueError("Frozen-base harmonic selection has not been computed")
        return sorted({p for s in self.reasoning if s.key in self.selected for p in s.positions})

    def to_dict(self):
        return asdict(self)


def read_records(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    seen = set()
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            value = Record(**json.loads(line))
            if value.trajectory_id in seen:
                raise ValueError("Duplicate trajectory ID")
            seen.add(value.trajectory_id)
            yield value


def from_provenance(sample, provenance, turn_scores):
    """Fractional memberships reproduce character-overlap WebKD token weights."""
    ids, groups = list(map(int, sample["input_ids"])), list(map(int, sample["groups"]))
    if provenance["trajectory_id"] != sample["trajectory_id"] or provenance["length"] != len(ids):
        raise ValueError("Provenance/corpus order differs")
    offsets = provenance["offsets"]
    if len(offsets) != len(ids):
        raise ValueError("Offsets/corpus length differs")
    segments, assigned = [], np.zeros(len(ids), dtype=float)
    for turn in provenance["turns"]:
        spans = turn["spans"]
        scores = turn_scores.get(str(turn["message_index"]))
        if turn["status"] == "eligible":
            if scores is None or len(scores.get("weights", [])) != len(spans):
                raise ValueError("Missing original per-span WebKD scoring result")
            weights = scores["weights"]
        else:
            weights = [1.0] * len(spans)
        for si, ((left, right), weight) in enumerate(zip(spans, weights, strict=True)):
            positions, fractions = [], []
            for pos, (a, b) in enumerate(offsets):
                overlap = max(0, min(b, right) - max(a, left))
                if pos > 0 and groups[pos] == 1 and b > a and overlap:
                    positions.append(pos)
                    fractions.append(overlap / (b - a))
                    assigned[pos] += fractions[-1]
            if positions:
                key = f"m{turn['message_index']}:s{si}"
                segments.append(Segment(key, "reasoning", positions, fractions, float(weight)))
    if (assigned > 1 + 1e-7).any():
        raise ValueError("Overlapping character spans")
    # Preserve all assistant protocol, content, action and final-ending targets.
    # Context boundaries split response segments; reasoning gaps split them too.
    current_p, current_f = [], []

    def finish():
        if current_p:
            segments.append(
                Segment(f"response:{len(segments)}", "response", current_p.copy(), current_f.copy())
            )
            current_p.clear()
            current_f.clear()

    for pos in range(1, len(ids)):
        remaining = max(0.0, 1 - assigned[pos]) if groups[pos] > 0 else 0.0
        if remaining > 1e-7:
            current_p.append(pos)
            current_f.append(remaining)
        else:
            finish()
    finish()
    result = Record(sample["trajectory_id"], ids, groups, segments)
    if "weights" in sample:
        actual = np.zeros(len(ids))
        for segment in segments:
            for p, f in zip(segment.positions, segment.fractions, strict=True):
                actual[p] += f * segment.webkd_weight
        if not np.allclose(actual[result.positions], sample["weights"], atol=2e-6, rtol=0):
            raise ValueError("Converted weights differ from existing immutable WebKD sidecar")
    return result


def prepare_existing(corpus_index, provenance_index, webkd_path):
    """Read verified original corpus and full scoring store; never re-render a template."""
    from workflow_distillation.epoch_data import EpochDataset, sha256_file
    from workflow_distillation.lora_fp16.span_cache import SpanDataset, load_sidecar

    provenance_index, webkd_path = Path(provenance_index), Path(webkd_path)
    source_manifest = json.loads(Path(corpus_index).read_text())
    provenance_meta = source_manifest.get("provenance", {})
    if (
        "mask_policy" in provenance_meta
        or "annotation_files_sha256" in provenance_meta
        or any(row.get("masked_target_tokens", 0) for row in source_manifest["data"])
    ):
        raise ValueError("Luna/step-masked corpus is not allowed in this recipe")
    if any(
        r["length"] != r["full_token_count"] or r.get("omitted_reasoning_messages")
        for r in source_manifest["data"]
    ):
        raise ValueError("Expected a full trajectory with all original reasoning")
    manifest = json.loads(provenance_index.read_text())
    corpus_hash = sha256_file(Path(corpus_index))
    if manifest["corpus_sha256"] != corpus_hash:
        raise ValueError("Provenance corpus checksum differs")
    wm = json.loads((webkd_path / "index.json").read_text())
    sidecar = load_sidecar(webkd_path, corpus_hash, wm["reference"])
    if wm["reference"]["provenance_sha256"] != sha256_file(provenance_index):
        raise ValueError("WebKD provenance checksum differs")
    dataset = SpanDataset(EpochDataset(corpus_index), sidecar)
    if len(manifest["data"]) != len(dataset):
        raise ValueError("Provenance record count differs")
    for i, row in enumerate(manifest["data"]):
        path = provenance_index.parent / row["file"]
        if (
            path.resolve().parent != provenance_index.parent.resolve()
            or sha256_file(path) != row["sha256"]
        ):
            raise ValueError("Unsafe/corrupt provenance file")
        with gzip.open(path, "rt") as stream:
            provenance = json.load(stream)
        scores = {}
        for turn in provenance["turns"]:
            mi = turn["message_index"]
            path = webkd_path / "parts" / f"{i:06d}.turn-{mi}.json"
            if path.is_file():
                scores[str(mi)] = json.loads(path.read_text())
        yield from_provenance(dataset.record(i), provenance, scores)


def write_dataset(root, records, identity, tokenizer_files=None):
    """Commit a complete portable dataset, preserving exact token arrays and span mass."""
    import os

    from workflow_distillation.epoch_data import sha256_file

    from .artifacts import atomic_json, digest

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    canonical = [r.to_dict() for r in records]
    if not canonical or len({r["trajectory_id"] for r in canonical}) != len(canonical):
        raise ValueError("Nonempty unique trajectories required")
    content_hash = digest(canonical)
    if (root / "dataset.json").exists():
        _, existing = load_dataset(root)
        if existing["identity"] != identity or existing["content_hash"] != content_hash:
            raise ValueError("Dataset identity/content differs; choose a new output")
        return existing
    temporary = root / "records.jsonl.gz.tmp"
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as stream:
            for value in canonical:
                stream.write(
                    (json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n").encode()
                )
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, root / "records.jsonl.gz")
    manifest = dict(
        format_version=1,
        identity=identity,
        records=len(canonical),
        tokens=sum(len(r.input_ids) for r in records),
        tokenizer_files=tokenizer_files or {},
        content_hash=content_hash,
        sha256=sha256_file(root / "records.jsonl.gz"),
    )
    atomic_json(root / "dataset.json", manifest)
    return manifest


def load_dataset(root):
    from workflow_distillation.epoch_data import sha256_file

    from .artifacts import digest

    root = Path(root)
    m = json.loads((root / "dataset.json").read_text())
    if m.get("format_version") != 1 or sha256_file(root / "records.jsonl.gz") != m["sha256"]:
        raise ValueError("Dataset checksum/version differs")
    records = list(read_records(root / "records.jsonl.gz"))
    if (
        len(records) != m["records"]
        or sum(len(r.input_ids) for r in records) != m["tokens"]
        or digest([r.to_dict() for r in records]) != m["content_hash"]
    ):
        raise ValueError("Dataset content differs")
    return records, m

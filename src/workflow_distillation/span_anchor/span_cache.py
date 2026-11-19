"""Atomic per-record/variant scoring recovery and verified compact training sidecars."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from workflow_distillation.epoch_data import sha256_file

NAMES = ("ref_logp", "weights", "anchor_weights")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sample_hash(ids, groups):
    return hashlib.sha256(
        np.asarray(ids, dtype="<u4").tobytes() + np.asarray(groups, dtype="u1").tobytes()
    ).hexdigest()


def target_groups(groups):
    groups = np.asarray(groups)
    positions = np.flatnonzero(groups > 0)
    positions = positions[positions > 0]
    return groups[positions]


def validate_arrays(groups, q, w, gamma):
    g = target_groups(groups)
    arrays = [np.asarray(a, dtype="<f4") for a in (q, w, gamma)]
    if (
        not len(g)
        or any(a.shape != g.shape or not np.isfinite(a).all() for a in arrays)
        or np.any(arrays[0] > 1e-5)
        or np.any((arrays[1] < 0.1 - 1e-7) | (arrays[1] > 1))
        or np.any(arrays[1][g != 1] != 1)
        or np.any((arrays[2] < 1e-6 - 1e-12) | (arrays[2] > 1))
    ):
        raise ValueError("Invalid sidecar values, target mass or non-reasoning weights")
    return arrays


class ScoreStore:
    def __init__(self, path, corpus_hash, reference_identity):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.identity = dict(corpus_sha256=corpus_hash, reference=reference_identity)
        identity_path = self.path / "identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != self.identity:
                raise ValueError("Scoring cache identity differs; choose a new output")
        else:
            atomic_json(identity_path, self.identity)
        (self.path / "parts").mkdir(exist_ok=True)

    def _file(self, index, suffix):
        return self.path / "parts" / f"{index:06d}.{suffix}"

    def _save_npz(self, path, **values):
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("wb") as stream:
            np.savez_compressed(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)

    def write_reference(self, index, ids, groups, q):
        q = np.asarray(q, dtype="<f4")
        if q.shape != target_groups(groups).shape or not np.isfinite(q).all() or np.any(q > 1e-5):
            raise ValueError("Invalid reference cache")
        self._save_npz(
            self._file(index, "ref.npz"), sample_hash=sample_hash(ids, groups), ref_logp=q
        )

    def read_reference(self, index, ids, groups):
        path = self._file(index, "ref.npz")
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            if str(data["sample_hash"]) != sample_hash(ids, groups):
                raise ValueError("Reference token alignment differs")
            q = data["ref_logp"].copy()
        if q.shape != target_groups(groups).shape or not np.isfinite(q).all():
            raise ValueError("Invalid reference cache")
        return q

    def write_turn(self, index, message_index, value):
        atomic_json(self._file(index, f"turn-{message_index}.json"), value)

    def read_turn(self, index, message_index):
        path = self._file(index, f"turn-{message_index}.json")
        return json.loads(path.read_text()) if path.exists() else None

    def write_record(self, index, ids, groups, q, w, gamma):
        arrays = validate_arrays(groups, q, w, gamma)
        self._save_npz(
            self._file(index, "complete.npz"),
            sample_hash=sample_hash(ids, groups),
            **dict(zip(NAMES, arrays, strict=True)),
        )

    def read_record(self, index, ids, groups):
        path = self._file(index, "complete.npz")
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            if str(data["sample_hash"]) != sample_hash(ids, groups):
                raise ValueError("Sidecar token alignment differs")
            arrays = [data[n].copy() for n in NAMES]
        arrays = validate_arrays(groups, *arrays)
        return dict(zip(NAMES, arrays, strict=True))

    def finalize(self, records):
        from contextlib import ExitStack

        rows, offset = [], 0
        with ExitStack() as stack:
            streams = {
                n: stack.enter_context((self.path / f"{n}.bin.tmp").open("wb")) for n in NAMES
            }
            for i, sample in enumerate(records):
                arrays = self.read_record(i, sample["input_ids"], sample["groups"])
                if arrays is None:
                    raise ValueError(f"Scoring incomplete at record {i}")
                count = len(arrays["weights"])
                rows.append(
                    dict(
                        trajectory_id=sample["trajectory_id"],
                        offset=offset,
                        count=count,
                        sample_hash=sample_hash(sample["input_ids"], sample["groups"]),
                    )
                )
                for name in NAMES:
                    arrays[name].tofile(streams[name])
                offset += count
            for stream in streams.values():
                stream.flush()
                os.fsync(stream.fileno())
        for name in NAMES:
            (self.path / f"{name}.bin.tmp").replace(self.path / f"{name}.bin")
        manifest = dict(
            format_version=1,
            profile="offline-span-anchor-v1",
            complete=True,
            **self.identity,
            tokens=offset,
            data=rows,
            file_sha256={f"{n}.bin": sha256_file(self.path / f"{n}.bin") for n in NAMES},
        )
        atomic_json(self.path / "index.json", manifest)
        return manifest


class Sidecar:
    def __init__(self, path, manifest):
        self.manifest = manifest
        self.arrays = {n: np.memmap(path / f"{n}.bin", dtype="<f4", mode="r") for n in NAMES}

    def record(self, index):
        row = self.manifest["data"][index]
        return {n: a[row["offset"] : row["offset"] + row["count"]] for n, a in self.arrays.items()}


def load_sidecar(path, corpus_hash, reference_identity):
    path = Path(path)
    m = json.loads((path / "index.json").read_text())
    if (
        m.get("complete") is not True
        or m.get("profile") != "offline-span-anchor-v1"
        or m.get("format_version") != 1
        or m["corpus_sha256"] != corpus_hash
        or m["reference"] != reference_identity
    ):
        raise ValueError("Sidecar identity/profile differs or incomplete")
    offset = 0
    for row in m["data"]:
        if row["offset"] != offset or row["count"] < 1:
            raise ValueError("Invalid sidecar offsets")
        offset += row["count"]
    if not offset or offset != m["tokens"]:
        raise ValueError("Invalid sidecar target count")
    for name in NAMES:
        p = path / f"{name}.bin"
        if p.stat().st_size != offset * 4 or sha256_file(p) != m["file_sha256"][p.name]:
            raise ValueError(f"Sidecar checksum/size mismatch: {name}")
    return Sidecar(path, m)


class SpanDataset:
    def __init__(self, dataset, sidecar):
        self.dataset, self.sidecar = dataset, sidecar
        self.records, self.manifest = dataset.records, dataset.manifest
        if len(self.records) != len(sidecar.manifest["data"]):
            raise ValueError("Sidecar/corpus count differs")
        for i, row in enumerate(sidecar.manifest["data"]):
            record = dataset.record(i)
            if row["trajectory_id"] != record["trajectory_id"] or row["sample_hash"] != sample_hash(
                record["input_ids"], record["groups"]
            ):
                raise ValueError("Sidecar/corpus token order differs")
            validate_arrays(
                record["groups"],
                **dict(
                    zip(("q", "w", "gamma"), (sidecar.record(i)[n] for n in NAMES), strict=True)
                ),
            )

    def __len__(self):
        return len(self.records)

    def record(self, index):
        return {**self.dataset.record(index), **self.sidecar.record(index)}

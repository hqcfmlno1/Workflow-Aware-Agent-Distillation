"""Restartable frozen teacher-forced scoring, with full-corpus normalization."""

from dataclasses import replace

from .artifacts import Store, atomic_json, digest
from .objectives import council_statistics, segment_means
from .selection import council_weights, select_repulsion


def _span_values(record, values):
    means = segment_means(record, values).tolist()
    return {s.key: v for s, v in zip(record.segments, means, strict=True) if s.kind == "reasoning"}


def score_base(runtime, records, config, output, identity, emit=lambda e: None):
    store = Store(output, identity)
    result = []
    for index, record in enumerate(records):
        key = f"{index:06d}"
        part = store.get(key)
        record_hash = digest(record.to_dict())
        if part is None:
            states = runtime.snapshot(record, None, record.positions)
            entropy, _ = council_statistics([states], runtime.head, config.projection_chunk)
            spans = _span_values(record, entropy)
            selection = select_repulsion(
                [s.webkd_weight for s in record.reasoning],
                [spans[s.key] for s in record.reasoning],
                config.repulsion_fraction,
            )
            part = dict(
                record_hash=record_hash,
                entropy=spans,
                selection=selection,
                selected=[record.reasoning[i].key for i in selection["selected"]],
            )
            store.put(key, part)
        if part["record_hash"] != record_hash:
            raise ValueError("Scoring cache trajectory differs")
        result.append(replace(record, selected=part["selected"]))
        emit(
            dict(
                event="base_scored",
                record=index + 1,
                total=len(records),
                trajectory_id=record.trajectory_id,
                selected=len(part["selected"]),
            )
        )
    return result


def score_council(runtime, records, config, output, identity, emit=lambda e: None):
    store = Store(output, identity)
    parts, all_u, all_v = [], [], []
    for index, record in enumerate(records):
        key, record_hash = f"{index:06d}", digest(record.to_dict())
        part = store.get(key)
        if part is None:
            states = [runtime.snapshot(record, name, record.positions) for name in runtime.names]
            u, v = council_statistics(states, runtime.head, config.projection_chunk)
            part = dict(
                record_hash=record_hash,
                uncertainty=_span_values(record, u),
                disagreement=_span_values(record, v),
            )
            store.put(key, part)
        if part["record_hash"] != record_hash:
            raise ValueError("Council cache trajectory differs")
        parts.append(part)
        all_u.extend(part["uncertainty"][s.key] for s in record.reasoning)
        all_v.extend(part["disagreement"][s.key] for s in record.reasoning)
        emit(
            dict(
                event="council_scored",
                record=index + 1,
                total=len(records),
                trajectory_id=record.trajectory_id,
            )
        )
    weights, moments = council_weights(
        all_u,
        all_v,
        lambda_u=config.uncertainty_coefficient,
        lambda_d=config.disagreement_coefficient,
        epsilon=config.disagreement_epsilon,
        standardization_epsilon=config.standardization_epsilon,
    )
    atomic_json(store.root / "moments.json", moments)
    result, cursor = [], 0
    for record in records:
        n = len(record.reasoning)
        values = dict(
            zip([s.key for s in record.reasoning], weights[cursor : cursor + n], strict=True)
        )
        result.append(replace(record, council_weights=values))
        cursor += n
    return result

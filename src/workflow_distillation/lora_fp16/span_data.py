"""Exact rendered provenance and WebKD-style reasoning-span weights.

Segmentation/rank semantics independently adapted from Apache-2.0 WebKD,
revision 95cdc1d21ce3b590063f6ab9c5bbd3eb25063de5. See third_party/webkd/NOTICE.
"""

from __future__ import annotations

import copy
import math
import re

import numpy as np

UPSTREAM_REVISION = "95cdc1d21ce3b590063f6ab9c5bbd3eb25063de5"
SCORING_VERSION = "coding-span-v1-singleton-unit"
_BOUNDARY = re.compile(r"(?:\n+|(?<=[.!?;:])[ \t]+)")


def segment_reasoning(text, min_chars=24, max_spans=24):
    if min_chars < 1 or max_spans < 1:
        raise ValueError("Positive segmentation limits required")
    if not text:
        return []
    cuts, pos = [], 0
    for match in _BOUNDARY.finditer(text):
        cuts.append((pos, match.end()))
        pos = match.end()
    if pos < len(text):
        cuts.append((pos, len(text)))

    def nonspace(a, b):
        return len(re.sub(r"\s+", "", text[a:b]))

    spans = []
    for a, b in cuts:
        if spans and nonspace(a, b) < min_chars:
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))
    while len(spans) > 1 and nonspace(*spans[0]) < min_chars:
        spans[1] = (spans[0][0], spans[1][1])
        spans.pop(0)
    while len(spans) > max_spans:
        i = min(range(len(spans) - 1), key=lambda j: nonspace(*spans[j]) + nonspace(*spans[j + 1]))
        spans[i : i + 2] = [(spans[i][0], spans[i + 1][1])]
    return spans


def span_weights(deltas, floor=0.1, dead_zone=0.05):
    if not 0 < floor <= 1 or not math.isfinite(dead_zone) or dead_zone <= 0:
        raise ValueError("Invalid floor/dead zone")
    if any(not math.isfinite(v) for v in deltas):
        raise ValueError("Nonfinite contributions")
    n = len(deltas)
    if n <= 1:
        return [1.0] * n
    order = sorted(range(n), key=lambda i: deltas[i])
    ranks, i = [0.0] * n, 0
    while i < n:
        j = i + 1
        while j < n and deltas[order[j]] == deltas[order[i]]:
            j += 1
        for k in range(i, j):
            ranks[order[k]] = (i + j - 1) / 2
        i = j
    shrink = min(1, (max(deltas) - min(deltas)) / dead_zone)
    return [
        min(1.0, max(floor, floor + (1 - floor) * (0.5 + (r / (n - 1) - 0.5) * shrink)))
        for r in ranks
    ]


def token_weights(offsets, spans, values, default=1.0):
    """Character-overlap mapping; uncovered chars and zero-width tokens keep default."""
    if len(spans) != len(values):
        raise ValueError("Span/value length mismatch")
    result = []
    for left, right in offsets:
        value = default
        if right > left:
            for (a, b), weight in zip(spans, values, strict=True):
                overlap = max(0, min(right, b) - max(left, a))
                value += overlap / (right - left) * (weight - default)
        result.append(value)
    return result


def prepare_record(messages, tools, tokenizer, expected_ids, groups):
    from scripts.data.prepare_action_focused_sft import serialize
    from workflow_distillation.weighted_sft_data import annotate_record

    rendered = serialize(messages, tools, tokenizer)
    annotated = annotate_record(dict(messages=messages, tools=tools, **rendered), tokenizer)
    if annotated["input_ids"] != list(expected_ids) or annotated["groups"] != list(groups):
        raise ValueError("Rendered token IDs/groups differ from immutable corpus")
    text = rendered["text"]
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    marked = copy.deepcopy(messages)
    markers = {}
    for mi, message in enumerate(marked):
        if message["role"] != "assistant":
            continue
        content = (message.get("content") or "").strip()
        reasoning = message.get("reasoning_content")
        if not isinstance(reasoning, str):
            reasoning = ""
            if "</think>" in content:
                reasoning = (
                    content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                )
                content = content.split("</think>")[-1].lstrip("\n")
        for field, value in [("reasoning_content", reasoning), ("content", content)]:
            if field == "content" and not value:
                continue
            pair = []
            for edge in ("begin", "end"):
                marker = f"__SPAN_LORA_{mi}_{field}_{edge}__"
                if marker in text:
                    raise ValueError("Provenance marker collision")
                markers[marker] = (mi, field, edge)
                pair.append(marker)
            message[field] = pair[0] + value.strip() + pair[1]
    marked_text = tokenizer.apply_chat_template(
        marked, tools=tools, tokenize=False, add_generation_prompt=False
    )
    pattern = re.compile("|".join(re.escape(k) for k in markers))
    clean, positions, previous, removed = [], {}, 0, 0
    for match in pattern.finditer(marked_text):
        clean.append(marked_text[previous : match.start()])
        positions[markers[match.group()]] = match.start() - removed
        removed += len(match.group())
        previous = match.end()
    clean.append(marked_text[previous:])
    if "".join(clean) != text:
        raise ValueError("Provenance marker render changed training text")
    turns, rationale, envelopes = [], [], []
    for mi, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        a = positions.get((mi, "reasoning_content", "begin"))
        b = positions.get((mi, "reasoning_content", "end"))
        spans = (
            []
            if a is None
            else [(a + left, a + right) for left, right in segment_reasoning(text[a:b])]
        )
        if a is not None and b > a:
            rationale.append((a, b))
        calls = [c for c in rendered["action_candidates"] if c["message_index"] == mi]
        ends = [positions.get((mi, f, "end"), 0) for f in ("reasoning_content", "content")]
        ends.extend(c["char_span"][1] for c in calls)
        ending = text.find("<|im_end|>\n", max(v or 0 for v in ends))
        if ending < 0:
            raise ValueError("Missing response ending")
        envelopes.append((ending, ending + len("<|im_end|>\n")))
        turn = dict(
            message_index=mi,
            spans=spans,
            call_count=len(calls),
            submit=False,
            status="eligible"
            if spans and calls
            else ("no_action_unscored" if not calls else "no_rendered_reasoning"),
        )
        if calls:
            left, right = calls[0]["char_span"][0], calls[-1]["char_span"][1]
            tokens = [i for i, (begin, end) in enumerate(offsets) if begin < right and end > left]
            start, end = tokens[0], tokens[-1] + 1
            left, right = offsets[start][0], offsets[end - 1][1]
            if tokenizer.encode(text[:left], add_special_tokens=False) != list(
                expected_ids[:start]
            ):
                raise ValueError("Action boundary changes prefix tokenization")
            if tokenizer.encode(text[left:right], add_special_tokens=False) != list(
                expected_ids[start:end]
            ):
                raise ValueError("Action boundary changes target tokenization")
            turn.update(
                action_tokens=[start, end],
                action_chars=[left, right],
                action_ids=list(expected_ids[start:end]),
                submit=bool(np.any(np.asarray(groups[start:end]) == 4)),
            )
        turns.append(turn)
    gamma = token_weights(
        offsets, rationale + envelopes, [1.0] * (len(rationale) + len(envelopes)), default=1e-6
    )
    gamma = [
        v if g > 0 and i > 0 else 0.0 for i, (v, g) in enumerate(zip(gamma, groups, strict=True))
    ]
    return dict(
        text=text,
        offsets=offsets,
        turns=turns,
        anchor_weights=gamma,
        omitted_reasoning_messages=annotated["omitted_reasoning_messages"],
    )

"""CPU-only token provenance for the pinned Qwen3.5 chat template."""

from __future__ import annotations

import copy
import re
from collections import Counter

from scripts.data.prepare_action_focused_sft import serialize, terminal_candidate

WEIGHTS = (0.0, 1.0, 0.75, 1.5, 2.0)
TEMPLATE_SHA256 = "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"


def annotate_record(record, tokenizer):
    """Verify the official render, then label only structured assistant regions.

    Field markers are stripped from a second official render. They never enter
    the saved input, and their removal must recover the original text exactly.
    """
    rendered = serialize(record["messages"], record["tools"], tokenizer)
    if rendered["text"] != record["text"]:
        raise ValueError("Saved text does not match the official render")
    if rendered["action_candidates"] != record["action_candidates"]:
        raise ValueError("Action spans do not match the official render")
    text = rendered["text"]
    marked = copy.deepcopy(record["messages"])
    markers = {}
    assistants = []

    def wrap(value, message_index, group):
        value = value.strip()
        tokens = []
        for edge in ("begin", "end"):
            token = f"__WAAD_SFT_{message_index}_{group}_{edge}__"
            if token in text:
                raise ValueError("Field marker collision")
            markers[token] = (message_index, group, edge)
            tokens.append(token)
        return tokens[0] + value + tokens[1]

    for mi, message in enumerate(marked):
        if message["role"] != "assistant":
            continue
        assistants.append(mi)
        content = (message.get("content") or "").strip()
        reasoning = message.get("reasoning_content")
        if not isinstance(reasoning, str):
            reasoning = ""
            if "</think>" in content:
                reasoning = (
                    content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                )
                content = content.split("</think>")[-1].lstrip("\n")
        message["reasoning_content"] = wrap(reasoning, mi, 1)
        if content:
            message["content"] = wrap(content, mi, 2)

    marked_text = tokenizer.apply_chat_template(
        marked, tools=record["tools"], tokenize=False, add_generation_prompt=False
    )
    pattern = re.compile("|".join(re.escape(token) for token in markers))
    positions = {}
    pieces = []
    previous = removed = 0
    for match in pattern.finditer(marked_text):
        pieces.append(marked_text[previous : match.start()])
        positions[markers[match.group()]] = match.start() - removed
        removed += len(match.group())
        previous = match.end()
    pieces.append(marked_text[previous:])
    if "".join(pieces) != text:
        raise ValueError("Field marker removal changed the official render")

    regions = []
    omitted = []
    for mi in assistants:
        calls = [c for c in rendered["action_candidates"] if c["message_index"] == mi]
        if any(c["char_span"] is None for c in calls):
            raise ValueError("Unverified tool-call span cannot receive training labels")
        rstart = positions.get((mi, 1, "begin"))
        rend = positions.get((mi, 1, "end"))
        cstart = positions.get((mi, 2, "begin"))
        cend = positions.get((mi, 2, "end"))
        if rstart is not None:
            header = "<|im_start|>assistant\n<think>\n"
            start = rstart - len(header)
            body_start = rstart
        else:
            if record["messages"][mi].get("reasoning_content"):
                omitted.append(mi)
            header = "<|im_start|>assistant\n"
            body_start = cstart if cstart is not None else calls[0]["char_span"][0]
            start = body_start - len(header)
        if text[start:body_start] != header:
            raise ValueError("Assistant header could not be verified")
        final_call = False
        if calls:
            last_call = calls[-1]
            source = record["messages"][mi]["tool_calls"][last_call["call_index"]]["function"]
            final_call = mi == len(record["messages"]) - 1 and terminal_candidate(
                source["arguments"].get("command", "")
            )
            end_marker = calls[-1]["char_span"][1]
        elif cend is not None:
            end_marker = cend
        else:
            end_marker = rend
            if text[end_marker:].startswith("\n</think>\n\n"):
                end_marker += len("\n</think>\n\n")
        if not text[end_marker:].startswith("<|im_end|>\n"):
            raise ValueError("Assistant ending could not be verified")
        end = end_marker + len("<|im_end|>\n")
        # Explicit protocol regions supply the gaps around structured fields.
        default = 3 if calls or rstart is not None else 2
        local = [(body_start, end, default)]
        if rstart is not None and rend > rstart:
            local.append((rstart, rend, 1))
        if cstart is not None:
            local.append((cstart, cend, 2))
        for call in calls:
            left, right = call["char_span"]
            group = 4 if final_call and call is calls[-1] else 3
            local.append((left, right, group))
        if final_call:
            local.append((end_marker, end, 4))
        regions.extend(local)

    # Bytearray character groups avoid confusing literal tags with structure.
    char_groups = bytearray(len(text))
    for left, right, group in regions:
        char_groups[left:right] = bytes([group]) * (right - left)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    groups = []
    boundary_masked = 0
    for left, right in encoded["offset_mapping"]:
        choices = set(char_groups[left:right])
        # A BPE whitespace token can cross the supplied assistant prefix when
        # reasoning is empty. Never supervise any portion of supplied context.
        if 0 in choices and len(choices) > 1 and text[left:right].isspace():
            groups.append(0)
            boundary_masked += 1
            continue
        if len(choices) != 1:
            raise ValueError(f"Ambiguous token region at characters {left}:{right}: {choices}")
        groups.append(choices.pop())
    counts = Counter(groups)
    if not counts[3] or not counts[4]:
        raise ValueError("Trajectory lacks ordinary actions or terminal submission")
    return {
        "input_ids": encoded["input_ids"],
        "groups": groups,
        "group_counts": [counts[i] for i in range(5)],
        "denominator": sum(WEIGHTS[i] * counts[i] for i in range(1, 5)),
        "omitted_reasoning_messages": omitted,
        "boundary_masked_tokens": boundary_masked,
    }


def plan_windows(groups, *, max_length, target_length):
    """Return (context start, target start, end); never duplicate target loss."""
    length = len(groups)
    if length < 2 or groups[0]:
        raise ValueError("Targets require an available predecessor token")
    if target_length is None:
        if length > max_length:
            raise ValueError("Trajectory exceeds the full-history context policy")
        return [(0, 1, length)]
    if not 0 < target_length < max_length:
        raise ValueError("Target length must leave room for conditioning context")
    result = []
    for target_start in range(1, length, target_length):
        end = min(target_start + target_length, length)
        if any(groups[target_start:end]):
            start = max(0, end - max_length)
            result.append((start, target_start, end))
    return result

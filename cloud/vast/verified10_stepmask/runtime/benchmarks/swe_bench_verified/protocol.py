"""Translate Qwen native XML bash calls into OpenAI tool calls, without execution."""

from __future__ import annotations

import copy
import json
import re
import uuid

CALL = re.compile(
    r"<tool_call>\s*<function=bash>\s*<parameter=command>\n?"
    r"(?P<command>.*?)\n</parameter>\s*</function>\s*</tool_call>",
    re.DOTALL,
)


def normalize_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    result = copy.deepcopy(messages)
    for message in result:
        if message.get("role") not in ("system", "user", "assistant", "tool"):
            raise ValueError("Only text system/user/assistant/tool messages are supported")
        if not isinstance(message.get("content", ""), (str, type(None))):
            raise ValueError("This endpoint is text-only")
        message["content"] = message.get("content") or ""
        for call in message.get("tool_calls") or []:
            function = call["function"]
            arguments = function["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if function["name"] != "bash" or (
                not isinstance(arguments, dict)
                or set(arguments) != {"command"}
                or not isinstance(arguments["command"], str)
            ):
                raise ValueError("Only bash(command: string) is supported")
            function["arguments"] = arguments
    return result


def parse_completion(text, *, thinking=True, truncated=False):
    reasoning = ""
    if "</think>" in text:
        reasoning, text = text.split("</think>", 1)
        reasoning = reasoning.removeprefix("<think>").strip()
    elif thinking:
        # Generation starts inside <think>. Never execute a call from unfinished reasoning.
        return {"role": "assistant", "content": "", "reasoning_content": text}, []
    matches = list(CALL.finditer(text))
    remainder = CALL.sub("", text)
    malformed = "<tool_call>" in remainder or "</tool_call>" in remainder
    if truncated or malformed or (matches and text[matches[-1].end() :].strip()):
        return {"role": "assistant", "content": text.strip(), "reasoning_content": reasoning}, []
    calls = [
        {
            "id": f"call_{uuid.uuid4().hex}",
            "type": "function",
            "function": {
                "name": "bash",
                "arguments": json.dumps({"command": match["command"]}),
            },
        }
        for match in matches
    ]
    message = {"role": "assistant", "content": remainder.strip(), "reasoning_content": reasoning}
    if calls:
        message["tool_calls"] = calls
    return message, calls

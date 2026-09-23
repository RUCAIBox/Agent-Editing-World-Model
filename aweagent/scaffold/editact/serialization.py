"""Model-facing serialization matching the reference inference scaffolds."""

import json
import re
from dataclasses import asdict, is_dataclass
from typing import Any

from aweagent.core.agent.trajectory import Action
from aweagent.core.llm.types import LLMResponse, Message


def text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def reasoning(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return reasoning(value.get("text") or value.get("content"))
    if isinstance(value, list):
        return "\n".join(part for item in value if (part := reasoning(item)))
    return ""


def name(value: str, *, judge=False, search=False) -> str:
    if search:
        return value
    aliases = (
        {"execute_bash": "bash", "str_replace": "str_replace_editor"}
        if judge
        else {
            "bash": "execute_bash",
            "str_replace": "str_replace_editor",
        }
    )
    return aliases.get(value.strip(), value.strip())


def arguments(value, *, search=False, tool_name=""):
    if not isinstance(value, str):
        return value
    try:
        result = json.loads(value)
    except ValueError:
        if search:
            return {"queries": [value]} if tool_name == "web_search" else {}
        return value
    return result if not search or isinstance(result, dict) else {}


def render_call(call: dict, *, judge=False, search=False) -> str:
    function = call.get("function", call)
    tool_name = name(function.get("name", ""), judge=judge, search=search)
    args = arguments(function.get("arguments", {}), search=search, tool_name=tool_name)
    return f"- {tool_name}({json.dumps(args, ensure_ascii=False, sort_keys=judge)})"


def history(messages: list[Message], *, judge=False, search=False) -> str:
    blocks = []
    visible = 0
    for index, message in enumerate(messages):
        if message.role == "system":
            continue
        visible += 1
        attributes = [f"message_index={index if judge else visible}", f"role={message.role}"]
        if not search:
            if message.name:
                attributes.append(f"name={name(message.name, judge=judge)}")
            if message.tool_call_id:
                attributes.append(f"tool_call_id={message.tool_call_id}")
        lines = ["[" + " ".join(attributes) + "]"]
        thought = message.reasoning_raw if search else reasoning(message.reasoning_raw)
        if message.role == "assistant" and (thought is not None if search else bool(thought)):
            lines.append(f"assistant_reasoning: {thought}")
        content = text(message.content)
        if judge and isinstance(message.content, str):
            content = content.strip()
        if content or (not judge and message.role != "assistant"):
            label = "tool_observation" if message.role == "tool" else message.role + "_content"
            lines.append(f"{label}: {content}")
        if message.tool_calls and (judge or message.role == "assistant"):
            lines.append("tool_calls:")
            lines.extend(
                render_call(call.to_dict(), judge=judge, search=search)
                for call in message.tool_calls
            )
        block = "\n".join(lines)
        blocks.append(block.strip() if judge else block)
    return "\n\n".join(blocks) or ("" if search else "(empty)")


def candidate(action: Action, index: int, *, selected=None, search=False) -> str:
    lines = [f"[message_index={index} role=assistant]"]
    thought = (
        action.reasoning_text
        if search
        else (text(action.reasoning_text).strip() or reasoning(action.reasoning_raw))
    )
    if thought:
        lines.append(
            ("agent_thought: " if selected is not None else "assistant_reasoning: ") + thought
        )
    content = text(action.content)
    if selected is not None and isinstance(action.content, str):
        content = content.strip()
    if content:
        lines.append("assistant_content: " + content)
    calls = [
        render_call(call, judge=selected is not None, search=search) for call in action.tool_calls
    ]
    if selected is None:
        if calls:
            lines.extend(["tool_calls:", *calls])
    else:
        lines.extend(["selected_tool_call:", calls[selected]])
        siblings = [call for i, call in enumerate(calls) if i != selected]
        lines.append(
            "sibling_tool_calls:\n" + "\n".join(siblings)
            if siblings
            else "sibling_tool_calls: (none)"
        )
    return "\n".join(lines)


def tag(content: str | None, key: str) -> str:
    match = re.search(rf"<{key}>\s*(.*?)\s*</{key}>", content or "", re.I | re.S)
    return match.group(1).strip() if match else ""


def action_type(content: str | None, *, search=False) -> str | None:
    value = content or ""
    label = tag(value, "action_type").lower()
    labels = ("critical", "exploratory", "noisy")
    if label in labels:
        return label
    if search:
        value = label or value
    found = [s for s in labels if re.search(rf"\b{s}\b", value.lower())]
    if found and (search or len(found) == 1):
        return found[0]
    if not search:
        raise ValueError("Action Judge returned no unambiguous action_type")
    return None


def revision_thought(response: LLMResponse) -> str:
    return (
        reasoning(response.reasoning_text)
        or reasoning(response.reasoning_raw)
        or tag(response.content, "think")
    )


def snapshot(value: Action | LLMResponse) -> dict:
    usage = asdict(value.usage) if is_dataclass(value.usage) else value.usage
    return {
        "content": value.content,
        "reasoning_text": value.reasoning_text,
        "reasoning_raw": value.reasoning_raw,
        "tool_calls": [c.to_dict() if hasattr(c, "to_dict") else c for c in value.tool_calls],
        "usage": usage,
        "finish_status": value.finish_status,
        "finish_reason": getattr(value, "finish_reason", None),
    }

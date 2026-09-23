"""Parse native revisions using the reference domain-specific action contract."""

import json
import uuid

from aweagent.scaffold.editact import serialization as render


def parse_calls(response, *, search, allow_parallel, available):
    calls = [call.to_dict() for call in response.tool_calls]
    if search and not calls:
        # The native Search scaffold also accepts its earlier text action format.
        content = render.tag(response.content, "action") or response.content or ""
        decoder = json.JSONDecoder()
        for index, char in enumerate(content):
            if char != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(content[index:])
            except ValueError:
                continue
            if isinstance(payload, dict):
                calls = [payload]
                break
    if not calls or (len(calls) != 1 and not (search and allow_parallel)):
        raise ValueError(f"Expected exactly one native tool call; got {len(calls)}")
    result = []
    for call in calls:
        function = call.get("function", call)
        name = function.get("name") or function.get("tool_name") or function.get("tool")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Generated action is missing tool_name")
        name = name.strip()
        raw = function.get("arguments", function.get("args", {}))
        if search:
            name = {"web_search": "search_api", "web_extractor": "link_summary_tool"}.get(
                name, name
            )
            args = render.arguments(raw, search=True, tool_name=name)
            if not isinstance(args, dict):
                args = {}
            if name == "link_summary_tool" and "prompt" not in args and "question" in args:
                args["prompt"] = args["question"]
            if name not in {"search_api", "link_summary_tool"}:
                raise ValueError("Search revision must be a search/read action")
        else:
            name = render.name(name)
            args = json.loads(raw) if isinstance(raw, str) else raw
            validate_code(name, args)
            if name == "execute_bash" and name not in available and "bash" in available:
                name = "bash"
        if name not in available:
            raise ValueError(f"State Revision returned unavailable tool: {name}")
        result.append(
            {
                "id": "call_editact_" + uuid.uuid4().hex,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }
        )
    return result


def validate_code(name, args):
    if not isinstance(args, dict):
        raise ValueError("Tool arguments must be a JSON object")
    if name == "execute_bash":
        if not isinstance(args.get("command"), str) or not args["command"].strip():
            raise ValueError("Bash command must not be empty")
        timeout = args.get("timeout")
        if timeout is not None and (
            isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0
        ):
            raise ValueError("Bash timeout must be positive")
    elif name == "str_replace_editor":
        command = args.get("command")
        if command not in {"view", "create", "str_replace", "insert"}:
            raise ValueError("Invalid editor command")
        if not isinstance(args.get("path"), str) or not args["path"].startswith("/"):
            raise ValueError("Editor path must be absolute")
        if command == "create" and not isinstance(args.get("file_text"), str):
            raise ValueError("Editor create requires file_text")
        if command == "str_replace" and (
            not isinstance(args.get("old_str"), str)
            or not args["old_str"]
            or not isinstance(args.get("new_str"), str)
        ):
            raise ValueError("Editor replacement requires old_str and new_str")
        if command == "insert" and (
            isinstance(args.get("insert_line"), bool)
            or not isinstance(args.get("insert_line"), int)
            or not isinstance(args.get("new_str"), str)
        ):
            raise ValueError("Editor insert requires insert_line and new_str")
    elif name != "finish" or args:
        raise ValueError("Unsupported revision tool or nonempty finish arguments")


def signature(calls):
    return json.dumps(
        sorted(
            json.dumps(
                {
                    "name": render.name(call["function"]["name"]),
                    "arguments": json.loads(call["function"]["arguments"]),
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            for call in calls
        ),
        ensure_ascii=False,
    )

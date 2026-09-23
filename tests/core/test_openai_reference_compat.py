"""Reference Qwen parser recovery and per-step generation parameter handling."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aweagent.core.llm.backends.openai import OpenAIBackend
from aweagent.core.llm.client import LLMClient
from aweagent.core.llm.config import LLMConfig
from aweagent.core.llm.types import Message


def backend(**kwargs):
    instance = object.__new__(OpenAIBackend)
    instance.config = LLMConfig(model="test", **kwargs)
    return instance


def test_runtime_limit_overrides_configured_alias_and_keeps_sampling_body():
    api = backend(
        params={
            "max_completion_tokens": 32768,
            "extra_body": {"top_k": 20, "enable_thinking": True},
            "thinking": {"type": "enabled"},
        }
    )
    request = api._build_params(
        [Message("user", "Task")], max_tokens=65500, extra_body={"min_p": 0.0}
    )
    assert request["max_tokens"] == 65500
    assert "max_completion_tokens" not in request
    assert request["extra_body"] == {
        "top_k": 20,
        "enable_thinking": True,
        "min_p": 0.0,
        "thinking": {"type": "enabled"},
    }
    assert "min_p" not in api.config.params["extra_body"]


async def test_client_preserves_dynamic_limit_through_both_merge_layers():
    api = backend(
        params={
            "max_completion_tokens": 32768,
            "extra_body": {"top_k": 20, "enable_thinking": True},
        }
    )
    create = AsyncMock(return_value=response("Thought", content="Answer"))
    api._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = object.__new__(LLMClient)
    client.config, client._backend = api.config, api
    await client.chat([Message("user", "Task")], max_tokens=65500, extra_body={"min_p": 0.0})
    request = create.call_args.kwargs
    assert request["max_tokens"] == 65500
    assert "max_completion_tokens" not in request
    assert request["extra_body"] == {"top_k": 20, "enable_thinking": True, "min_p": 0.0}
    assert api.config.params == {
        "max_completion_tokens": 32768,
        "extra_body": {"top_k": 20, "enable_thinking": True},
    }


def response(reasoning, content=None, tool_calls=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content, reasoning_content=reasoning, tool_calls=tool_calls
                ),
                finish_reason="stop",
            )
        ],
        usage=None,
    )


XML = (
    "Inspect the repository.\n<tool_call><function=execute_bash>"
    "<parameter=command>pwd</parameter></function></tool_call>"
)


def test_recovers_complete_qwen_tool_call_from_reasoning_without_duplication():
    result = backend()._parse_response(response(XML))
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "execute_bash"
    assert json.loads(result.tool_calls[0].arguments) == {"command": "pwd"}
    assert result.reasoning_text == "Inspect the repository."
    assert result.reasoning_raw == "Inspect the repository."
    assert result.content is None


def test_does_not_recover_quoted_tools_when_normal_content_or_calls_exist():
    result = backend()._parse_response(response(XML, content="Explaining a call"))
    assert result.tool_calls == []
    assert result.reasoning_raw == XML
    call = SimpleNamespace(id="real", function=SimpleNamespace(name="finish", arguments="{}"))
    result = backend()._parse_response(response(XML, tool_calls=[call]))
    assert [c.name for c in result.tool_calls] == ["finish"]


def test_incomplete_qwen_tool_call_is_not_executed():
    result = backend()._parse_response(response(XML[:-12]))
    assert result.tool_calls == []


def test_revision_reasoning_is_serialized_once():
    api = backend(reasoning={"preserve": True, "format": "reasoning_content"})
    request = api._build_params([Message("assistant", "", reasoning_raw="Verify the source.")])
    assert request["messages"] == [
        {"role": "assistant", "content": "", "reasoning_content": "Verify the source."}
    ]

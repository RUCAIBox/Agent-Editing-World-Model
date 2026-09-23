"""Regression coverage for the reference inference protocol, without live services."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aweagent.core.agent.trajectory import Action
from aweagent.core.llm.types import LLMResponse, Message, TokenUsage, ToolCall
from aweagent.scaffold.editact import serialization as render
from aweagent.scaffold.editact_for_search.agent import EditActSearchAgent, SearchToolAlias
from aweagent.scaffold.editact_for_search.context import FOLD_TEXT, SearchObservationCondenser
from aweagent.scaffold.editact_for_swe.agent import EditActSWEAgent
from aweagent.scaffold.editact_terminal.agent import EditActTerminalAgent
from tests.scaffold.test_editact import action, context, make_agent


def search_context(agent, responses, steps=4):
    agent._condenser._tokenizer = SimpleNamespace(encode=lambda text: list(text))
    ctx = context(agent)
    ctx.llm = SimpleNamespace(chat=AsyncMock(side_effect=responses))
    ctx.max_steps = steps
    for tool in ctx.tools:
        tool.execute = AsyncMock(return_value="real observation")
    return ctx


async def test_search_loop_dynamic_budget_reasoning_writeback_and_actor_final_answer():
    agent = make_agent(EditActSearchAgent)
    agent._judge.chat.return_value = LLMResponse(content="<action_type>noisy</action_type>")
    agent._revision.chat.return_value = LLMResponse(
        reasoning_text="Verify the date.",
        tool_calls=[
            ToolCall("wm", "link_summary_tool", '{"url":"https://example.org", "prompt":"Date"}')
        ],
    )
    ctx = search_context(
        agent,
        [
            LLMResponse(tool_calls=[ToolCall("actor", "search_api", '{"query":"old"}')]),
            LLMResponse(tool_calls=[ToolCall("done", "finish", '{"answer":"2000"}')]),
        ],
    )
    agent.settings.max_context_tokens = 3000
    judge, revision = agent._judge, agent._revision
    result = await agent.create_loop(ctx).run("Which year?")
    assert result.finish_reason == "finish"
    assert result.metadata["final_answer"] == "2000"
    assert result.metadata["stats"]["steps"] == 2
    for request in ctx.llm.chat.call_args_list:
        messages = request.kwargs["messages"]
        expected = 3000 - sum(len(str(m.content or "")) for m in messages)
        assert request.kwargs["max_tokens"] == expected
    written = result.messages[2]
    assert written.reasoning_raw == "Verify the date."
    assert written.content == ""
    assert written.tool_calls[0].name == "link_summary_tool"
    ctx.get_tool("search_api").execute.assert_not_called()
    ctx.get_tool("link_summary_tool").execute.assert_awaited_once()
    ctx.get_tool("finish").execute.assert_not_called()
    judge.chat.assert_awaited_once()
    revision.chat.assert_awaited_once()
    judge.close.assert_awaited_once()


async def test_search_error_is_not_relabelled_as_successful_forced_answer():
    agent = make_agent(EditActSearchAgent)
    agent.actor._force_final_answer = True
    agent.actor._rollout_retries = 2
    ctx = search_context(agent, [ConnectionError("offline")])
    result = await agent.create_loop(ctx).run("Question")
    assert result.finish_reason == "error"
    assert result.error == "offline"
    assert result.metadata["final_answer"] == ""
    ctx.llm.chat.assert_awaited_once()


async def test_search_message_only_receives_reference_reminder_and_not_empty_retries():
    agent = make_agent(EditActSearchAgent)
    ctx = search_context(
        agent,
        [
            LLMResponse(),
            LLMResponse(
                tool_calls=[ToolCall("done", "finish", '{"answer":"x"}')],
            ),
        ],
    )
    result = await agent.create_loop(ctx).run("Question")
    request = ctx.llm.chat.call_args_list[1].kwargs
    assert request["messages"][-1].role == "user"
    assert request["messages"][-1].content == agent.get_no_tool_call_prompt()
    assert len(result.trajectory.steps) == 2


async def test_search_context_guard_only_requests_finish():
    agent = make_agent(EditActSearchAgent)
    agent.actor._force_final_answer = True
    agent.settings.max_context_tokens = 4000
    ctx = search_context(
        agent,
        [
            LLMResponse(
                tool_calls=[
                    ToolCall("done", "finish", '{"answer":"final"}'),
                ]
            )
        ],
    )
    result = await agent.create_loop(ctx).run("Question")
    request = ctx.llm.chat.call_args.kwargs
    assert [t["function"]["name"] for t in request["tools"]] == ["finish"]
    assert request["messages"][-1].name == "hint"
    assert result.finish_reason == "finish"
    assert result.metadata["answer_provenance"]["context_guard_triggered"]


async def test_search_last_step_adds_hint_without_implicit_extra_rollout():
    agent = make_agent(EditActSearchAgent)
    ctx = search_context(agent, [LLMResponse(content="Still thinking")], steps=1)
    result = await agent.create_loop(ctx).run("Question")
    ctx.llm.chat.assert_awaited_once()
    assert ctx.llm.chat.call_args.kwargs["messages"][-1].name == "hint"
    assert result.finish_reason == "max_steps"
    assert result.metadata["final_answer"] == ""


async def test_fold_never_truncates_the_two_latest_observations():
    folder = SearchObservationCondenser(maximum=50, target=10)
    folder._tokenizer = SimpleNamespace(encode=lambda text: list(text))
    messages = [Message(role="tool", content=str(i) * 100) for i in range(3)]
    folded = await folder.condense(messages)
    assert [m.content for m in folded] == [FOLD_TEXT, "1" * 100, "2" * 100]
    assert await folder.condense(messages[1:]) == messages[1:]


@pytest.mark.parametrize("cls", [EditActTerminalAgent, EditActSWEAgent])
async def test_invalid_revision_does_not_trigger_format_repair(cls):
    agent = make_agent(cls)
    agent._judge.chat.return_value = LLMResponse(content="noisy")
    agent._revision.chat.return_value = LLMResponse(
        reasoning_text="An invalid action", tool_calls=[ToolCall("bad", "execute_bash", "{}")]
    )
    original = action()
    assert await agent.edit(original, context(agent)) is original
    agent._revision.chat.assert_awaited_once()


@pytest.mark.parametrize("cls", [EditActSearchAgent, EditActTerminalAgent, EditActSWEAgent])
async def test_valid_native_action_does_not_require_replacement_reasoning(cls):
    agent = make_agent(cls)
    agent._judge.chat.return_value = LLMResponse(content="noisy")
    tool = "search_api" if cls == EditActSearchAgent else "bash"
    args = {"query": "date"} if cls == EditActSearchAgent else {"command": "pwd", "timeout": 0.5}
    agent._revision.chat.return_value = LLMResponse(
        tool_calls=[ToolCall("r", tool, json.dumps(args))]
    )
    original = action("search_api", {"query": "old"}) if cls == EditActSearchAgent else action()
    edited = await agent.edit(original, context(agent))
    assert edited is not original
    assert edited.content == ""
    assert edited.reasoning_text == ""


async def test_search_text_action_compatibility_and_question_argument():
    agent = make_agent(EditActSearchAgent)
    agent._judge.chat.return_value = LLMResponse(content="noisy")
    agent._revision.chat.return_value = LLMResponse(
        content=(
            '<think>Check date.</think><action>{"tool_name":"web_extractor",'
            '"arguments":{"url":"https://example.org", "question":"Date?"}}</action>'
        )
    )
    result = await agent.edit(action("search_api", {"query": "old"}), context(agent))
    call = result.tool_calls[0]["function"]
    assert call["name"] == "link_summary_tool"
    assert json.loads(call["arguments"])["prompt"] == "Date?"


async def test_page_observation_wrapper_and_list_aliases():
    base = SimpleNamespace(execute=AsyncMock(return_value="Extract"))
    tool = SearchToolAlias(base, "link_summary_tool")
    result = await tool.execute(
        {"urls": ["https://example.org/a", "https://example.org/b"], "goal": "Date"}
    )
    assert (
        result
        == "Extracted content for: https://example.org/a\n\nExtract\n\nExtracted content for: https://example.org/b\n\nExtract"
    )
    assert base.execute.await_count == 2


def test_history_formats_keep_reference_indices_ids_and_reasoning():
    messages = [
        Message(role="system", content="private system"),
        Message(role="user", content=" Task "),
        Message(
            role="assistant",
            content="",
            reasoning_raw=[{"text": " Verify "}],
            tool_calls=[ToolCall("c", "execute_bash", '{"command":"pwd"}')],
        ),
        Message(role="tool", name="execute_bash", tool_call_id="c", content=" /workspace "),
    ]
    judge = render.history(messages, judge=True)
    assert (
        "[message_index=3 role=tool name=bash tool_call_id=c]\ntool_observation: /workspace"
        in judge
    )
    assert "assistant_reasoning: Verify" in judge
    sr = render.history(messages)
    assert (
        "[message_index=3 role=tool name=execute_bash tool_call_id=c]\n"
        "tool_observation:  /workspace "
        in sr
    )
    search = render.history(messages, search=True)
    assert "tool_call_id=" not in search
    assert "name=" not in search
    assert "private system" not in search


async def test_terminal_no_added_128k_context_cutoff():
    agent = make_agent(EditActTerminalAgent)
    agent.actor.step = AsyncMock(
        side_effect=[
            Action(
                type="tool_call",
                tool_calls=[ToolCall("c", "execute_bash", '{"command":"pwd"}').to_dict()],
                usage=TokenUsage(180000, 2000, 182000),
            ),
            Action(type="finish"),
        ]
    )
    agent._judge.chat.return_value = LLMResponse(content="critical")
    ctx = context(agent)
    ctx.max_steps = 2
    ctx.task_info["skip_patch_extraction"] = True
    ctx.get_tool("execute_bash").execute = AsyncMock(return_value="done")
    result = await agent.create_loop(ctx).run("Task")
    assert result.finish_reason == "finish"
    assert agent.actor.step.await_count == 2


async def test_benchmark_trajectory_persists_world_model_audit():
    from aweagent.core.agent.loop import AgentResult
    from aweagent.core.task.runner import TaskResult, _build_trajectory_record

    agent = make_agent(EditActTerminalAgent)
    agent._judge.chat.return_value = LLMResponse(content="critical", usage=TokenUsage(10, 2, 12))
    ctx = context(agent)
    chosen = await agent.edit(action(), ctx)
    ctx.trajectory.add_step(step=0, action=chosen)
    result = TaskResult(instance_id="test", agent_result=AgentResult(trajectory=ctx.trajectory))
    record = _build_trajectory_record(result)
    event = record["trajectory"][0]["action"]["metadata"]["editact"]
    assert event["judge_results"][0]["response"]["usage"]["total_tokens"] == 12
    assert event["original_action"]["tool_calls"] == chosen.tool_calls
    json.dumps(record)

"""EditAct routing, failure handling, history and native action contracts."""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aweagent.core.agent.context import AgentContext
from aweagent.core.agent.trajectory import Action
from aweagent.core.config.schema import AweAgentConfig
from aweagent.core.llm.types import LLMResponse, Message, TokenUsage, ToolCall
from aweagent.scaffold.editact.serialization import action_type
from aweagent.scaffold.editact_for_search.agent import EditActSearchAgent
from aweagent.scaffold.editact_for_swe.agent import EditActSWEAgent
from aweagent.scaffold.editact_terminal.agent import EditActTerminalAgent

CLASSES = [EditActSearchAgent, EditActTerminalAgent, EditActSWEAgent]


def make_agent(cls):
    config = AweAgentConfig(
        agent={
            "type": {
                EditActSearchAgent: "editact_for_search",
                EditActTerminalAgent: "editact_terminal",
                EditActSWEAgent: "editact_for_swe",
            }[cls],
            "force_final_answer": False,
            "tool_options": {
                "editact": {
                    "world_model": {
                        "model": "test-wm",
                        "api_key": "test-only",
                        "base_url": "http://localhost:30000/v1",
                    }
                }
            },
        }
    )
    agent = cls.from_config(config)
    agent._judge = SimpleNamespace(chat=AsyncMock(), close=AsyncMock())
    agent._revision = SimpleNamespace(chat=AsyncMock(), close=AsyncMock())
    return agent


def action(name="execute_bash", args=None):
    return Action(
        type="tool_call",
        content="original thought",
        reasoning_text="proposal reasoning",
        tool_calls=[ToolCall("actor1", name, json.dumps(args or {"command": "pwd"})).to_dict()],
        usage=TokenUsage(30, 5, 35),
    )


def context(agent):
    return AgentContext(
        llm=SimpleNamespace(),
        tools=agent.get_tools(),
        messages=[
            Message(role="system", content="Actor system"),
            Message(role="user", content="Task"),
        ],
    )


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("label", ["critical", "exploratory"])
async def test_retain_productive_actions(cls, label):
    agent = make_agent(cls)
    agent._judge.chat.return_value = LLMResponse(content=f"<action_type>{label}</action_type>")
    proposal = action("search_api", {"query": "fact"}) if cls == EditActSearchAgent else action()
    ctx = context(agent)
    result = await agent.edit(proposal, ctx)
    assert result is proposal
    agent._revision.chat.assert_not_called()
    assert len(ctx.messages) == 2
    assert ctx.trajectory.metadata["editact"][0]["intervention"] == "none"


@pytest.mark.parametrize("cls", CLASSES)
async def test_rewrite_noisy_before_execution(cls):
    agent = make_agent(cls)
    name, args = (
        ("search_api", {"query": "verified clue"})
        if cls == EditActSearchAgent
        else ("execute_bash", {"command": "printf verified"})
    )
    proposal = action(name, args)
    agent._judge.chat.return_value = LLMResponse(content="<action_type>noisy</action_type>")
    agent._revision.chat.return_value = LLMResponse(
        content="<think>Check the missing evidence.</think>",
        tool_calls=[ToolCall("wm1", name, json.dumps(args))],
    )
    ctx = context(agent)
    result = await agent.edit(proposal, ctx)
    assert result.reasoning_text == "Check the missing evidence."
    assert result.content == ""
    assert result.reasoning_raw == result.reasoning_text
    assert result.tool_calls[0]["id"] != "wm1"
    assert result.usage == proposal.usage
    assert len(ctx.messages) == 2
    assert ctx.trajectory.metadata["editact"][0]["intervention"] == "rewrite"
    assert "tool_choice" not in agent._revision.chat.call_args.kwargs
    assert "Actor system" not in agent._judge.chat.call_args.kwargs["messages"][1].content


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("stage", ["judge", "revision"])
async def test_service_failure_retains_proposal(cls, stage):
    agent = make_agent(cls)
    proposal = action("search_api", {"query": "fact"}) if cls == EditActSearchAgent else action()
    agent._judge.chat.return_value = LLMResponse(content="<action_type>noisy</action_type>")
    getattr(agent, "_" + stage).chat.side_effect = ConnectionError("test failure")
    ctx = context(agent)
    assert await agent.edit(proposal, ctx) is proposal
    assert ctx.trajectory.metadata["editact"][0]["intervention"] == "fallback"


async def test_no_partial_replacement_when_parallel_call_invalid():
    agent = make_agent(EditActSWEAgent)
    agent._judge.chat.return_value = LLMResponse(content="noisy")
    agent._revision.chat.return_value = LLMResponse(
        reasoning_text="Fix it",
        tool_calls=[
            ToolCall("1", "execute_bash", '{"command": "pwd"}'),
            ToolCall("2", "str_replace_editor", "{}"),
        ],
    )
    proposal = action()
    ctx = context(agent)
    assert await agent.edit(proposal, ctx) is proposal
    assert agent._revision.chat.await_count == 1
    assert all(m.role != "assistant" for m in agent._revision.chat.call_args.kwargs["messages"])


@pytest.mark.parametrize("cls", CLASSES)
async def test_actor_finish_not_judged(cls):
    agent = make_agent(cls)
    finish = replace(
        action("finish", {"answer": "x"} if cls == EditActSearchAgent else {}), type="finish"
    )
    assert await agent.edit(finish, context(agent)) is finish
    agent._judge.chat.assert_not_called()


@pytest.mark.parametrize("cls", [EditActTerminalAgent, EditActSWEAgent])
async def test_wm_code_finish_has_no_answer(cls):
    agent = make_agent(cls)
    agent._judge.chat.return_value = LLMResponse(content="noisy")
    agent._revision.chat.return_value = LLMResponse(
        reasoning_text="Everything verified.", tool_calls=[ToolCall("1", "finish", "{}")]
    )
    result = await agent.edit(action(), context(agent))
    assert result.type == "finish"
    assert result.tool_calls[0]["function"]["arguments"] == "{}"


async def test_search_wm_cannot_submit_answer():
    agent = make_agent(EditActSearchAgent)
    agent._judge.chat.return_value = LLMResponse(content="noisy")
    agent._revision.chat.return_value = LLMResponse(
        reasoning_text="Answer it.", tool_calls=[ToolCall("1", "finish", '{"answer": "x"}')]
    )
    proposal = action("search_api", {"query": "clue"})
    assert await agent.edit(proposal, context(agent)) is proposal


@pytest.mark.parametrize(
    "args",
    [
        {"command": "create", "path": "/repo/a.py"},
        {"command": "insert", "path": "/repo/a.py", "insert_line": "one", "new_str": "a"},
        {"command": "view", "path": "relative.py"},
    ],
)
async def test_validate_revision_editor_arguments(args):
    agent = make_agent(EditActSWEAgent)
    agent._judge.chat.return_value = LLMResponse(content="noisy")
    agent._revision.chat.return_value = LLMResponse(
        reasoning_text="Edit",
        tool_calls=[
            ToolCall("1", "str_replace_editor", json.dumps(args)),
        ],
    )
    proposal = action()
    assert await agent.edit(proposal, context(agent)) is proposal


def test_ambiguous_judge_rejected():
    for text in (
        "critical or noisy",
        "",
        "<action_type>wrong</action_type>",
    ):
        with pytest.raises(ValueError):
            action_type(text)
    assert action_type("Not exploratory. <action_type>CRITICAL</action_type>") == "critical"
    assert (
        action_type("<action_type>noisy</action_type><action_type>critical</action_type>")
        == "noisy"
    )


async def test_loop_commits_only_edited_action_and_closes_clients():
    agent = make_agent(EditActTerminalAgent)
    agent.actor.step = AsyncMock(side_effect=[action(), Action(type="finish")])
    agent._judge.chat.return_value = LLMResponse(content="noisy")
    agent._revision.chat.return_value = LLMResponse(
        reasoning_text="Use the verified location.",
        tool_calls=[ToolCall("1", "execute_bash", '{"command": "printf verified"}')],
    )
    ctx = context(agent)
    ctx.max_steps = 3
    ctx.task_info["skip_patch_extraction"] = True
    tool = ctx.get_tool("execute_bash")
    tool.execute = AsyncMock(return_value="verified")
    judge, revision = agent._judge, agent._revision
    result = await agent.create_loop(ctx).run("Task")
    assert result.finish_reason == "finish"
    tool.execute.assert_awaited_once()
    assert tool.execute.call_args.args[0] == {"command": "printf verified"}
    assert any(
        m.reasoning_raw == "Use the verified location." and m.content == "" for m in result.messages
    )
    assert not any(m.content == "original thought" for m in result.messages)
    judge.close.assert_awaited_once()
    revision.close.assert_awaited_once()


async def test_none_usage_does_not_crash_loop():
    agent = make_agent(EditActTerminalAgent)
    agent.actor.step = AsyncMock(
        return_value=Action(type="finish", usage=TokenUsage(None, 1, None))
    )
    ctx = context(agent)
    ctx.task_info["skip_patch_extraction"] = True
    result = await agent.create_loop(ctx).run("Task")
    assert result.finish_reason == "finish"


async def test_terminal_initialization_persists_in_history():
    agent = make_agent(EditActTerminalAgent)
    ctx = context(agent)
    ctx.task_info = {"instruction": "Create the requested file", "workdir": "/workspace"}
    ctx.llm = SimpleNamespace(
        chat=AsyncMock(return_value=LLMResponse(tool_calls=[ToolCall("done", "finish", "{}")]))
    )
    await agent.step(ctx)
    initial = [m.content for m in ctx.messages if m.role == "user"]
    assert any("Create the requested file" in text for text in initial)
    ctx.current_step = 1
    await agent.step(ctx)
    assert [m.content for m in ctx.messages if m.role == "user"] == initial


async def test_revision_client_constructor_failure_falls_back(monkeypatch):
    agent = make_agent(EditActSWEAgent)
    agent._revision = None
    agent._judge.chat.return_value = LLMResponse(content="noisy")

    def fail(config):
        raise RuntimeError("invalid provider initialization")

    monkeypatch.setattr("aweagent.scaffold.editact.agent.LLMClient", fail)
    proposal = action()
    assert await agent.edit(proposal, context(agent)) is proposal


@pytest.mark.parametrize(
    "name,args",
    [
        ("execute_bash", {"command": "   "}),
        ("finish", {"answer": "not a code artifact"}),
        ("str_replace_editor", {"command": "str_replace", "path": "/repo/a", "old_str": ""}),
    ],
)
async def test_semantically_empty_or_wrong_domain_arguments_rejected(name, args):
    agent = make_agent(EditActSWEAgent)
    agent._judge.chat.return_value = LLMResponse(content="noisy")
    agent._revision.chat.return_value = LLMResponse(
        reasoning_text="Revision", tool_calls=[ToolCall("1", name, json.dumps(args))]
    )
    proposal = action()
    assert await agent.edit(proposal, context(agent)) is proposal


async def test_only_empty_revision_is_retried_with_unchanged_request():
    agent = make_agent(EditActSWEAgent)
    agent._judge.chat.return_value = LLMResponse(content="noisy")
    agent._revision.chat.side_effect = [
        LLMResponse(),
        LLMResponse(
            reasoning_text="Inspect the directory",
            tool_calls=[ToolCall("2", "execute_bash", '{"command": "pwd"}')],
        ),
    ]
    ctx = context(agent)
    result = await agent.edit(action(), ctx)
    assert result.reasoning_text == "Inspect the directory"
    assert ctx.trajectory.metadata["editact"][0]["intervention"] == "rewrite"
    assert len(ctx.trajectory.metadata["editact"][0]["revision_attempts"]) == 2
    assert agent._revision.chat.call_args_list[0] == agent._revision.chat.call_args_list[1]


async def test_cancellation_is_not_swallowed():
    import asyncio

    agent = make_agent(EditActSWEAgent)
    agent._judge.chat.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await agent.edit(action(), context(agent))


async def test_search_folding_preserves_original_history():
    from aweagent.scaffold.editact_for_search.agent import SearchObservationCondenser

    condenser = SearchObservationCondenser(maximum=20, target=10)
    condenser._tokenizer = SimpleNamespace(encode=lambda s: list(s))
    messages = [
        Message(role="user", content="Question"),
        Message(role="assistant", content="Reason", tool_calls=[ToolCall("1", "search_api", "{}")]),
        Message(role="tool", content="a" * 100, tool_call_id="1"),
        Message(
            role="assistant", content="Read", tool_calls=[ToolCall("2", "link_summary_tool", "{}")]
        ),
        Message(role="tool", content="latest", tool_call_id="2"),
        Message(role="tool", content="latest too", tool_call_id="3"),
    ]
    folded = await condenser.condense(messages)
    assert messages[2].content == "a" * 100
    assert folded[2].content == "Content folded due to space limitation"
    assert folded[2].tool_call_id == "1"
    assert folded[1] == messages[1]
    assert folded[1] is not messages[1]
    assert folded[-2].content == "latest"
    assert folded[-1].content == "latest too"


@pytest.mark.parametrize(
    "domain,expected",
    [
        ("search", EditActSearchAgent),
        ("terminal", EditActTerminalAgent),
        ("doc2repo", EditActSWEAgent),
    ],
)
def test_shipped_configs_are_registered_and_public(monkeypatch, domain, expected):
    from pathlib import Path

    from aweagent.core.config.loader import load_config
    from aweagent.scaffold.registry import agent_registry

    for role in ("ACTOR", "WM"):
        monkeypatch.setenv(role + "_MODEL", "test-model")
        monkeypatch.setenv(role + "_BASE_URL", "http://localhost:30000/v1")
        monkeypatch.setenv(role + "_API_KEY", "test-only")
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs/editact" / f"{domain}.yaml")
    agent = agent_registry.get(config.agent.type).from_config(config)
    assert isinstance(agent, expected)
    if domain == "search":
        assert {t.name for t in agent.get_tools()} == {"search_api", "link_summary_tool", "finish"}
        assert config.agent.tool_options["web_search"]["backend"] == "serpapi"
        assert config.agent.tool_options["web_fetch"]["reader_backend"] == "jina"
    else:
        assert config.runtime.backend == "docker"


def test_doc2repo_task_filter_does_not_include_other_swe_tasks():
    from aweagent.tasks.beyond_swe.task import BeyondSWETask

    records = [
        {"instance_id": "doc", "task": "doc2repo"},
        {"instance_id": "fix", "task": "domainfix"},
    ]
    assert len(BeyondSWETask(instances=records).get_instances()) == 2
    task = BeyondSWETask(instances=records, task_type="doc2repo")
    assert [i.id for i in task.get_instances()] == ["doc"]


async def test_stateful_condenser_runs_once_and_view_matches_judge():
    agent = make_agent(EditActTerminalAgent)
    agent._judge.chat.return_value = LLMResponse(content="critical")
    ctx = context(agent)
    ctx.task_info = {"instruction": "Original request", "workdir": "/workspace"}
    ctx.llm = SimpleNamespace(
        chat=AsyncMock(
            return_value=LLMResponse(
                tool_calls=[ToolCall("1", "execute_bash", '{"command":"pwd"}')]
            )
        )
    )
    condenser = SimpleNamespace(
        condense=AsyncMock(return_value=[Message(role="user", content="Unique condensed view")])
    )
    ctx.condenser = condenser
    await agent.step(ctx)
    condenser.condense.assert_awaited_once()
    assert ctx.condenser is condenser
    assert ctx.llm.chat.call_args.kwargs["messages"][0].content == "Unique condensed view"
    judge_input = agent._judge.chat.call_args.kwargs["messages"][1].content
    assert "Unique condensed view" in judge_input
    assert "Original request" not in judge_input
    assert any("Original request" in (m.content or "") for m in ctx.messages)


async def test_summarizer_client_ownership_on_close():
    from aweagent.core.tool.search.web_fetch_tool import WebFetchTool

    injected = SimpleNamespace(close=AsyncMock())
    borrowed = WebFetchTool(llm=injected)
    await borrowed.close()
    injected.close.assert_not_called()
    owned = WebFetchTool()
    owned._llm = injected
    await owned.close()
    injected.close.assert_awaited_once()
    assert owned._llm is None


async def test_parallel_search_revision_is_explicit_opt_in():
    agent = make_agent(EditActSearchAgent)
    agent.settings.allow_parallel_revision = True
    agent._judge.chat.return_value = LLMResponse(content="noisy")
    agent._revision.chat.return_value = LLMResponse(
        reasoning_text="Check two independent sources.",
        tool_calls=[
            ToolCall("1", "search_api", '{"query":"source one"}'),
            ToolCall("2", "search_api", '{"query":"source two"}'),
        ],
    )
    ctx = context(agent)
    result = await agent.edit(action("search_api", {"query": "old"}), ctx)
    assert len(result.tool_calls) == 2
    assert ctx.trajectory.metadata["editact"][0]["intervention"] == "rewrite"


async def test_repeated_identical_revision_falls_back():
    agent = make_agent(EditActTerminalAgent)
    agent._judge.chat.return_value = LLMResponse(content="noisy")
    agent._revision.chat.return_value = LLMResponse(
        reasoning_text="Inspect the directory.",
        tool_calls=[ToolCall("1", "execute_bash", '{"command":"pwd"}')],
    )
    proposal = action(args={"command": "printf previous"})
    ctx = context(agent)
    assert await agent.edit(proposal, ctx) is not proposal
    assert await agent.edit(proposal, ctx) is proposal
    assert await agent.edit(proposal, ctx) is proposal
    assert ctx.trajectory.metadata["editact"][-1]["fallback_reason"] == "duplicate_replacement"
    assert agent._revision.chat.await_count == 3


async def test_search_tool_alias_delegates_to_existing_tool():
    from aweagent.scaffold.editact_for_search.agent import SearchToolAlias

    original = SimpleNamespace(execute=AsyncMock(return_value="real observations"))
    alias = SearchToolAlias(original, "search_api")
    params = {"query": "target fact"}
    assert await alias.execute(params) == "real observations"
    original.execute.assert_awaited_once_with(params, session=None)

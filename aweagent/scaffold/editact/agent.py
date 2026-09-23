"""Pre-execution Action Judge and native-tool-call State Revision."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from aweagent.core.agent.context import AgentContext
from aweagent.core.agent.loop import AgentLoop, AgentResult
from aweagent.core.agent.protocol import Agent
from aweagent.core.agent.trajectory import Action
from aweagent.core.llm.client import LLMClient
from aweagent.core.llm.types import Message
from aweagent.scaffold.editact import serialization as render
from aweagent.scaffold.editact.config import EditActConfig, settings_from_config
from aweagent.scaffold.editact.prompts import search, swe, terminal
from aweagent.scaffold.editact.revision import parse_calls, signature

logger = logging.getLogger(__name__)


class EditActLoop(AgentLoop):
    async def _run_once(self, task_prompt: str) -> AgentResult:
        result = await super()._run_once(task_prompt)
        if (
            self.agent.domain == "terminal"
            and result.finish_reason == "error"
            and any(
                marker in (result.error or "").lower()
                for marker in (
                    "requested token count exceeds",
                    "maximum context length",
                    "context length",
                    "context window",
                )
            )
        ):
            result.finish_reason = "context_length"
        return result

    async def run(self, task_prompt: str) -> AgentResult:
        try:
            return await super().run(task_prompt)
        finally:
            await self.agent.close()


class _ObservedCondenser:
    """Capture the actor's exact view, including for stateful condensers."""

    def __init__(self, condenser: Any):
        self.condenser = condenser
        self.messages: list[Message] | None = None

    async def condense(self, messages: list[Message]) -> list[Message]:
        self.messages = await self.condenser.condense(messages)
        return self.messages


class EditActAgent(Agent):
    """Wrap an upstream actor without replacing its tools or execution loop."""

    domain = ""
    actor_class: type[Agent]

    @classmethod
    def from_config(cls, config):
        return cls._build(config)

    @classmethod
    def from_config_with_constraints(cls, config, instance_constraints):
        return cls._build(config, instance_constraints)

    @classmethod
    def _build(cls, config, constraints=None):
        settings = settings_from_config(config)
        if constraints is not None:
            actor = cls.actor_class.from_config_with_constraints(config, constraints)
        else:
            actor = cls.actor_class.from_config(config)
        return cls(actor, settings)

    def __init__(self, actor: Agent, settings: EditActConfig):
        self.actor = actor
        self.settings = settings
        self._judge = None
        self._revision = None
        self._last_replacement = None
        self._prompts = {"search": search, "terminal": terminal, "swe": swe}[self.domain]

    def get_system_prompt(self, task_info):
        return self.actor.get_system_prompt(task_info)

    def get_tools(self):
        return self.actor.get_tools()

    def get_tool_call_format(self):
        return self.actor.get_tool_call_format()

    def get_no_tool_call_prompt(self):
        return self.actor.get_no_tool_call_prompt()

    def get_loop_policy(self, context):
        return self.actor.get_loop_policy(context)

    def create_loop(self, context):
        return EditActLoop(self, context, policy=self.get_loop_policy(context))

    async def close(self) -> None:
        for name in ("_judge", "_revision"):
            client = getattr(self, name)
            if client is not None:
                await client.close()
                setattr(self, name, None)
        for tool in self.get_tools():
            close = getattr(tool, "close", None)
            if close is not None:
                await close()

    async def step(self, context: AgentContext) -> Action:
        if context.training is not None:
            raise ValueError("EditAct is an inference scaffold; token-level RL is not supported")
        if context.current_step == 0:
            self._last_replacement = None
        # Preserve upstream prompt initialization and capture the same history
        # used for sampling, without invoking a stateful condenser twice.
        condenser = context.condenser
        observed = _ObservedCondenser(condenser) if condenser is not None else None
        context.condenser = observed
        try:
            original = await self.actor.step(context)
        finally:
            context.condenser = condenser
        messages = (
            observed.messages if observed and observed.messages is not None else context.messages
        )
        view = replace(context, messages=list(messages), condenser=None)
        return await self.edit(original, view)

    async def edit(self, original: Action, context: AgentContext) -> Action:
        record: dict[str, Any] = {
            "domain": self.domain,
            "original_action": render.snapshot(original),
            "judge_results": [],
            "revision_attempts": [],
            "intervention": "none",
        }
        available = {tool.name: tool for tool in context.tools}
        judgeable = (
            {"search_api", "link_summary_tool", "web_search", "web_extractor"}
            if self.domain == "search"
            else {"execute_bash", "bash", "str_replace_editor", "str_replace"}
        )
        indices = [
            i
            for i, call in enumerate(original.tool_calls)
            if call.get("function", call).get("name") in judgeable
        ]
        supported = bool(indices) and (
            self.domain != "search" or len(indices) == len(original.tool_calls)
        )
        if not self.settings.enabled or not supported:
            self._last_replacement = None
            record["bypass_reason"] = "disabled_or_final_or_no_supported_action"
            return self._record(original, record, context)
        try:
            if self._judge is None:
                self._judge = LLMClient(self.settings.judge or self.settings.world_model)
            # Search classifies one decision; code classifies each proposed call.
            for index in [None] if self.domain == "search" else indices:
                response = await self._judge.chat(
                    messages=self._messages(context.messages, original, "judge", index),
                )
                entry = {"call_index": index, "response": render.snapshot(response)}
                record["judge_results"].append(entry)
                entry["action_type"] = render.action_type(
                    response.content, search=self.domain == "search"
                )
            if not any(item["action_type"] == "noisy" for item in record["judge_results"]):
                self._last_replacement = None
                return self._record(original, record, context)
        except Exception as exc:
            return self._fallback(original, record, context, "judge_failed", exc)

        allowed = judgeable | {"finish"}
        schemas = [tool.schema for tool in context.tools if tool.name in allowed]
        messages = self._messages(context.messages, original, "revision")
        try:
            if self._revision is None:
                self._revision = LLMClient(self.settings.revision or self.settings.world_model)
            attempts = 1 if self.domain == "search" else self.settings.revision_attempts
            for _ in range(attempts):
                # Empty code responses alone are retried, with the identical request.
                # Do not force tool_choice='required': reasoning APIs may reject it.
                response = await self._revision.chat(messages=messages, tools=schemas)
                record["revision_attempts"].append({"response": render.snapshot(response)})
                if response.content or response.tool_calls or response.finish_status == "length":
                    break
            replacement = self._replacement(response, original, available)
            current_signature = signature(replacement.tool_calls)
            duplicate = current_signature == self._last_replacement
            self._last_replacement = current_signature
            record["replacement_action"] = render.snapshot(replacement)
            if self.settings.reject_consecutive_duplicates and duplicate:
                record.update(intervention="fallback", fallback_reason="duplicate_replacement")
                return self._record(original, record, context)
            record["intervention"] = "rewrite"
            logger.info("EditAct %s: noisy proposal revised", self.domain)
            return self._record(replacement, record, context)
        except (ValueError, TypeError, KeyError) as exc:
            if record["revision_attempts"]:
                record["revision_attempts"][-1]["error"] = str(exc)
            return self._fallback(original, record, context, "invalid_revision", exc)
        except Exception as exc:
            return self._fallback(original, record, context, "revision_failed", exc)

    def _messages(self, messages, action, mode, index=None):
        visible = sum(m.role != "system" for m in messages)
        if mode == "judge" and self.domain != "search":
            user = self._prompts.JUDGE_USER_PROMPT.format(
                prefix_history=render.history(messages, judge=True),
                current_action=render.candidate(action, len(messages), selected=index),
            )
        else:
            tag = "history" if mode == "judge" else "current_history"
            prefix = ""
            if self.domain == "search":
                prefix = (
                    "Judge the value type of the current candidate action based only "
                    "on the history so far."
                    if mode == "judge"
                    else "Generate a better next action based only on the history so far "
                    "and the noisy candidate action."
                ) + "\n\n"
            is_search = self.domain == "search"
            user = (
                f"{prefix}<{tag}>\n{render.history(messages, search=is_search)}\n</{tag}>\n\n"
                f"<current_action>\n{render.candidate(action, visible + 1, search=is_search)}\n"
                "</current_action>"
            )
        system = (
            self._prompts.JUDGE_SYSTEM_PROMPT
            if mode == "judge"
            else self._prompts.REVISION_SYSTEM_PROMPT
        )
        return [Message(role="system", content=system), Message(role="user", content=user)]

    def _replacement(self, response, original, available):
        calls = parse_calls(
            response,
            search=self.domain == "search",
            allow_parallel=self.settings.allow_parallel_revision,
            available=available,
        )
        thought = render.revision_thought(response)
        return Action(
            type="finish" if calls[0]["function"]["name"] == "finish" else "tool_call",
            content="",
            reasoning_text=thought,
            reasoning_raw=thought if self.domain == "search" else thought or None,
            tool_calls=calls,
            usage=original.usage,
            llm_response_raw=None,
            finish_status=response.finish_status,
        )

    def _fallback(self, original, record, context, reason, exc):
        self._last_replacement = None
        record.update(
            intervention="fallback",
            fallback_reason=reason,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        logger.warning(
            "EditAct %s %s (%s); retaining actor proposal", self.domain, reason, type(exc).__name__
        )
        return self._record(original, record, context)

    @staticmethod
    def _record(action, record, context):
        record["executed_action"] = render.snapshot(action)
        record["step"] = context.current_step
        context.trajectory.metadata.setdefault("editact", []).append(record)
        action.metadata["editact"] = record
        return action

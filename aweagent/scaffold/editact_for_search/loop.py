"""Search control flow matching the evaluated native-tool-call scaffold."""

import logging
import time
import uuid
from dataclasses import replace

from aweagent.core.agent.loop import AgentLoop, AgentResult
from aweagent.core.agent.stats import RunStats
from aweagent.core.agent.trajectory import Action, Trajectory
from aweagent.core.llm.types import Message, ToolCall
from aweagent.scaffold.editact import serialization as render
from aweagent.scaffold.editact_for_search.agent import FINAL_HINT
from aweagent.scaffold.editact_for_search.context import FOLD_TEXT

logger = logging.getLogger(__name__)


def finish_answer(calls):
    for call in calls:
        function = call.get("function", {})
        if function.get("name") == "finish":
            args = render.arguments(function.get("arguments", "{}"), search=True)
            if args.get("answer") is not None:
                return str(args["answer"]).strip()
    return None


def response_action(response):
    calls = [call.to_dict() for call in response.tool_calls]
    for call in calls:
        if not call.get("id"):
            call["id"] = "call_" + uuid.uuid4().hex
    return Action(
        type="tool_call" if calls else "message",
        content=response.content or "",
        reasoning_text=response.reasoning_text,
        reasoning_raw=response.reasoning_raw,
        tool_calls=calls,
        usage=response.usage,
        finish_status=response.finish_status,
        token_ids=response.completion_token_ids,
        logprobs=response.logprobs,
        weight_version=response.weight_version,
        llm_response_raw=response.raw,
    )


def assistant_message(action):
    return Message(
        role="assistant",
        content=action.content or "",
        reasoning_raw=action.reasoning_raw,
        tool_calls=[ToolCall.from_dict(call) for call in action.tool_calls] or None,
    )


class EditActSearchLoop(AgentLoop):
    async def run(self, task_prompt):
        if self.ctx.training is not None:
            raise ValueError("EditAct is an inference scaffold; token-level RL is not supported")
        try:
            result = await self._run_once(task_prompt)
            attempts = 1
            while (
                result.finish_reason == "max_steps"
                and attempts <= self.agent.actor._rollout_retries
            ):
                attempts += 1
                result = await self._run_once(task_prompt)
            if attempts > 1:
                result.metadata["rollout_attempts"] = attempts
            if (
                self.agent.actor._rollout_retries > 0
                and result.finish_reason == "max_steps"
                and self.agent.actor._force_final_answer
            ):
                result = await self._force_final_answer(result, "max_steps_after_retries")
            return result
        finally:
            await self.agent.close()

    def _hint(self, *, at_tail=False):
        messages = self.ctx.messages[-1:] if at_tail else self.ctx.messages
        if not any(m.role == "tool" and m.name == "hint" for m in messages):
            self.ctx.messages.append(Message(role="tool", name="hint", content=FINAL_HINT))

    def _budget(self, messages):
        settings = self.agent.settings
        prompt_tokens = self.agent._condenser.count_tokens(messages)
        return prompt_tokens, max(
            1, min(settings.max_context_tokens - prompt_tokens, settings.max_output_tokens_cap)
        )

    async def _run_once(self, task_prompt):
        self.ctx.messages = [
            Message(role="system", content=self.agent.get_system_prompt(self.ctx.task_info)),
            Message(role="user", content=task_prompt),
        ]
        self.ctx.trajectory = Trajectory()
        self.agent._last_replacement = None
        stats = RunStats()
        stats.start()
        fold_step = None
        settings = self.agent.settings

        def result(reason, answer="", error=None):
            stats.finish()
            return AgentResult(
                trajectory=self.ctx.trajectory,
                messages=list(self.ctx.messages),
                finish_reason=reason,
                error=error,
                metadata={
                    "stats": stats.to_dict(),
                    "final_answer": answer,
                    "answer_provenance": {
                        "source": "editact_search",
                        "finish_tool_used": reason == "finish",
                        "forced_final_answer": False,
                        "context_fold_trigger_step": fold_step,
                    },
                },
            )

        for step in range(self.ctx.max_steps):
            self.ctx.current_step = step
            if step + 1 == self.ctx.max_steps:
                self._hint()
            inference = await self.agent._condenser.condense(self.ctx.messages)
            if fold_step is None and any(
                m.role == "tool" and m.content == FOLD_TEXT for m in inference
            ):
                fold_step = step
            prompt_tokens, maximum = self._budget(inference)
            remaining = settings.max_context_tokens - prompt_tokens
            if (
                self.agent.actor._force_final_answer
                and settings.force_final_answer_remaining_tokens
                and remaining < settings.force_final_answer_remaining_tokens
            ):
                guarded = result("context_length")
                guarded.metadata["answer_provenance"].update(
                    {
                        "context_guard_triggered": True,
                        "context_guard_step": step,
                        "context_guard_prompt_tokens": prompt_tokens,
                        "context_guard_remaining_tokens": remaining,
                        "context_guard_threshold_tokens": (
                            settings.force_final_answer_remaining_tokens
                        ),
                    }
                )
                return await self._force_final_answer(guarded, "remaining_context_below_threshold")
            try:
                started = time.monotonic()
                response = await self.ctx.llm.chat(
                    messages=inference,
                    tools=self.ctx.get_tool_schemas(),
                    max_tokens=maximum,
                )
                usage = response.usage
                stats.record_llm_call(
                    time.monotonic() - started,
                    (getattr(usage, "prompt_tokens", 0) or 0) if usage else prompt_tokens,
                    (getattr(usage, "completion_tokens", 0) or 0) if usage else 0,
                )
            except Exception as exc:
                logger.exception("EditAct Search actor failed at step %s", step)
                return result("error", error=str(exc))
            action = await self.agent.edit(
                response_action(response),
                replace(self.ctx, messages=inference, condenser=None),
            )
            trajectory_step = self.ctx.trajectory.add_step(
                step=step,
                action=action,
                reasoning_text=action.reasoning_text,
                llm_response_raw=action.llm_response_raw,
            )
            self.ctx.messages.append(assistant_message(action))
            answer = finish_answer(action.tool_calls)
            if answer is not None:
                stats.end_step()
                return result("finish", answer)
            if not action.tool_calls:
                self.ctx.messages.append(
                    Message(role="user", content=self.agent.get_no_tool_call_prompt())
                )
                stats.end_step()
                continue
            for call in action.tool_calls:
                function = call["function"]
                tool_name = function["name"]
                args = render.arguments(
                    function.get("arguments", "{}"), search=True, tool_name=tool_name
                )
                canonical = {"web_search": "search_api", "web_extractor": "link_summary_tool"}.get(
                    tool_name, tool_name
                )
                tool = self.ctx.get_tool(canonical)
                obs = (
                    await tool.execute(args, session=self.ctx.session)
                    if tool
                    else f"Error: Unknown tool {tool_name}"
                )
                trajectory_step.observations.append(obs)
                self.ctx.messages.append(
                    Message(role="tool", name=tool_name, content=obs, tool_call_id=call["id"])
                )
                stats.record_tool_call(tool_name, 0.0)
            stats.end_step()
        last = self.ctx.messages[-1]
        answer = (
            render.tag(last.content, "answer") or (last.content or "").strip()
            if last.role == "assistant"
            else ""
        )
        return result("max_steps", answer)

    async def _force_final_answer(self, result, reason):
        self._hint(at_tail=True)
        provenance = result.metadata["answer_provenance"]
        provenance.update(forced_final_answer=True, forced_final_answer_reason=reason)
        inference = await self.agent._condenser.condense(self.ctx.messages)
        _, maximum = self._budget(inference)
        tools = [t.schema for t in self.ctx.tools if t.name == "finish"]
        try:
            response = await self.ctx.llm.chat(messages=inference, tools=tools, max_tokens=maximum)
        except Exception as exc:
            provenance["forced_final_answer_error"] = str(exc)
            return result
        action = response_action(response)
        self.ctx.trajectory.add_step(
            step=len(self.ctx.trajectory.steps),
            action=action,
            reasoning_text=action.reasoning_text,
            llm_response_raw=action.llm_response_raw,
        )
        self.ctx.messages.append(assistant_message(action))
        answer = finish_answer(action.tool_calls)
        provenance["finish_tool_used"] = answer is not None
        if answer is None:
            answer = render.tag(response.content, "answer") or (response.content or "").strip()
        result.messages = list(self.ctx.messages)
        result.metadata["final_answer"] = answer
        if answer:
            result.finish_reason = "finish"
        return result

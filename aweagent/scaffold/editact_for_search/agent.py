"""Search EditAct with training-compatible names for public AweAgent tools."""

from dataclasses import replace

from aweagent.core.tool.protocol import Tool
from aweagent.scaffold.deepsearch.agent import DeepSearchAgent
from aweagent.scaffold.editact.agent import EditActAgent
from aweagent.scaffold.editact.prompts.search import ACTOR_SYSTEM_PROMPT
from aweagent.scaffold.editact_for_search.context import SearchObservationCondenser

FINAL_HINT = (
    "Do not call search_api or link_summary_tool. "
    "Use only information already retrieved and submit the final answer now "
    "by calling finish(answer=...). The answer argument must contain only the "
    "shortest final answer string."
)
NO_TOOL_CALL_PROMPT = (
    "CRITICAL: Continue by calling one of the available tools. "
    "Use search/read tools to keep investigating, or call finish(answer=...) "
    "if you are ready to submit the final answer."
)


class SearchToolAlias(Tool):
    def __init__(self, tool, name):
        self.tool = tool
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        if self.name == "search_api":
            return "Search for information from the internet."
        if self.name == "finish":
            return "Submit the final answer."
        return "Read a webpage and summarize content relevant to the prompt."

    @property
    def parameters(self):
        descriptions = {
            "search_api": {"query": "The search query."},
            "link_summary_tool": {
                "url": "The webpage URL.",
                "prompt": "The information to extract from the page.",
            },
            "finish": {"answer": "Only the shortest final answer string."},
        }[self.name]
        return {
            "type": "object",
            "properties": {
                k: {"type": "string", "description": v} for k, v in descriptions.items()
            },
            "required": list(descriptions),
        }

    async def execute(self, params, session=None):
        if self.name == "search_api":
            query = params.get("query", params.get("queries", []))
            queries = query if isinstance(query, list) else [query]
            queries = [q for q in queries if isinstance(q, str) and q.strip()]
            if not queries:
                return "Error: empty search query."
            return await self.tool.execute(
                {"query": queries if len(queries) > 1 else queries[0]}, session=session
            )
        if self.name == "link_summary_tool":
            url = params.get("url", params.get("urls", []))
            urls = url if isinstance(url, list) else [url]
            urls = [u for u in urls if isinstance(u, str) and u.strip()]
            if not urls:
                return "Error: empty URL."
            prompt = str(params.get("prompt", params.get("goal", "")))
            parts = []
            for url in urls:
                fetched = await self.tool.execute({"url": url, "prompt": prompt}, session=session)
                parts.append(f"Extracted content for: {url}\n\n{fetched}")
            return "\n\n".join(parts)
        return "Final answer submitted."

    async def close(self):
        close = getattr(self.tool, "close", None)
        if close is not None:
            await close()


class EditActSearchAgent(EditActAgent):
    domain = "search"
    actor_class = DeepSearchAgent

    def __init__(self, actor, settings):
        super().__init__(actor, settings)
        aliases = {"web_search": "search_api", "web_fetch": "link_summary_tool", "finish": "finish"}
        actor._tools = [
            SearchToolAlias(tool, aliases[tool.name]) if tool.name in aliases else tool
            for tool in actor.get_tools()
        ]
        self._condenser = SearchObservationCondenser(
            settings.tool_context_max_tokens,
            settings.tool_context_target_tokens,
            settings.tokenizer_path,
        )

    def get_system_prompt(self, task_info):
        return ACTOR_SYSTEM_PROMPT.format(tool_names=", ".join(t.name for t in self.get_tools()))

    async def step(self, context):
        from aweagent.scaffold.editact_for_search.loop import response_action

        if context.training is not None:
            raise ValueError("EditAct is an inference scaffold; token-level RL is not supported")
        if context.current_step == 0:
            self._last_replacement = None
        messages = await self._condenser.condense(context.messages)
        maximum = max(
            1,
            min(
                self.settings.max_context_tokens - self._condenser.count_tokens(messages),
                self.settings.max_output_tokens_cap,
            ),
        )
        response = await context.llm.chat(
            messages=messages,
            tools=context.get_tool_schemas(),
            max_tokens=maximum,
        )
        return await self.edit(response_action(response), replace(context, messages=messages))

    def create_loop(self, context):
        from aweagent.scaffold.editact_for_search.loop import EditActSearchLoop

        return EditActSearchLoop(self, context)

    def get_no_tool_call_prompt(self):
        return NO_TOOL_CALL_PROMPT

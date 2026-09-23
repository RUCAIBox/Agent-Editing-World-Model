"""Configuration shared by the three EditAct scaffolds."""

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt

from aweagent.core.llm.config import LLMConfig


class EditActConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    world_model: LLMConfig
    judge: LLMConfig | None = None
    revision: LLMConfig | None = None
    enabled: bool = True
    # Only empty code-domain responses are retried, without changing the prompt.
    revision_attempts: PositiveInt = 3
    allow_parallel_revision: bool = False
    reject_consecutive_duplicates: bool = True
    tool_context_max_tokens: PositiveInt = 32000
    tool_context_target_tokens: PositiveInt = 5000
    tokenizer_path: str = "Qwen/Qwen3.5-35B-A3B"
    max_context_tokens: PositiveInt = 250000
    max_output_tokens_cap: PositiveInt = 65500
    force_final_answer_remaining_tokens: NonNegativeInt = 8192


def settings_from_config(config) -> EditActConfig:
    raw = config.agent.tool_options.get("editact")
    if raw is None:
        raise ValueError("Configure agent.tool_options.editact.world_model before using EditAct")
    if config.agent.tool_call_format != "openai_function":
        raise ValueError("EditAct requires agent.tool_call_format=openai_function")
    settings = EditActConfig.model_validate(raw)
    if settings.tool_context_target_tokens > settings.tool_context_max_tokens:
        raise ValueError("tool_context_target_tokens must not exceed tool_context_max_tokens")
    if settings.allow_parallel_revision and config.agent.type != "editact_for_search":
        raise ValueError("Parallel State Revision is supported only for search")
    for llm in (settings.world_model, settings.judge, settings.revision):
        if llm is not None and any(
            "${" in str(value) for value in (llm.model, llm.base_url, llm.api_key)
        ):
            raise ValueError("Unresolved environment variable in EditAct LLM configuration")
    return settings

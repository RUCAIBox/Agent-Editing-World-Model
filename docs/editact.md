# EditAct Inference

## Execution Contract

1. The base actor proposes a reasoning/action pair from the current real history.
2. AJ classifies that proposal. Search judges the turn; Terminal/SWE judge each supported tool call in a multi-call proposal.
3. If any classified action is noisy, SR proposes a replacement from the same history. The proposed action has not been executed and no future observation or evaluation score is supplied.
4. A valid replacement replaces the whole proposed turn, including its reasoning. Otherwise the original proposal is retained.
5. The existing execution loop runs the selected tools and records their real observations.

This is state editing, not action voting or environment simulation. A standalone actor `finish` bypasses AJ. Search judges only turns containing exclusively search/read calls; Terminal/SWE judge the supported calls even in a mixed proposal. The search WM cannot submit an answer. Terminal and SWE may revise to `finish({})`, leaving artifact validation to the evaluator.

The Search prompt and native tool names follow the search AEWM interface. Terminal uses the terminal interface; the default SWE WM prompts target Doc2Repo reconstruction. The base SearchSWE actor still selects its task-specific prompt. Do not assume one trained WM/prompt combination is calibrated for every software task.

## Configuration

Add the following to an existing native-tool-call actor configuration:

```yaml
agent:
  type: editact_terminal  # or editact_for_search / editact_for_swe
  tool_call_format: openai_function
  tool_options:
    editact:
      world_model:
        backend: openai
        model: ${WM_MODEL}
        base_url: ${WM_BASE_URL}
        api_key: ${WM_API_KEY:-EMPTY}
        reasoning:
          preserve: true
        params:
          temperature: 1.0
          max_tokens: 32768
      enabled: true
      revision_attempts: 3
      allow_parallel_revision: false
      reject_consecutive_duplicates: true
      tool_context_max_tokens: 32000
      tool_context_target_tokens: 5000
```

`world_model` is required. Optional `judge` and `revision` blocks use the same LLM configuration schema and **replace**, rather than merge with, `world_model` for their respective stage. Supply full model/endpoint settings in an override. Provider retry and timeout settings belong inside these LLM blocks. `revision_attempts` limits empty-response attempts for Terminal/SWE (default three). Search makes one SR request. Neither domain adds format-repair feedback or resamples a malformed revision.

Parallel SR is opt-in and search-only; keep it disabled for BrowseComp and for comparability with single-action experiments. A parallel revision is accepted only if every call is a supported search/read action. Terminal/SWE always require one call and validate the operation-specific arguments. No tool from a rejected revision is executed. Search retains compatibility with text-wrapped action JSON and the `web_search`/`web_extractor` aliases; `question` in a page-read revision is normalized to `prompt`.

### Reference Budgets

| Domain | Actor budget | AJ output | SR output |
| --- | --- | --- | --- |
| Search | 600 steps; 250,000-token context estimate; dynamic output up to 65,500 | 1,024 | 10,240 |
| Terminal | 10,000 steps; 3h; no scaffold context cap | 8,192 | 32,768 |
| Doc2Repo | 400 steps; 245,760-token context cap; 32,768 output | 8,192 | 32,768 |

Search budgets and observation folding use the configured Qwen tokenizer. The reference context estimator counts message content, rather than serialized chat-template tokens. Above 32,000 tool-observation tokens, older whole observations are replaced with `Content folded due to space limitation`, targeting 5,000 tokens without truncating the two latest observations. As a result, the target can remain exceeded. Install `.[editact]`; set `TOKENIZER_PATH` for local/offline tokenization.

Search asks the actor to finish when fewer than 8,192 estimated context tokens remain, and adds the reference final hint at the last allowed step. A connection failure is an error, never a synthetic successful final answer. `rollout_retries` applies only to trajectories that reach the step limit. Actor reasoning must be preserved (`reasoning.preserve: true`); an edited turn has empty content and replacement reasoning in `reasoning_raw`.

The recipes include the reference sampling parameters. The Doc2Repo recipe sets `tool_choice: required` for the trained revision model; set `auto` instead when using a provider that disallows forced tool calls with reasoning. Search and Terminal leave tool choice to the model.

Public search uses the upstream SerpAPI backend and Jina Reader. Backend substitution remains available through the original tool protocols and registry. Code tools use the existing `RuntimeSession` interface, with Docker as the shipped runtime. The EditAct implementation neither imports nor configures private plugins.

For reproducibility, Terminal retains the Bash tool description used in the reference experiments; SWE retains its own tool description. This does not add persistent-shell capabilities to a runtime. Docker commands run in fresh shells: filesystem changes persist, but working-directory changes and shell variables do not. Use absolute paths or chain dependent commands when targeting Docker.

## Outputs and Failure Handling

The usual action/message history contains only the selected continuation. In-memory `trajectory.metadata.editact` stores per-decision audit records. Benchmark JSONL also persists each record under `trajectory[].action.metadata.editact`:

- `original_action` and `executed_action`;
- `judge_results`, including `action_type` and response usage when available;
- `revision_attempts`, including rejected revisions and validation errors;
- `intervention`: `none`, `rewrite`, or `fallback`;
- `fallback_reason` / `error_type`, or a `bypass_reason`.

Code-domain AJ parses the first valid action-type tag, or an unambiguous label; an unparseable response falls back. Search preserves its reference label parser and retains the actor when no noisy label is found. Native revisions are parsed before execution; a valid call does not require nonempty reasoning. Invalid SR output falls back immediately. Only empty code-domain responses are retried. Transport failure falls back after the LLM client's retries. Cancellation still propagates to the caller. A completed run is not evidence that WM guidance was healthy: inspect the audit.

WM calls have their own usage records; the actor action's `usage` is not inflated with separate AJ/SR requests, since the loop also uses it for actor context budgeting. For inference-cost accounting, sum the actor usage and every AJ/SR attempt, as well as page summarization and evaluator calls where applicable. Do not add both `original_action.usage` and `executed_action.usage`: they refer to the same actor request.

The single-task example saves JSON and a `.workspace.tar.gz` archive for code tasks. It is a smoke test, not a benchmark evaluator. Benchmark runs use the unchanged task/evaluator workflows and require their public data, task images, and test assets. See the domain recipes linked in the README.

## Testing

```bash
python -m pip install pytest pytest-asyncio
python -m pytest tests/scaffold/test_editact.py tests/scaffold/test_editact_alignment.py tests/agent/test_agent_loop.py tests/core/test_llm_backends.py tests/tasks
```

The EditAct tests do not require credentials or network services. They exercise gating, native tool validation, fallback/retry behavior, history replacement, public configuration, and task filtering. The three `examples/editact/run_*.sh` scripts exercise real endpoints and tools using credentials from the environment. Start with these before a full evaluation.

An AEWM endpoint and an actor endpoint are external dependencies. Never put their real keys in tests, sample YAML, or checked-in trajectories. Keep local smoke-test outputs under the ignored `results/` directory.

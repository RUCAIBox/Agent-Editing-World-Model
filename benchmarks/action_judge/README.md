# Action Judge Benchmark

Evaluate AEWM or any OpenAI-compatible chat model on the **3,000-decision**
Action Judge Benchmark used in the paper. Evaluation data is distributed separately
at [RUC-AIBOX/AEWM-Action-Judge-Bench](https://huggingface.co/datasets/RUC-AIBOX/AEWM-Action-Judge-Bench).
This directory contains evaluation code only. No sandbox, search credentials, training data,
or private framework extension is required: the model classifies recorded
decisions and does not execute their tools.

## Data

| Domain | Source benchmarks | Critical | Exploratory | Noisy | Total |
| --- | --- | ---: | ---: | ---: | ---: |
| Search | BrowseComp | 300 | 245 | 455 | 1,000 |
| Terminal | Terminal-Bench 2.0 | 300 | 245 | 455 | 1,000 |
| SWE | Doc2Repo (500) + NL2Repo (500) | 300 | 245 | 455 | 1,000 |
| **Overall** | | **900** | **735** | **1,365** | **3,000** |

These are individual decisions, not 3,000 distinct tasks. Each example contains
a task history ending **before** the proposed action's execution, and the
candidate reasoning and action to classify. Earlier observations in the history
are visible; the candidate's observation, future continuation, annotation
rationales, and correctness information are not included in model input.

All selected decisions have three agreeing annotations. Structural validation
checks tool arguments and matched observations in the source trajectories;
quality filtering and review exclude low-confidence cases. Search examples come
from correct trajectories; Terminal trajectories finish normally; Doc2Repo and
NL2Repo trajectories finish normally with score strictly above 0.7. The full set
is domain-balanced, but not class-balanced. The optional **balanced-core** report
uses 245 examples per domain/class cell (2,205 total); it is not the paper's
3,000-case headline result.

### Included Files

- `evaluate.py`: API evaluation, retries, resume, offline validation and scoring.
- `run.sh`: a ready-to-run launcher.

Use `--data-file` to read the separately downloaded JSONL (or JSONL.gz) file.
The alternative `--data-dir` format accepts `search.jsonl.gz`, `terminal.jsonl.gz`,
and `swe.jsonl.gz` with a `manifest.json` containing their SHA-256 checksums and
case counts. Neither format is bundled in this code repository.

Each record has this structure (abbreviated illustration, not a benchmark example):

```json
{
  "sample_id": "example_id",
  "domain": "search",
  "source_benchmark": "browsecomp",
  "messages": [
    {"role": "system", "content": "Full domain-specific Action Judge instructions..."},
    {"role": "user", "content": "Pre-action history and candidate reasoning/action..."}
  ],
  "gold_action_type": "critical",
  "balanced_core": true
}
```

**Send only `messages` to the model.** Gold labels, IDs, and split membership are
used locally for scoring, never interpolated into the prompt. The original
domain-specific prompts are embedded in each record, rather than reconstructed
from the online inference scaffold.

### Data Distribution

See the separate dataset card for release-specific provenance and preprocessing.
The evaluator preserves the supplied sample IDs, labels, and balanced-core
membership. Keep downloaded data and generated predictions outside version control.

The repository's Apache-2.0 license covers its evaluation code, not a blanket
relicensing of third-party benchmark text, webpage excerpts, or source code.
Those materials remain subject to their original licenses and terms. This is
held-out evaluation data; do not add it to training mixtures.

## Quick Start

Run from the repository root. The benchmark runner requires Python 3.11+ and
the OpenAI SDK; `tqdm` is optional for a progress bar. A full framework install
is not required.

```bash
python -m pip install 'openai>=1.66.0,<3.0.0' tqdm

# Point to the separately downloaded benchmark file.
export DATA_FILE="$HOME/datasets/AEWM-Action-Judge-Bench/action_judge_benchmark_3000.jsonl"

# Offline validation: no model or API key required.
python benchmarks/action_judge/evaluate.py --data-file "$DATA_FILE" --validate-only

export EVAL_MODEL="your-served-aewm-model"
export EVAL_BASE_URL="http://localhost:8000/v1"
export EVAL_API_KEY="EMPTY"  # Replace for an authenticated endpoint.
export OUTPUT_DIR="results/action_judge/my_model"

bash benchmarks/action_judge/run.sh
```

Defaults: **concurrency 5, temperature 1.0, max output tokens 8,192**, request
timeout 600 seconds, and at most five attempts for transient API failures.
For another model, change `EVAL_MODEL`, the endpoint, and the output directory.
The runner makes real API calls; the full evaluation can incur substantial costs.
Keep credentials in environment variables, never in committed scripts.

Smoke-test a few examples before a full run, using a separate output directory:

```bash
OUTPUT_DIR="results/action_judge/smoke" \
  bash benchmarks/action_judge/run.sh --domain search --limit 3

CONCURRENCY=10 OUTPUT_DIR="results/action_judge/my_model" \
  bash benchmarks/action_judge/run.sh
```

For serving backends that require an explicit reasoning toggle, pass the option
supported by that backend, for example:

```bash
bash benchmarks/action_judge/run.sh \
  --extra-body-json '{"chat_template_kwargs":{"enable_thinking":true}}'
```

Do not pass gold labels through extra parameters. Tool definitions are embedded
in the recorded text; the API request deliberately has no executable tools.

## Context Length

The benchmark contains long histories. The original Qwen3.5 chat-template audit
measured a mean of about **31,759 input tokens** and a maximum of **254,089**;
42 examples exceed 131,072 input tokens. Token counts vary by tokenizer and the
public-release redactions. Provision enough context for both the input and the
requested output. A 256K serving window can be tight for the very longest case
with an 8,192-token output budget. The runner never silently truncates, drops,
or substitutes over-context examples: an API rejection is reported as an error.

## Outputs and Scoring

Each output directory contains:

- `predictions.jsonl`: append-only model responses, usage when available,
  latency, parse results, and safe error type/status. Latest attempts take
  precedence when an API-failed case is retried.
- `run_config.json`: model, endpoint, decoding options, and selected-data digest;
  the API key is not stored.
- `metrics.json`: Accuracy, Macro-F1, per-class metrics, confusion matrices, and
  counts for Search, Terminal, SWE, Overall, and the balanced core.
- `report.md`: a readable results table with completion and error counts.

JSON metric values are fractions; the Markdown report displays percentages.
Overall scores pool all decisions rather than averaging domain Macro-F1s.
Macro-F1 always averages the three classes with zero division mapped to zero.
An unparseable answer, an API-failed case, or a pending case stays in the
denominator and counts as incorrect. API failures, invalid outputs, and pending
cases are listed separately, so a partial run cannot look like a completed one.

For compatibility with the paper evaluator, parsing first looks for
`<action_type>critical|exploratory|noisy</action_type>`, then a JSON-style
`action_type` field, then a single unambiguous class name in answer content.
Provider-side `reasoning_content` is retained for inspection, but is not used
as a fallback classification answer. Invalid model answers are scored as such,
not regenerated until they become valid.

Recompute the report without making API requests:

```bash
python benchmarks/action_judge/evaluate.py --data-file "$DATA_FILE" \
  --output-dir results/action_judge/my_model --score-only
```

Use the same `--domain` and `--limit` when scoring a filtered smoke run.

## Resume

The launcher enables `--resume`. Repeating the same command skips completed API
responses, including invalid-format responses, and retries only missing cases
or API failures. It flushes each completed response to disk. A changed model,
prompt set, temperature, token budget, or extra request configuration requires
a new output directory; this avoids mixing incompatible evaluations.

One writer may use an output directory at a time. Normal completion and Ctrl+C
release `.evaluation.lock`. After an uncatchable termination, confirm the
previous process has stopped before removing that empty lock directory.
There is no automatic stale-lock deletion across machines.

## Tests

```bash
python -m pip install pytest pytest-asyncio
python -m pytest tests/benchmarks/test_action_judge.py -q
```

Tests use synthetic examples and a mock API, with no paid model requests.

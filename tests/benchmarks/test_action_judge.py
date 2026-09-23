import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_module(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "benchmarks/action_judge" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluation = load_module("evaluate")


def case(label="critical", sample_id="one", domain="search"):
    messages = [
        {"role": "system", "content": "Judge this decision."},
        {"role": "user", "content": f"Task and candidate {sample_id}"},
    ]
    return {
        "sample_id": sample_id,
        "domain": domain,
        "messages": messages,
        "source_benchmark": "browsecomp",
        "gold_action_type": label,
        "balanced_core": True,
        "prompt_sha256": evaluation.digest(messages),
    }


def result(c, content, error=None):
    return {
        "sample_id": c["sample_id"],
        "prompt_sha256": c["prompt_sha256"],
        "content": content,
        "error": error,
    }


@pytest.mark.parametrize(
    "content,label,strict",
    [
        ("<action_type>CRITICAL</action_type>", "critical", True),
        ('{"action_type":"exploratory"}', "exploratory", False),
        ("This action is noisy.", "noisy", False),
        ("critical or noisy", None, False),
        ("", None, False),
        ("<action_type>unknown</action_type>", None, False),
        ("<think>critical or exploratory</think><action_type>noisy</action_type>", "noisy", True),
    ],
)
def test_parser(content, label, strict):
    assert evaluation.parse_output(content) == {
        "pred_action_type": label,
        "strict_format_ok": strict,
    }


def test_scoring_includes_api_invalid_and_pending():
    cases = [
        case(label, str(i))
        for i, label in enumerate(["critical", "exploratory", "noisy", "noisy", "critical"])
    ]
    predictions = {
        "0": result(cases[0], "critical"),
        "1": result(cases[1], "exploratory"),
        "2": result(cases[2], "ambiguous"),
        "3": result(cases[3], "noisy", {"type": "ConnectionError"}),
    }
    scores = evaluation.score(cases, predictions)
    assert scores["accuracy"] == 2 / 5
    assert scores["macro_f1"] == pytest.approx((2 / 3 + 1 + 0) / 3)
    assert (scores["api_errors"], scores["invalid_outputs"], scores["pending"]) == (1, 1, 1)


def test_overall_pools_examples():
    cases = [case("critical", "a"), case("noisy", "b", "swe"), case("noisy", "c", "swe")]
    predictions = {c["sample_id"]: result(c, "critical") for c in cases}
    metrics = evaluation.summarize(cases, predictions)
    assert metrics["overall"]["accuracy"] == 1 / 3
    assert set(metrics) == {"search", "swe", "overall", "balanced_core"}


def test_partial_prediction_recovery(tmp_path):
    c = case()
    path = tmp_path / "predictions.jsonl"
    complete = json.dumps(result(c, "critical")) + "\n"
    path.write_bytes(complete.encode() + b'{"sample_id":')
    assert evaluation.read_predictions(path, [c])["one"]["content"] == "critical"
    assert path.read_bytes().endswith(b":")
    evaluation.read_predictions(path, [c], repair_tail=True)
    assert path.read_text() == complete


def test_resume_checks_prompt(tmp_path):
    c = case()
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps({**result(c, "critical"), "prompt_sha256": "different"}) + "\n")
    with pytest.raises(ValueError, match="do not match"):
        evaluation.read_predictions(path, [c])


def test_lock_prevents_second_writer_and_cleans_up(tmp_path):
    with evaluation.output_lock(tmp_path):
        with pytest.raises(RuntimeError, match="locked"):
            with evaluation.output_lock(tmp_path):
                pytest.fail("must not enter")
    assert not (tmp_path / ".evaluation.lock").exists()


def write_data(tmp_path, cases):
    path = tmp_path / "search.jsonl.gz"
    with gzip.open(path, "wt") as handle:
        for c in cases:
            handle.write(json.dumps(c) + "\n")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    path.name: {
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "cases": len(cases),
                    }
                }
            }
        )
    )
    return path


def test_data_validation_and_checksum(tmp_path):
    path = write_data(tmp_path, [case()])
    assert len(evaluation.load_cases(tmp_path, "search")) == 1
    path.write_bytes(path.read_bytes() + b"invalid")
    with pytest.raises(ValueError, match="checksum"):
        evaluation.load_cases(tmp_path, "search")


@pytest.mark.parametrize("compressed", [False, True])
def test_external_jsonl_file_and_domain_filter(tmp_path, compressed):
    cases = [case("critical", "s"), case("noisy", "t", "terminal")]
    path = tmp_path / ("benchmark.jsonl.gz" if compressed else "benchmark.jsonl")
    opener = gzip.open if compressed else open
    with opener(path, "wt", encoding="utf-8") as handle:
        for row in cases:
            handle.write(json.dumps(row) + "\n")
    assert len(evaluation.load_cases(path)) == 2
    assert [row["sample_id"] for row in evaluation.load_cases(path, "terminal")] == ["t"]
    assert len(evaluation.load_cases(path, limit=1)) == 1


def test_external_jsonl_validates_labels_and_duplicates(tmp_path):
    path = tmp_path / "benchmark.jsonl"
    path.write_text(json.dumps(case("invalid")) + "\n")
    with pytest.raises(ValueError, match="Invalid evaluation row"):
        evaluation.load_cases(path)
    path.write_text((json.dumps(case()) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        evaluation.load_cases(path)


def test_missing_dataset_has_actionable_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="distributed separately"):
        evaluation.load_cases(tmp_path)


def test_data_file_cli(tmp_path):
    path = tmp_path / "benchmark.jsonl"
    args = evaluation.parse_args(["--validate-only", "--data-file", str(path)])
    assert args.data_file == path


def test_data_duplicate_detection(tmp_path):
    write_data(tmp_path, [case(), case()])
    with pytest.raises(ValueError, match="Duplicate"):
        evaluation.load_cases(tmp_path, "search")


def test_no_reference_message_allowed(tmp_path):
    c = case()
    c["messages"].append({"role": "assistant", "content": "critical"})
    write_data(tmp_path, [c])
    with pytest.raises(ValueError, match="Invalid evaluation"):
        evaluation.load_cases(tmp_path, "search")


def fake_response(content="<action_type>noisy</action_type>"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, reasoning_content="critical"),
                finish_reason="stop",
            )
        ],
        usage=None,
    )


@pytest.mark.asyncio
async def test_request_does_not_send_gold_or_execute_tools(tmp_path):
    args = evaluation.parse_args(["--model", "test", "--output-dir", str(tmp_path)])
    create = AsyncMock(return_value=fake_response())
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    c = case()
    prediction = await evaluation.predict(client, c, args)
    request = create.call_args.kwargs
    assert request["messages"] == c["messages"]
    assert set(request) == {"model", "messages", "temperature", "max_tokens"}
    assert prediction["pred_action_type"] == "noisy"
    assert prediction["error"] is None


@pytest.mark.asyncio
async def test_retry_transient_not_invalid_output(monkeypatch, tmp_path):
    args = evaluation.parse_args(["--model", "test", "--output-dir", str(tmp_path)])
    create = AsyncMock(
        side_effect=[ConnectionError("secret not to be stored"), fake_response("invalid")]
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(evaluation.asyncio, "sleep", AsyncMock())
    prediction = await evaluation.predict(client, case(), args)
    assert create.await_count == 2
    assert prediction["error"] is None and prediction["pred_action_type"] is None


@pytest.mark.asyncio
async def test_error_body_not_persisted(monkeypatch):
    args = evaluation.parse_args(["--model", "test", "--retries", "2"])
    create = AsyncMock(side_effect=ConnectionError("private credential"))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(evaluation.asyncio, "sleep", AsyncMock())
    prediction = await evaluation.predict(client, case(), args)
    assert create.await_count == 2
    assert prediction["error"]["type"] == "ConnectionError"
    assert "private credential" not in json.dumps(prediction)


def test_cannot_override_model_messages():
    with pytest.raises(SystemExit):
        evaluation.parse_args(["--model", "test", "--extra-body-json", '{"messages": []}'])


@pytest.mark.asyncio
async def test_sdk_roundtrip_resume_and_config_guard(monkeypatch, tmp_path):
    import httpx
    import openai

    sent = []

    async def handler(request):
        body = json.loads(request.content)
        sent.append(body)
        return httpx.Response(
            200,
            json={
                "id": "mock",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "critical"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    original_client = openai.AsyncOpenAI

    def make_client(**kwargs):
        client = original_client(
            **kwargs,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False),
        )
        # Avoid executor wakeups for SDK platform detection in restricted test environments.
        from openai._base_client import get_platform

        client._platform = get_platform()
        return client

    monkeypatch.setattr(openai, "AsyncOpenAI", make_client)
    args = evaluation.parse_args(
        [
            "--model",
            "test",
            "--output-dir",
            str(tmp_path),
            "--resume",
            "--timeout",
            "2",
            "--retries",
            "1",
        ]
    )
    cases = [case(), case("noisy", "two")]
    await evaluation.evaluate(cases, args)
    assert len(sent) == 2
    assert json.loads((tmp_path / "metrics.json").read_text())["overall"]["accuracy"] == 0.5
    await evaluation.evaluate(cases, args)
    assert len(sent) == 2
    # Resume reattempts API errors, not successful model answers.
    with (tmp_path / "predictions.jsonl").open("a") as handle:
        handle.write(json.dumps(result(cases[1], "", {"type": "ConnectionError"})) + "\n")
    await evaluation.evaluate(cases, args)
    assert len(sent) == 3
    args.model = "different-model"
    with pytest.raises(ValueError, match="configuration differs"):
        await evaluation.evaluate(cases, args)

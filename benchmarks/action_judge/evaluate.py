#!/usr/bin/env python3
"""Evaluate the released Action Judge Benchmark through an OpenAI-compatible API.

The label parser and three-class scoring follow the original paper evaluator.
Only the stored system/user messages are sent to the model; tools are never executed.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
LABELS = ("critical", "exploratory", "noisy")
DOMAINS = ("search", "terminal", "swe")
ACTION_TYPE_RE = re.compile(r"<action_type>\s*([^<]+?)\s*</action_type>", re.I | re.S)
JSON_ACTION_TYPE_RE = re.compile(
    r"[\"']?action_type[\"']?\s*[:=]\s*[\"']?(critical|exploratory|noisy)[\"']?", re.I
)


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def load_cases(data_dir: Path, domain: str = "all", limit: int | None = None) -> list[dict]:
    """Read an external JSONL file or checksummed, per-domain compressed splits."""
    cases = []
    ids = set()
    prompts = set()
    if data_dir.is_file():
        splits = [(None, data_dir, None)]
    else:
        manifest_path = data_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "Benchmark data is distributed separately. Pass --data-file with a JSONL "
                "file, or --data-dir with the checksummed domain splits."
            )
        manifest = json.loads(manifest_path.read_text())
        splits = [
            (name, data_dir / f"{name}.jsonl.gz", manifest["files"][f"{name}.jsonl.gz"])
            for name in (DOMAINS if domain == "all" else (domain,))
        ]
    for name, path, info in splits:
        if info and hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Data checksum mismatch: {path.name}")
        count = 0
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                messages = row["messages"]
                if (
                    row["domain"] not in DOMAINS
                    or (name is not None and row["domain"] != name)
                    or row["gold_action_type"] not in LABELS
                    or not isinstance(row["sample_id"], str)
                    or not row["sample_id"]
                    or not isinstance(row["balanced_core"], bool)
                    or len(messages) != 2
                    or [m["role"] for m in messages] != ["system", "user"]
                    or any(set(m) != {"role", "content"} for m in messages)
                    or any(not isinstance(m["content"], str) or not m["content"] for m in messages)
                ):
                    raise ValueError(f"Invalid evaluation row: {row.get('sample_id')}")
                prompt_hash = digest(messages)
                if row["sample_id"] in ids or prompt_hash in prompts:
                    raise ValueError("Duplicate sample ID or visible prompt")
                ids.add(row["sample_id"])
                prompts.add(prompt_hash)
                if domain == "all" or row["domain"] == domain:
                    cases.append({**row, "prompt_sha256": prompt_hash})
                count += 1
        if info and count != info["cases"]:
            raise ValueError(f"Case count mismatch: {path.name}")
    return cases[:limit] if limit is not None else cases


def parse_output(content: str) -> dict:
    match = ACTION_TYPE_RE.search(content)
    raw = match.group(1).strip().lower() if match else ""
    label = raw if raw in LABELS else None
    if label is None:
        json_match = JSON_ACTION_TYPE_RE.search(content)
        if json_match:
            label = json_match.group(1).lower()
    if label is None:
        found = {label for label in LABELS if re.search(rf"\b{label}\b", content.lower())}
        if len(found) == 1:
            label = found.pop()
    return {"pred_action_type": label, "strict_format_ok": raw in LABELS}


def score(cases: list[dict], predictions: dict[str, dict]) -> dict:
    matrix = {label: dict.fromkeys((*LABELS, "invalid"), 0) for label in LABELS}
    api_errors = invalid = completed = strict = 0
    for case in cases:
        row = predictions.get(case["sample_id"])
        pred = None
        if row is not None:
            completed += 1
            api_errors += bool(row.get("error"))
            parsed = parse_output(row.get("content") or "")
            pred = parsed["pred_action_type"] if not row.get("error") else None
            strict += not row.get("error") and parsed["strict_format_ok"]
            invalid += not row.get("error") and pred is None
        matrix[case["gold_action_type"]][pred if pred in LABELS else "invalid"] += 1
    per_class = {}
    for label in LABELS:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[gold][label] for gold in LABELS)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    correct = sum(matrix[label][label] for label in LABELS)
    return {
        "total": len(cases),
        "completed": completed,
        "pending": len(cases) - completed,
        "api_errors": api_errors,
        "invalid_outputs": invalid,
        "correct": correct,
        "accuracy": correct / len(cases) if cases else 0.0,
        "macro_f1": sum(m["f1"] for m in per_class.values()) / 3,
        "strict_format_rate": strict / len(cases) if cases else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def summarize(cases: list[dict], predictions: dict[str, dict]) -> dict:
    groups = {name: [c for c in cases if c["domain"] == name] for name in DOMAINS}
    groups["overall"] = cases
    groups["balanced_core"] = [c for c in cases if c["balanced_core"]]
    return {name: score(rows, predictions) for name, rows in groups.items() if rows}


def report(metrics: dict) -> str:
    lines = [
        "# Action Judge Benchmark",
        "",
        "Accuracy and macro-F1 are percentages; API errors and invalid outputs count as incorrect.",
        "Pending cases also remain in the denominator. Overall pools cases across domains.",
        "",
        "| Split | Completed / total | Pending | API errors | Invalid outputs | "
        "Accuracy | Macro-F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, s in metrics.items():
        lines.append(
            f"| {name} | {s['completed']} / {s['total']} | {s['pending']} | "
            f"{s['api_errors']} | {s['invalid_outputs']} | "
            f"{s['accuracy'] * 100:.2f} | {s['macro_f1'] * 100:.2f} |"
        )
    return "\n".join(lines) + "\n"


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_predictions(path: Path, cases: list[dict], repair_tail: bool = False) -> dict[str, dict]:
    if not path.exists():
        return {}
    by_id = {c["sample_id"]: c for c in cases}
    predictions = {}
    with path.open("rb") as handle:
        while line := handle.readline():
            offset = handle.tell() - len(line)
            if not line.endswith(b"\n"):
                # Each completed response is flushed as one newline-terminated record.
                if repair_tail:
                    with path.open("r+b") as writable:
                        writable.truncate(offset)
                print("Ignoring an interrupted final prediction record.", file=sys.stderr)
                break
            row = json.loads(line)
            case = by_id.get(row.get("sample_id"))
            if case is None or row.get("prompt_sha256") != case["prompt_sha256"]:
                raise ValueError("Saved predictions do not match the selected data")
            predictions[case["sample_id"]] = row
    return predictions


@contextmanager
def output_lock(output: Path):
    lock = output / ".evaluation.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError(
            f"Output is locked: {lock}. Stop any other writer first. If a process was killed, "
            "remove this empty lock directory only after confirming it has stopped."
        ) from None
    try:
        yield
    finally:
        lock.rmdir()


def identity(cases: list[dict], args: argparse.Namespace) -> dict:
    return {
        "protocol": "action_judge_v1",
        "data_sha256": digest(
            [
                (c["sample_id"], c["prompt_sha256"], c["gold_action_type"], c["balanced_core"])
                for c in cases
            ]
        ),
        "model": args.model,
        "base_url": args.base_url.rstrip("/"),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "extra_body": args.extra_body,
    }


async def predict(client, case: dict, args: argparse.Namespace) -> dict:
    started = time.monotonic()
    error = None
    for attempt in range(1, args.retries + 1):
        try:
            response = await client.chat.completions.create(
                model=args.model,
                messages=case["messages"],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                **({"extra_body": args.extra_body} if args.extra_body else {}),
            )
            choice = response.choices[0]
            content = choice.message.content or ""
            return {
                "sample_id": case["sample_id"],
                "prompt_sha256": case["prompt_sha256"],
                "content": content,
                "reasoning_content": getattr(choice.message, "reasoning_content", None),
                **parse_output(content),
                "finish_reason": choice.finish_reason,
                "usage": response.usage.model_dump() if response.usage else {},
                "attempts": attempt,
                "error": None,
                "latency_seconds": round(time.monotonic() - started, 3),
            }
        except Exception as exc:
            # Do not persist exception bodies: gateways may echo credentials or request headers.
            status = getattr(exc, "status_code", None)
            error = {"type": type(exc).__name__, "http_status": status}
            if status in (400, 401, 403, 404, 413, 422) or attempt == args.retries:
                break
            await asyncio.sleep(min(2 ** (attempt - 1), 30) + random.random())
    return {
        "sample_id": case["sample_id"],
        "prompt_sha256": case["prompt_sha256"],
        "content": "",
        "pred_action_type": None,
        "strict_format_ok": False,
        "finish_reason": "error",
        "attempts": attempt,
        "error": error,
        "latency_seconds": round(time.monotonic() - started, 3),
    }


async def evaluate(cases: list[dict], args: argparse.Namespace) -> None:
    # Validation and offline scoring require only Python's standard library.
    from openai import AsyncOpenAI

    config_path = args.output_dir / "run_config.json"
    prediction_path = args.output_dir / "predictions.jsonl"
    run_identity = identity(cases, args)
    if prediction_path.exists() and not args.resume:
        raise ValueError("Predictions exist; use --resume or a fresh output directory")
    if args.resume and (config_path.exists() or prediction_path.exists()):
        if not config_path.exists() or json.loads(config_path.read_text()) != run_identity:
            raise ValueError("Resume configuration differs; use a new output directory")
    write_json(config_path, run_identity)
    predictions = read_predictions(prediction_path, cases, repair_tail=True)
    pending = [
        c
        for c in cases
        if c["sample_id"] not in predictions or predictions[c["sample_id"]].get("error")
    ]
    print(
        f"Cases={len(cases)}; retained={len(cases) - len(pending)}; pending={len(pending)}",
        flush=True,
    )
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    progress = tqdm(total=len(pending), desc="Action Judge", unit="case") if tqdm else None
    queue = iter(pending)
    done = 0
    try:
        async with AsyncOpenAI(
            base_url=args.base_url,
            api_key=args.api_key,
            timeout=args.timeout,
            max_retries=0,
        ) as client:
            with prediction_path.open("a", encoding="utf-8") as output:

                async def worker():
                    nonlocal done
                    for case in queue:
                        result = await predict(client, case, args)
                        output.write(json.dumps(result, ensure_ascii=False) + "\n")
                        output.flush()
                        predictions[case["sample_id"]] = result
                        done += 1
                        if progress:
                            progress.update(1)
                        elif done % 10 == 0 or done == len(pending):
                            print(f"Completed {done}/{len(pending)} pending cases", flush=True)

                async with asyncio.TaskGroup() as tasks:
                    for _ in range(min(args.concurrency, len(pending))):
                        tasks.create_task(worker())
    finally:
        if progress:
            progress.close()
        metrics = summarize(cases, predictions)
        write_json(args.output_dir / "metrics.json", metrics)
        (args.output_dir / "report.md").write_text(report(metrics), encoding="utf-8")
    print(report(metrics))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_mutually_exclusive_group()
    data.add_argument("--data-dir", type=Path, default=DATA_DIR)
    data.add_argument("--data-file", type=Path, help="External benchmark JSONL or JSONL.gz file")
    parser.add_argument("--domain", choices=("all", *DOMAINS), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("results/action_judge"))
    parser.add_argument("--model", default=os.environ.get("EVAL_MODEL"))
    parser.add_argument(
        "--base-url", default=os.environ.get("EVAL_BASE_URL", "http://localhost:8000/v1")
    )
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--extra-body-json", default="{}")
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--score-only", action="store_true")
    args = parser.parse_args(argv)
    for key in ("concurrency", "max_tokens", "timeout", "retries"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        args.extra_body = json.loads(args.extra_body_json)
        if not isinstance(args.extra_body, dict):
            raise ValueError("expected an object")
        reserved = {
            "model",
            "messages",
            "tools",
            "tool_choice",
            "temperature",
            "max_tokens",
            "stream",
            "n",
        }
        if reserved & args.extra_body.keys():
            raise ValueError(
                "extra-body cannot override model, messages, tools, or sampling fields"
            )
    except ValueError as exc:
        parser.error(f"Invalid --extra-body-json: {exc}")
    args.api_key = os.environ.get("EVAL_API_KEY", "EMPTY")
    if not (args.validate_only or args.score_only) and not args.model:
        parser.error("Set EVAL_MODEL or pass --model")
    return args


def main(argv=None):
    args = parse_args(argv)
    cases = load_cases(args.data_file or args.data_dir, args.domain, args.limit)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "cases": len(cases),
                    "domains": dict(Counter(c["domain"] for c in cases)),
                    "labels": dict(Counter(c["gold_action_type"] for c in cases)),
                    "balanced_core": sum(c["balanced_core"] for c in cases),
                },
                indent=2,
            )
        )
        return
    if args.score_only:
        path = args.output_dir / "predictions.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        print(report(summarize(cases, read_predictions(path, cases))))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_lock(args.output_dir):
        asyncio.run(evaluate(cases, args))


if __name__ == "__main__":
    main()

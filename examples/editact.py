"""Run one user-supplied task with EditAct, without benchmark data."""

import argparse
import asyncio
import json
import logging
import shlex
from dataclasses import asdict
from pathlib import Path

from aweagent.core.agent.context import AgentContext
from aweagent.core.config.loader import load_config
from aweagent.core.llm.client import LLMClient
from aweagent.core.runtime.docker import DockerRuntime
from aweagent.core.task.pipeline import build_agent_factory


async def run(args):
    config = load_config(args.config)
    agent = build_agent_factory(config)()
    prompt = args.task if args.task is not None else Path(args.task_file).read_text()
    if any("${" in str(v) for v in (config.llm.model, config.llm.base_url, config.llm.api_key)):
        raise ValueError("Set ACTOR_MODEL, ACTOR_BASE_URL and ACTOR_API_KEY")
    async with LLMClient(config.llm) as llm:
        context = AgentContext(
            llm=llm,
            tools=agent.get_tools(),
            max_steps=args.max_steps,
            max_context_length=config.agent.max_context_length,
            task_info={
                "instruction": prompt,
                "workdir": config.runtime.workdir,
                "dataset_id": config.task.dataset_id,
                "task_type": config.task.task_type,
                "skip_patch_extraction": True,
            },
        )
        if config.agent.type == "editact_for_search":
            result = await asyncio.wait_for(agent.create_loop(context).run(prompt), args.timeout)
        else:
            runtime = DockerRuntime(config.runtime)
            async with runtime.session() as session:
                context.session = session
                workdir = shlex.quote(config.runtime.workdir)
                await session.execute(f"mkdir -p {workdir}", cwd="/")
                result = await asyncio.wait_for(
                    agent.create_loop(context).run(prompt), args.timeout
                )
                # Preserve generated files before the disposable container is removed.
                archive = await session.execute(
                    f"tar -czf /tmp/editact-workspace.tar.gz -C {workdir} ."
                )
                if not archive.success:
                    raise RuntimeError("Could not archive the sandbox workspace")
                output = Path(args.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.with_suffix(".workspace.tar.gz").write_bytes(
                    await session.download_file("/tmp/editact-workspace.tar.gz")
                )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False, default=str) + "\n")
    events = result.trajectory.metadata.get("editact", [])
    print(
        json.dumps(
            {
                "finish_reason": result.finish_reason,
                "steps": len(result.trajectory.steps),
                "revisions": sum(e["intervention"] == "rewrite" for e in events),
                "fallbacks": sum(e["intervention"] == "fallback" for e in events),
                "output": str(output),
            },
            indent=2,
        )
    )
    if result.finish_reason == "error":
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    task = parser.add_mutually_exclusive_group(required=True)
    task.add_argument("--task")
    task.add_argument("--task-file")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output", default="results/editact/example.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    # Avoid exposing endpoint addresses in ordinary example logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

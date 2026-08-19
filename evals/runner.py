"""Eval harness.

Runs a YAML suite against the real loop and writes a report. Each task gets a
fresh workspace seeded with whatever the suite declares, so tasks cannot leak
into each other and a failure is reproducible on its own.

    python -m evals.runner --suite evals/tasks/core.yaml
    python -m evals.runner --suite evals/tasks/core.yaml --scripted   # no API key

The headline number is the completion rate: the share of tasks where every check
passed. Anything softer than that is a number you can talk yourself into.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from agent.config import get_settings
from agent.loop import Agent
from agent.tools.sandbox import Workspace
from evals.graders import CheckResult, run_check


@dataclass
class TaskOutcome:
    id: str
    prompt: str
    passed: bool
    checks: list[CheckResult]
    iterations: int
    duration_s: float
    usd: float
    status: str
    summary: str
    workspace: str
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt.strip(),
            "passed": self.passed,
            "checks": [c.as_dict() for c in self.checks],
            "iterations": self.iterations,
            "duration_s": self.duration_s,
            "usd": self.usd,
            "status": self.status,
            "summary": self.summary,
            "workspace": self.workspace,
            "error": self.error,
        }


@dataclass
class Report:
    suite: str
    started_at: str
    model: str
    outcomes: list[TaskOutcome] = field(default_factory=list)

    @property
    def completion_rate(self) -> float:
        if not self.outcomes:
            return 0.0
        return round(sum(1 for o in self.outcomes if o.passed) / len(self.outcomes), 4)

    @property
    def grounding_rate(self) -> float:
        """Share of runs whose summary only referenced files that exist."""
        judged = [
            c for o in self.outcomes for c in o.checks if c.name == "grounding"
        ]
        if not judged:
            return 0.0
        return round(sum(1 for c in judged if c.passed) / len(judged), 4)

    def as_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "started_at": self.started_at,
            "model": self.model,
            "tasks": len(self.outcomes),
            "completion_rate": self.completion_rate,
            "grounding_rate": self.grounding_rate,
            "avg_iterations": round(
                statistics.fmean([o.iterations for o in self.outcomes]), 2
            )
            if self.outcomes
            else 0.0,
            "total_usd": round(sum(o.usd for o in self.outcomes), 4),
            "outcomes": [o.as_dict() for o in self.outcomes],
        }

    def as_markdown(self) -> str:
        lines = [
            f"# eval report - {self.suite}",
            "",
            f"- model: `{self.model}`",
            f"- run at: {self.started_at}",
            f"- completion rate: **{self.completion_rate:.0%}** "
            f"({sum(1 for o in self.outcomes if o.passed)}/{len(self.outcomes)})",
            f"- grounding rate: **{self.grounding_rate:.0%}**",
            f"- spend: ${sum(o.usd for o in self.outcomes):.4f}",
            "",
            "| task | result | steps | time | failing checks |",
            "| --- | --- | --- | --- | --- |",
        ]
        for o in self.outcomes:
            failed = ", ".join(c.name for c in o.checks if not c.passed) or "-"
            lines.append(
                f"| `{o.id}` | {'pass' if o.passed else 'fail'} | {o.iterations} | "
                f"{o.duration_s:.1f}s | {failed} |"
            )
        return "\n".join(lines) + "\n"


def load_suite(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "tasks" not in data:
        raise ValueError(f"{path} does not look like a suite (no `tasks:` key)")
    return data


def seed_workspace(workspace: Workspace, files: dict[str, str] | None) -> None:
    for relative, content in (files or {}).items():
        workspace.write(relative, content)


def run_suite(
    suite_path: Path,
    *,
    scripted: bool = False,
    limit: int | None = None,
    only: str | None = None,
    out_dir: Path = Path("eval_reports"),
) -> Report:
    suite = load_suite(suite_path)
    settings = get_settings()

    def make_llm():
        # A fresh client per task: the scripted one is stateful, and sharing it
        # would let task 1 consume the script that task 2 needs.
        if not scripted:
            return None
        from agent.llm.scripted import ScriptedClient

        return ScriptedClient.demo()

    from agent.memory.embeddings import HashingEmbedder
    from agent.memory.store import InMemoryStore

    report = Report(
        suite=suite.get("suite", suite_path.stem),
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        model="scripted" if scripted else settings.model,
    )

    tasks = [t for t in suite["tasks"] if not only or t["id"] == only][:limit]
    root = Path(out_dir) / f"workspaces-{int(time.time())}"

    for task in tasks:
        task_id = task["id"]
        workspace = Workspace(root / task_id)
        seed_workspace(workspace, task.get("files"))

        # Memory is per-task so one run cannot coach the next; that would make
        # the suite look better than the agent is.
        agent = Agent(
            settings=settings,
            llm=make_llm(),
            memory=InMemoryStore(HashingEmbedder(256)),
            retriever=None,
            trace=True,
        )

        print(f"→ {task_id}", flush=True)
        result = agent.run(task["prompt"], workspace=workspace.root)
        checks = [run_check(spec, workspace, result) for spec in task.get("checks", [])]
        outcome = TaskOutcome(
            id=task_id,
            prompt=task["prompt"],
            passed=all(c.passed for c in checks) and bool(checks),
            checks=checks,
            iterations=result.iterations,
            duration_s=result.duration_s,
            usd=result.usage.usd,
            status=result.status.value,
            summary=result.summary,
            workspace=str(workspace.root),
            error=result.error,
        )
        report.outcomes.append(outcome)
        _print_outcome(outcome)

    _write(report, out_dir)
    return report


def _print_outcome(outcome: TaskOutcome) -> None:
    mark = "pass" if outcome.passed else "FAIL"
    print(f"  {mark}  {outcome.iterations} step(s), {outcome.duration_s:.1f}s")
    for check in outcome.checks:
        if not check.passed:
            print(f"        - {check.name}: {check.detail}")


def _write(report: Report, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"{report.suite}-{stamp}.json"
    md_path = out_dir / f"{report.suite}-{stamp}.md"
    json_path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    md_path.write_text(report.as_markdown(), encoding="utf-8")
    print(f"\nreport: {md_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an eval suite against the agent.")
    parser.add_argument("--suite", default="evals/tasks/core.yaml", type=Path)
    parser.add_argument("--scripted", action="store_true", help="Use the offline client")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", default=None, help="Run a single task by id")
    parser.add_argument("--out", type=Path, default=Path("eval_reports"))
    args = parser.parse_args(argv)

    report = run_suite(
        args.suite, scripted=args.scripted, limit=args.limit, only=args.only, out_dir=args.out
    )
    print(
        f"\ncompletion {report.completion_rate:.0%} · "
        f"grounding {report.grounding_rate:.0%} · "
        f"${sum(o.usd for o in report.outcomes):.4f}"
    )
    return 0 if report.completion_rate == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

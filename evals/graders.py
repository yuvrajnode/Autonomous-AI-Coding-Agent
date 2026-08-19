"""Graders.

A grader looks at the finished workspace and the run result and answers one
question with a boolean and a reason. Keeping them this small is what makes the
report readable: a task does not "score 0.6", it fails a named check.

The most useful grader here is `grounding`. It pulls every file-path-shaped
token out of the agent's own summary and checks that the file exists. An agent
that reports editing `src/utils/retry.py` when no such file was ever created is
hallucinating, and unlike a vague quality score, that is cheap to detect and
impossible to argue with.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

from agent.tools.sandbox import Workspace
from agent.types import RunResult, RunStatus

PATH_TOKEN = re.compile(r"\b[\w./-]+\.(?:py|md|txt|json|yaml|yml|toml|cfg|sql|js|ts)\b")


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def run_check(spec: dict[str, Any], workspace: Workspace, result: RunResult) -> CheckResult:
    kind = spec.get("type", "")
    grader = GRADERS.get(kind)
    if grader is None:
        return CheckResult(kind or "unknown", False, f"no grader named {kind!r}")
    try:
        return grader(spec, workspace, result)
    except Exception as exc:  # noqa: BLE001 - a broken grader fails its check, not the suite
        return CheckResult(kind, False, f"grader raised {type(exc).__name__}: {exc}")


# -- individual graders -----------------------------------------------------


def _file_exists(spec: dict[str, Any], ws: Workspace, _: RunResult) -> CheckResult:
    path = spec["path"]
    ok = ws.exists(path)
    return CheckResult(f"file_exists({path})", ok, "" if ok else "not found in the workspace")


def _file_contains(spec: dict[str, Any], ws: Workspace, _: RunResult) -> CheckResult:
    path, pattern = spec["path"], spec["pattern"]
    name = f"file_contains({path}, /{pattern}/)"
    if not ws.exists(path):
        return CheckResult(name, False, "file does not exist")
    body = ws.read(path)
    ok = re.search(pattern, body, re.MULTILINE) is not None
    return CheckResult(name, ok, "" if ok else "pattern not found")


def _python(spec: dict[str, Any], ws: Workspace, _: RunResult) -> CheckResult:
    """Run an assertion script against whatever the agent produced."""
    code = spec["code"]
    proc = subprocess.run(  # noqa: S603 - the code comes from the suite file, not the model
        [sys.executable, "-c", code],
        cwd=ws.root,
        capture_output=True,
        text=True,
        timeout=spec.get("timeout", 60),
        check=False,
    )
    ok = proc.returncode == 0
    detail = "" if ok else (proc.stderr.strip().splitlines() or ["failed"])[-1]
    return CheckResult(f"python({spec.get('label', 'assertion')})", ok, detail[:300])


def _tests_pass(spec: dict[str, Any], ws: Workspace, _: RunResult) -> CheckResult:
    argv = [sys.executable, "-m", "pytest", "-q", "--no-header"]
    if spec.get("target"):
        argv.append(spec["target"])
    proc = subprocess.run(  # noqa: S603
        argv, cwd=ws.root, capture_output=True, text=True,
        timeout=spec.get("timeout", 180), check=False,
    )
    ok = proc.returncode == 0
    tail = (proc.stdout.strip().splitlines() or [""])[-1]
    return CheckResult("tests_pass", ok, tail[:200])


def _succeeded(spec: dict[str, Any], _: Workspace, result: RunResult) -> CheckResult:
    ok = result.status is RunStatus.SUCCEEDED
    return CheckResult("run_succeeded", ok, result.error or result.status.value)


def _within_iterations(spec: dict[str, Any], _: Workspace, result: RunResult) -> CheckResult:
    limit = int(spec.get("max", 12))
    ok = result.iterations <= limit
    return CheckResult(
        f"within_iterations({limit})", ok, f"took {result.iterations}"
    )


def _grounding(spec: dict[str, Any], ws: Workspace, result: RunResult) -> CheckResult:
    """Every file the summary names must actually exist."""
    ignore = set(spec.get("ignore", []))
    mentioned = {
        token for token in PATH_TOKEN.findall(result.summary or "") if token not in ignore
    }
    missing = sorted(t for t in mentioned if not ws.exists(t.lstrip("./")))
    ok = not missing
    return CheckResult(
        "grounding",
        ok,
        "" if ok else f"summary references files that do not exist: {', '.join(missing)}",
    )


def _no_placeholder(spec: dict[str, Any], ws: Workspace, _: RunResult) -> CheckResult:
    """Catch the classic 'implementation left as an exercise' non-delivery."""
    markers = spec.get("markers", ["TODO", "FIXME", "NotImplementedError", "pass  # implement"])
    hits: list[str] = []
    for path in ws.walk():
        if not path.endswith((".py", ".md", ".js", ".ts")):
            continue
        body = ws.read(path)
        hits.extend(f"{path}: {m}" for m in markers if m in body)
    ok = not hits
    return CheckResult("no_placeholders", ok, "; ".join(hits[:4]))


GRADERS = {
    "file_exists": _file_exists,
    "file_contains": _file_contains,
    "python": _python,
    "tests_pass": _tests_pass,
    "succeeded": _succeeded,
    "within_iterations": _within_iterations,
    "grounding": _grounding,
    "no_placeholders": _no_placeholder,
}

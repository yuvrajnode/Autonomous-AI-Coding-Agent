"""Command execution.

Commands never go through a shell. They are split with shlex and executed
directly, which removes the entire class of `; rm -rf ~` problems, and the
executable itself has to be on the allowlist. Anything the agent genuinely
needs that is not here should be added deliberately, not by widening the rule.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys

from pydantic import BaseModel, Field

from agent.errors import ToolError
from agent.tools.registry import ToolContext, registry

ALLOWED_EXECUTABLES = {
    "python", "python3", "pytest", "ruff", "mypy", "black",
    "node", "npm", "npx", "pnpm", "yarn",
    "git", "make",
    "ls", "cat", "head", "tail", "wc", "sort", "uniq", "diff", "grep", "find", "echo",
}

# Sub-commands that mutate history, publish, or reach the network.
BLOCKED_GIT = {"push", "remote", "clone", "fetch", "pull", "config", "clean"}

MAX_OUTPUT = 12_000


class RunCommandArgs(BaseModel):
    command: str = Field(description="Command line to run, e.g. 'pytest -q tests/test_x.py'")
    timeout: int | None = Field(default=None, description="Seconds before the command is killed")


@registry.register(RunCommandArgs, mutating=True)
def run_command(ctx: ToolContext, args: RunCommandArgs) -> str:
    """Run a command inside the workspace and return its exit code, stdout and stderr.

    No shell is involved, so pipes, redirects and `&&` do not work - run one
    command per call. Only a fixed set of executables is permitted.
    """
    argv = _parse(args.command)
    timeout = args.timeout or ctx.settings.shell_timeout
    return _execute(argv, ctx, timeout, label=args.command)


class RunPythonArgs(BaseModel):
    code: str = Field(description="Python source to execute in the workspace directory")
    timeout: int | None = Field(default=None, description="Seconds before execution is killed")


@registry.register(RunPythonArgs, mutating=True)
def run_python(ctx: ToolContext, args: RunPythonArgs) -> str:
    """Execute a Python snippet with the workspace on sys.path.

    Good for checking that the module you just wrote imports and behaves. The
    snippet runs in a fresh process, so nothing carries over between calls.
    """
    timeout = args.timeout or ctx.settings.shell_timeout
    return _execute([sys.executable, "-c", args.code], ctx, timeout, label="python -c")


class RunTestsArgs(BaseModel):
    target: str = Field(default="", description="Optional test path or node id to narrow the run")
    timeout: int | None = Field(default=None, description="Seconds before the run is killed")


@registry.register(RunTestsArgs, mutating=True)
def run_tests(ctx: ToolContext, args: RunTestsArgs) -> str:
    """Run pytest in the workspace and return the summary.

    A task is not finished until this comes back green. If there are no tests,
    write one first.
    """
    argv = [sys.executable, "-m", "pytest", "-q", "--no-header"]
    if args.target:
        argv.append(args.target)
    timeout = args.timeout or max(ctx.settings.shell_timeout, 120)
    return _execute(argv, ctx, timeout, label="pytest")


# --------------------------------------------------------------------------


def _parse(command: str) -> list[str]:
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        raise ToolError(f"could not parse command: {exc}") from exc
    if not argv:
        raise ToolError("empty command")

    exe = os.path.basename(argv[0])
    if exe not in ALLOWED_EXECUTABLES:
        raise ToolError(
            f"{exe!r} is not on the allowlist. Permitted: {', '.join(sorted(ALLOWED_EXECUTABLES))}"
        )
    if exe == "git" and len(argv) > 1 and argv[1] in BLOCKED_GIT:
        raise ToolError(f"`git {argv[1]}` is blocked inside the sandbox")
    return argv


def _execute(argv: list[str], ctx: ToolContext, timeout: int, label: str) -> str:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ctx.workspace.root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("VOYAGE_API_KEY", None)

    try:
        proc = subprocess.run(  # noqa: S603 - argv is allowlisted and never shell-parsed
            argv,
            cwd=ctx.workspace.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ToolError(f"executable not found: {argv[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"`{label}` exceeded the {timeout}s timeout and was killed") from exc

    return _format(label, proc.returncode, proc.stdout, proc.stderr)


def _format(label: str, code: int, stdout: str, stderr: str) -> str:
    parts = [f"$ {label}", f"exit code: {code}"]
    if stdout.strip():
        parts.append("--- stdout ---\n" + _clip(stdout))
    if stderr.strip():
        parts.append("--- stderr ---\n" + _clip(stderr))
    if not stdout.strip() and not stderr.strip():
        parts.append("(no output)")
    return "\n".join(parts)


def _clip(text: str) -> str:
    text = text.rstrip()
    if len(text) <= MAX_OUTPUT:
        return text
    keep = MAX_OUTPUT // 2
    return f"{text[:keep]}\n... [{len(text) - MAX_OUTPUT} chars elided] ...\n{text[-keep:]}"

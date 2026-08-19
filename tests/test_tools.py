from __future__ import annotations

import pytest
from pydantic import BaseModel

from agent.errors import ToolError
from agent.tools import registry
from agent.tools.registry import ToolRegistry
from agent.types import ToolCall


def call(name: str, **arguments) -> ToolCall:
    return ToolCall(name=name, arguments=arguments)


# -- registry ---------------------------------------------------------------


def test_schema_is_generated_from_the_args_model():
    schema = next(s for s in registry.schemas() if s["name"] == "read_file")
    assert schema["input_schema"]["properties"].keys() >= {"path", "start_line", "end_line"}
    assert "path" in schema["input_schema"]["required"]
    assert schema["description"]


def test_unknown_tool_reports_the_alternatives(ctx):
    result = registry.invoke(call("teleport"), ctx)
    assert result.ok is False
    assert "read_file" in result.error


def test_bad_arguments_become_a_readable_error(ctx):
    result = registry.invoke(call("read_file"), ctx)
    assert result.ok is False
    assert "path" in result.error


def test_tool_exceptions_do_not_escape(ctx):
    local = ToolRegistry()

    class Args(BaseModel):
        pass

    @local.register(Args)
    def boom(_ctx, _args) -> str:
        """Always explodes."""
        raise RuntimeError("kaboom")

    result = local.invoke(call("boom"), ctx)
    assert result.ok is False
    assert "kaboom" in result.error


def test_tools_must_have_a_docstring():
    local = ToolRegistry()

    class Args(BaseModel):
        pass

    with pytest.raises(ValueError, match="docstring"):
        local.register(Args)(lambda _ctx, _args: "")


# -- filesystem -------------------------------------------------------------


def test_write_then_read_round_trips(ctx):
    registry.invoke(call("write_file", path="a.py", content="x = 1\ny = 2\n"), ctx)
    result = registry.invoke(call("read_file", path="a.py"), ctx)
    assert "1| x = 1" in result.content
    assert "lines 1-2 of 2" in result.content


def test_edit_requires_a_unique_match(ctx):
    ctx.workspace.write("a.py", "value = 1\nvalue = 1\n")
    result = registry.invoke(call("edit_file", path="a.py", find="value = 1", replace="v = 2"), ctx)
    assert result.ok is False
    assert "matches 2 times" in result.error
    assert ctx.workspace.read("a.py") == "value = 1\nvalue = 1\n"


def test_edit_rejects_a_missing_snippet(ctx):
    ctx.workspace.write("a.py", "value = 1\n")
    result = registry.invoke(call("edit_file", path="a.py", find="nope", replace="x"), ctx)
    assert result.ok is False


def test_edit_applies_a_unique_match(ctx):
    ctx.workspace.write("a.py", "alpha = 1\nbeta = 2\n")
    result = registry.invoke(
        call("edit_file", path="a.py", find="beta = 2", replace="beta = 3"), ctx
    )
    assert result.ok
    assert ctx.workspace.read("a.py") == "alpha = 1\nbeta = 3\n"


# -- search -----------------------------------------------------------------


def test_grep_reports_path_and_line(ctx):
    ctx.workspace.write("src/app.py", "import os\n\ndef handler():\n    return 1\n")
    result = registry.invoke(call("grep", pattern=r"def \w+"), ctx)
    assert "src/app.py:3: def handler():" in result.content


def test_grep_rejects_a_bad_regex(ctx):
    result = registry.invoke(call("grep", pattern="([unclosed"), ctx)
    assert result.ok is False
    assert "bad regex" in result.error


def test_find_files_matches_by_basename(ctx):
    ctx.workspace.write("src/deep/thing.py", "1")
    result = registry.invoke(call("find_files", pattern="*.py"), ctx)
    assert "src/deep/thing.py" in result.content


# -- shell ------------------------------------------------------------------


def test_shell_allowlist_blocks_unknown_executables(ctx):
    result = registry.invoke(call("run_command", command="curl https://example.com"), ctx)
    assert result.ok is False
    assert "allowlist" in result.error


def test_shell_does_not_expand_metacharacters(ctx):
    result = registry.invoke(call("run_command", command="echo hi && rm -rf /"), ctx)
    assert result.ok
    assert "&&" in result.content  # passed through as a literal argument


def test_shell_blocks_dangerous_git_subcommands(ctx):
    result = registry.invoke(call("run_command", command="git push origin main"), ctx)
    assert result.ok is False
    assert "blocked" in result.error


def test_run_python_executes_against_the_workspace(ctx):
    ctx.workspace.write("solution.py", "def double(n):\n    return n * 2\n")
    result = registry.invoke(
        call("run_python", code="from solution import double; print(double(21))"), ctx
    )
    assert result.ok
    assert "42" in result.content
    assert "exit code: 0" in result.content


# -- knowledge --------------------------------------------------------------


def test_submit_rejects_artifacts_that_do_not_exist(ctx):
    result = registry.invoke(
        call("submit_result", summary="done", artifacts=["ghost.py"]), ctx
    )
    assert result.ok is False
    assert "do not exist" in result.error


def test_submit_records_the_result(ctx):
    ctx.workspace.write("real.py", "1")
    result = registry.invoke(
        call("submit_result", summary="wrote real.py", artifacts=["real.py"]), ctx
    )
    assert result.ok
    assert ctx.scratch["submitted"]["artifacts"] == ["real.py"]


def test_remember_rejects_thin_memories(ctx):
    result = registry.invoke(call("remember", text="ok"), ctx)
    assert result.ok is False


def test_recall_round_trips_through_memory(ctx):
    registry.invoke(
        call("remember", text="the integration suite needs POSTGRES_URL set"), ctx
    )
    result = registry.invoke(call("recall", query="postgres url integration"), ctx)
    assert "POSTGRES_URL" in result.content


def test_knowledge_tools_fail_loudly_without_a_backend(ctx):
    ctx.memory = None
    result = registry.invoke(call("recall", query="anything"), ctx)
    assert result.ok is False
    with pytest.raises(ToolError):
        raise ToolError(result.error)

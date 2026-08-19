# Evals

The point of the harness is to make "the agent got better" a claim someone can
check. Everything here is designed so a result is either reproducible or clearly
labelled as not.

## Running

```bash
python -m evals.runner --suite evals/tasks/core.yaml            # live model
python -m evals.runner --suite evals/tasks/core.yaml --only fizzbuzz
python -m evals.runner --suite evals/tasks/core.yaml --scripted # harness only
```

Each task gets a fresh workspace under `eval_reports/workspaces-<timestamp>/`,
seeded with whatever the suite declares, and its own empty memory store — a run
that could read what an earlier run remembered would make the suite look better
than the agent is.

Workspaces are kept after the run. When a task fails, that directory is the
fastest way to see what the agent actually produced.

`--scripted` swaps in a canned client. It exercises the graders, the report and
the workspace seeding; it says nothing about model quality. A green scripted run
means the harness works.

## Writing a task

```yaml
- id: add-a-test
  prompt: >
    slugify.py has no tests. Write test_slugify.py covering the normal case,
    an empty string, and a string that is nothing but punctuation.
  files:
    slugify.py: |
      import re
      ...
  checks:
    - type: succeeded
    - type: file_exists
      path: test_slugify.py
    - type: tests_pass
    - type: no_placeholders
    - type: grounding
```

Two rules that keep a suite honest:

1. **Every task needs a check the agent cannot satisfy by talking.** `succeeded`
   alone is not a task — the agent decides that one. Pair it with `tests_pass`,
   a `python` assertion, or a `file_contains`.
2. **Prompts state the goal, not the method.** If the prompt names the tool to
   use, you are testing your own instructions.

## Graders

| type | passes when |
| --- | --- |
| `succeeded` | the run ended in `succeeded` |
| `file_exists` | `path` exists in the workspace |
| `file_contains` | `pattern` (a regex) matches `path` |
| `python` | the given `code` runs to exit 0 in the workspace |
| `tests_pass` | `pytest -q` is green |
| `within_iterations` | the run took at most `max` act/observe turns |
| `no_placeholders` | no `TODO` / `NotImplementedError` left behind |
| `grounding` | every file path named in the summary exists |

A task passes only if **every** check passes. No partial credit: a task where the
tests still fail is a failed task, and averaging it to 0.8 only makes the number
comfortable.

### `grounding`, specifically

It regexes file-path-shaped tokens out of the agent's own summary and asserts
each one exists in the workspace. That catches the most common and most damaging
hallucination in a coding agent — confidently reporting an edit to a file that
was never touched — without needing a judge model, a rubric, or an opinion.

It is a floor, not a ceiling. It says nothing about whether the *content* of a
claim is true, only that the artefacts it names are real.

## The numbers

`completion_rate` is the headline: tasks where every check passed, over tasks
run. `grounding_rate` is reported separately because it moves for different
reasons — prompt changes, retrieval floor changes — and averaging it into the
headline hides that.

Reports are written as both JSON and markdown so the JSON can be diffed across
runs and the markdown can be pasted into a PR.

## Extending

Add a grader by writing a function of `(spec, workspace, result) -> CheckResult`
and registering it in the `GRADERS` dict in `evals/graders.py`. Keep it to one
question with a boolean answer; that is what makes the report readable.

Worth adding next, in rough order of value:

- a larger suite drawn from real repository issues rather than toy tasks
- a regression mode that fails CI when completion drops against a stored baseline
- a `judge` grader for tasks with no machine-checkable answer, reported apart
  from the mechanical checks so the two never get averaged together

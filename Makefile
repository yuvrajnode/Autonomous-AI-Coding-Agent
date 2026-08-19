VENV ?= .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: install db-up db-down migrate run serve eval test lint fmt clean

install:
	python3 -m venv $(VENV)
	$(PIP) install -q -U pip
	$(PIP) install -q -r requirements-dev.txt
	$(PIP) install -q -e .

db-up:
	docker compose up -d postgres

db-down:
	docker compose down

migrate:
	$(PY) -m agent.cli migrate

run:
	$(PY) -m agent.cli run "$(TASK)" --workspace $(WS)

serve:
	$(VENV)/bin/uvicorn server.app:app --reload --port 8000

eval:
	$(PY) -m evals.runner --suite evals/tasks/core.yaml

test:
	$(VENV)/bin/pytest -q

lint:
	$(VENV)/bin/ruff check agent server evals tests
	$(VENV)/bin/mypy agent

fmt:
	$(VENV)/bin/ruff check --fix agent server evals tests
	$(VENV)/bin/ruff format agent server evals tests

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache traces eval_reports
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

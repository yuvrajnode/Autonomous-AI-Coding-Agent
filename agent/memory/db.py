"""Postgres plumbing.

Vectors are sent as literals cast with `::vector` rather than through an
adapter. It keeps the dependency list short and, more usefully, means the SQL
in the logs is the SQL that ran.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from agent.errors import StoreError

log = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def vector_literal(vec: Sequence[float]) -> str:
    return "[" + ",".join(f"{v:.7g}" for v in vec) + "]"


@contextmanager
def connect(url: str) -> Iterator[Any]:
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise StoreError("psycopg is not installed; run `pip install -r requirements.txt`") from exc

    try:
        conn = psycopg.connect(url, autocommit=True, connect_timeout=5)
    except Exception as exc:  # noqa: BLE001 - psycopg raises a wide family here
        raise StoreError(f"cannot reach postgres at {_redact(url)}: {exc}") from exc
    try:
        yield conn
    finally:
        conn.close()


def is_available(url: str) -> bool:
    try:
        with connect(url) as conn:
            conn.execute("SELECT 1")
        return True
    except StoreError:
        return False


def migrate(url: str, dim: int) -> list[str]:
    """Apply schema.sql. Idempotent - every statement is CREATE ... IF NOT EXISTS."""
    sql = SCHEMA_PATH.read_text(encoding="utf-8").format(dim=dim)
    applied: list[str] = []
    with connect(url) as conn:
        for statement in _split(sql):
            conn.execute(statement)
            applied.append(_label(statement))
    log.info("migration applied %d statements", len(applied))
    return applied


def _split(sql: str) -> list[str]:
    lines = [ln for ln in sql.splitlines() if not ln.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def _label(statement: str) -> str:
    return " ".join(statement.split()[:4])


def _redact(url: str) -> str:
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.partition("@")
    return f"{scheme}://***@{host}"

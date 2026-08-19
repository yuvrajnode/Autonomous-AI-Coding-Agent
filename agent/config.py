"""Runtime configuration.

Everything is read from the environment (or a local .env) exactly once and passed
down explicitly, so tests can build a Settings object without touching os.environ.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="ACA_", extra="ignore", case_sensitive=False
    )

    # --- LLM ---------------------------------------------------------------
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    model: str = "claude-sonnet-5"
    planner_model: str = "claude-opus-5"
    max_tokens: int = 4096
    temperature: float = 0.0
    llm_max_retries: int = 4

    # --- Embeddings --------------------------------------------------------
    embedding_provider: Literal["voyage", "hashing"] = "hashing"
    embedding_model: str = "voyage-3"
    embedding_dim: int = 1024
    voyage_api_key: str = Field(default="", alias="VOYAGE_API_KEY")

    # --- Storage -----------------------------------------------------------
    database_url: str = "postgresql://aca:aca@localhost:5433/aca"

    # --- Budgets -----------------------------------------------------------
    max_iterations: int = 24
    max_replans: int = 3
    step_timeout: int = 120
    usd_budget: float = 1.50

    # --- Sandbox -----------------------------------------------------------
    workspace_root: Path = Path("./workspaces")
    shell_timeout: int = 60
    max_file_bytes: int = 512_000

    # --- Retrieval ---------------------------------------------------------
    retrieval_top_k: int = 6
    retrieval_min_score: float = 0.25
    chunk_tokens: int = 400
    chunk_overlap: int = 60

    # --- Observability -----------------------------------------------------
    trace_dir: Path = Path("./traces")
    log_level: str = "INFO"

    @field_validator("temperature")
    @classmethod
    def _sane_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("temperature must be between 0 and 1")
        return v

    @property
    def has_llm_credentials(self) -> bool:
        return bool(self.anthropic_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

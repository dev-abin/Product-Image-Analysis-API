"""Validated application configuration, loaded once during startup."""

import os
from pathlib import Path

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True, allow_inf_nan=False)

    validation_workers: int = Field(
        default_factory=lambda: min(4, os.cpu_count() or 1), ge=1, alias="VALIDATION_WORKERS",
    )
    validation_max_waiting: int = Field(default=96, ge=0, alias="VALIDATION_MAX_WAITING")
    validation_wait_timeout: float = Field(default=3.0, gt=0, alias="VALIDATION_WAIT_TIMEOUT")
    log_queue_capacity: int = Field(default=1000, ge=1, alias="LOG_QUEUE_CAPACITY")

    gemini_api_key: SecretStr | None = Field(default=None, repr=False, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-flash-latest", alias="GEMINI_MODEL")
    gemini_max_concurrent: int = Field(default=5, ge=1, alias="GEMINI_MAX_CONCURRENT")
    gemini_max_waiting: int = Field(default=20, ge=0, alias="GEMINI_MAX_WAITING")
    gemini_requests_per_minute: float = Field(default=0.0, ge=0, alias="GEMINI_REQUESTS_PER_MINUTE")
    gemini_queue_timeout: float = Field(default=3.0, gt=0, alias="GEMINI_QUEUE_TIMEOUT")
    gemini_total_timeout: float = Field(default=30.0, gt=0, alias="GEMINI_TOTAL_TIMEOUT")
    gemini_attempt_timeout: float = Field(default=10.0, gt=0, alias="GEMINI_ATTEMPT_TIMEOUT")
    gemini_retry_backoff: float = Field(default=0.5, ge=0, alias="GEMINI_RETRY_BACKOFF")

    @classmethod
    def from_env(cls, env_file: Path | None = Path(__file__).with_name(".env")) -> "Settings":
        """Read .env without changing process state; environment overrides the file."""
        values = dotenv_values(env_file) if env_file is not None else {}
        values.update(os.environ)
        return cls.model_validate({
            field.alias: values[field.alias]
            for field in cls.model_fields.values()
            if field.alias in values and values[field.alias] is not None
        })

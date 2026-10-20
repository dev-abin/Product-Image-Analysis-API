import os

import pytest
from pydantic import ValidationError

from settings import Settings


@pytest.fixture(autouse=True)
def clear_settings_environment(monkeypatch):
    for field in Settings.model_fields.values():
        monkeypatch.delenv(field.alias, raising=False)


def test_defaults_without_env_file():
    settings = Settings(gemini_model="test-model")
    assert settings.validation_workers == min(4, os.cpu_count() or 1)
    assert settings.validation_max_waiting == 96
    assert settings.validation_wait_timeout == 3
    assert settings.log_queue_capacity == 1000
    assert settings.gemini_api_key is None
    assert settings.gemini_model == "test-model"
    assert settings.gemini_max_concurrent == 5
    assert settings.gemini_max_waiting == 20
    assert settings.gemini_requests_per_minute == 0
    assert settings.gemini_queue_timeout == 3
    assert settings.gemini_total_timeout == 30
    assert settings.gemini_attempt_timeout == 10
    assert settings.gemini_retry_backoff == 0.5


def test_env_overrides_file_without_mutating_process_environment(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY=file-secret\nGEMINI_MODEL=test-model\nGEMINI_MAX_CONCURRENT=2\n"
        "VALIDATION_WAIT_TIMEOUT=1.5\nUNRELATED_SETTING=ignored\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GEMINI_API_KEY", "environment-secret")
    monkeypatch.setenv("GEMINI_MAX_CONCURRENT", "3")
    settings = Settings.from_env(env_file)
    assert settings.gemini_api_key.get_secret_value() == "environment-secret"
    assert settings.gemini_model == "test-model"
    assert settings.gemini_max_concurrent == 3
    assert settings.validation_wait_timeout == 1.5
    assert "GEMINI_MODEL" not in os.environ
    assert "UNRELATED_SETTING" not in os.environ
    assert "environment-secret" not in repr(settings)
    assert "environment-secret" not in settings.model_dump_json()


def test_environment_model_overrides_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_MODEL=file-model\n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_MODEL", "environment-model")
    assert Settings.from_env(env_file).gemini_model == "environment-model"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_model_must_be_configured(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv("GEMINI_MODEL", value)
    with pytest.raises(ValidationError, match="GEMINI_MODEL"):
        Settings.from_env(env_file=None)


@pytest.mark.parametrize("key,value", [
    ("VALIDATION_WORKERS", "0"), ("VALIDATION_MAX_WAITING", "-1"),
    ("VALIDATION_WAIT_TIMEOUT", "nan"), ("LOG_QUEUE_CAPACITY", "0"),
    ("VALIDATION_WAIT_TIMEOUT", "0"), ("VALIDATION_WAIT_TIMEOUT", "inf"),
    ("GEMINI_MAX_CONCURRENT", "0"), ("GEMINI_MAX_WAITING", "-1"),
    ("GEMINI_REQUESTS_PER_MINUTE", "inf"), ("GEMINI_QUEUE_TIMEOUT", "0"),
    ("GEMINI_TOTAL_TIMEOUT", "nan"), ("GEMINI_ATTEMPT_TIMEOUT", "-1"),
    ("GEMINI_RETRY_BACKOFF", "-0.1"), ("VALIDATION_WORKERS", "invalid"),
])
def test_invalid_environment_is_rejected(monkeypatch, key, value):
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv(key, value)
    with pytest.raises(ValidationError, match=key):
        Settings.from_env(env_file=None)


def test_zero_waiting_pacing_and_backoff_are_valid(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    for key in ("VALIDATION_MAX_WAITING", "GEMINI_MAX_WAITING",
                "GEMINI_REQUESTS_PER_MINUTE", "GEMINI_RETRY_BACKOFF"):
        monkeypatch.setenv(key, "0")
    assert Settings.from_env(env_file=None).gemini_retry_backoff == 0


def test_settings_are_immutable():
    settings = Settings(gemini_model="test-model")
    with pytest.raises(ValidationError, match="frozen"):
        settings.gemini_max_concurrent = 10

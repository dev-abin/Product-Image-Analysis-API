import json
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import httpx
import pytest
from google.genai import errors

import analyse_image as module
from schemas import ProductAnalysis


@pytest.fixture
def provider(monkeypatch, facts_data):
    factory = MagicMock()
    client = factory.return_value.__enter__.return_value
    client.models.generate_content.return_value = SimpleNamespace(text=json.dumps(facts_data))
    monkeypatch.setattr(module.genai, "Client", factory)
    return factory, client.models.generate_content


def test_success_uses_image_schema_and_timeout(provider, facts_data):
    factory, generate = provider
    result = module.analyze_image(b"image-bytes", "image/png")
    assert result.model_dump() == facts_data
    options = factory.call_args.kwargs["http_options"]
    assert options.timeout == 30_000
    assert options.retry_options.attempts == 1
    assert factory.call_args.kwargs["api_key"] == "test-key"
    generate.assert_called_once()
    request = generate.call_args.kwargs
    assert request["contents"][0] == module.PROMPT
    assert request["contents"][1].inline_data.data == b"image-bytes"
    assert request["contents"][1].inline_data.mime_type == "image/png"
    assert request["config"].response_schema is ProductAnalysis
    assert request["config"].response_mime_type == "application/json"
    factory.return_value.__exit__.assert_called_once()


def test_false_booleans_are_valid(provider, facts_data):
    _, generate = provider
    facts_data.update(single_product=False, clean_packshot=False)
    generate.return_value.text = json.dumps(facts_data)
    assert module.analyze_image(b"image", "image/png").model_dump() == facts_data


def test_missing_key_skips_client(monkeypatch, provider):
    monkeypatch.delenv("GEMINI_API_KEY")
    assert module.analyze_image(b"image", "image/png") is None
    provider[0].assert_not_called()


@pytest.mark.parametrize("error", [
    httpx.ReadTimeout("timeout"), httpx.ConnectError("connection"),
    *[errors.APIError(code, {"message": "transient"})
      for code in [408, 429, 500, 502, 503, 504]],
])
@pytest.mark.parametrize("recovers", [True, False])
def test_transient_failure_retries_once(provider, facts_data, error, recovers):
    _, generate = provider
    generate.side_effect = [
        error, SimpleNamespace(text=json.dumps(facts_data)) if recovers else error,
    ]
    result = module.analyze_image(b"image", "image/png")
    assert generate.call_count == 2
    assert (result is not None) == recovers


@pytest.mark.parametrize("error", [
    errors.APIError(400, {"message": "bad request"}),
    errors.APIError(401, {"message": "unauthorized"}),
    errors.APIError(403, {"message": "forbidden"}),
    RuntimeError("unexpected provider failure"),
])
def test_non_retryable_errors(provider, error):
    _, generate = provider
    generate.side_effect = error
    assert module.analyze_image(b"image", "image/png") is None
    generate.assert_called_once()


@pytest.mark.parametrize("text", [None, "", "not JSON", "{}", "[]", "null", "```json\n{}\n```"])
def test_invalid_response_returns_none_without_retry(provider, text):
    _, generate = provider
    generate.return_value.text = text
    assert module.analyze_image(b"image", "image/png") is None
    generate.assert_called_once()


@pytest.mark.parametrize("changes", [
    {"single_product": "false"}, {"confidence": 1.1},
    {"worn_on": "head"}, {"extra_field": True},
])
def test_schema_invalid_response(provider, facts_data, changes):
    _, generate = provider
    generate.return_value.text = json.dumps({**facts_data, **changes})
    assert module.analyze_image(b"image", "image/png") is None
    generate.assert_called_once()


@pytest.mark.parametrize("stage", ["create", "enter", "exit"])
def test_client_lifecycle_errors_are_controlled(provider, stage):
    factory, _ = provider
    if stage == "create":
        factory.side_effect = RuntimeError("create")
    else:
        getattr(factory.return_value, f"__{stage}__").side_effect = RuntimeError(stage)
    assert module.analyze_image(b"image", "image/png") is None


def test_error_logs_do_not_expose_provider_details(monkeypatch, provider):
    _, generate = provider
    generate.side_effect = RuntimeError("secret-key and private image data")
    warning = Mock()
    monkeypatch.setattr(module.logger, "warning", warning)
    assert module.analyze_image(b"private image data", "image/png") is None
    warning.assert_called_once()
    assert warning.call_args.args == ("llm.error",)
    assert warning.call_args.kwargs["extra"]["error_type"] == "RuntimeError"
    assert warning.call_args.kwargs["extra"]["stage"] == "provider_call"
    assert "secret-key" not in str(warning.call_args)
    assert "private image data" not in str(warning.call_args)


def test_retry_and_completion_logs(monkeypatch, provider, facts_data, caplog):
    monkeypatch.setattr(module.logger, "handlers", [caplog.handler])
    monkeypatch.setattr(module.logger, "propagate", False)
    caplog.set_level("INFO", logger="image_api")
    provider[1].side_effect = [
        httpx.ReadTimeout("private provider detail"),
        SimpleNamespace(text=json.dumps(facts_data)),
    ]
    assert module.analyze_image(b"image", "image/png") is not None
    logs = [r for r in caplog.records if r.name == "image_api"]
    assert [r.getMessage() for r in logs] == [
        "llm.attempt", "llm.retry", "llm.attempt", "llm.completed",
    ]
    assert logs[1].attempt == 1
    assert logs[1].error_type == "ReadTimeout"
    assert logs[-1].attempt == 2
    assert logs[-1].latency_ms >= 0


def test_invalid_output_log_identifies_validation_stage(monkeypatch, provider):
    warning = Mock()
    monkeypatch.setattr(module.logger, "warning", warning)
    provider[1].return_value.text = "not JSON"
    assert module.analyze_image(b"image", "image/png") is None
    assert warning.call_args.kwargs["extra"]["stage"] == "output_validation"
    assert warning.call_args.kwargs["extra"]["error_type"] == "ValidationError"

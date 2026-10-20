import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from google.genai import errors

import analyse_image as module
import gemini_resources as resources
from schemas import ProductAnalysis
from settings import Settings


@pytest.fixture
def provider(monkeypatch, facts_data):
    generate = AsyncMock(return_value=SimpleNamespace(text=json.dumps(facts_data)))
    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate), aclose=AsyncMock())
    factory = Mock(return_value=Mock(aio=client))
    monkeypatch.setattr(resources.genai, "Client", factory)
    return factory, generate


def analyze(payload, mime, **options):
    async def run():
        async with resources.gemini_resources(Settings.from_env(env_file=None)) as shared:
            client = shared.client
            return await module.analyze_image(
                payload, mime, client, asyncio.Semaphore(5),
                model=shared.settings.gemini_model, retry_backoff=0, **options,
            )
    return asyncio.run(run())


def test_success_uses_image_schema_and_timeout(provider, facts_data):
    factory, generate = provider
    result = analyze(b"image-bytes", "image/png")
    assert result.model_dump() == facts_data
    options = factory.call_args.kwargs["http_options"]
    assert options.timeout == 10_000
    assert options.retry_options.attempts == 1
    assert factory.call_args.kwargs["api_key"] == "test-key"
    generate.assert_awaited_once()
    request = generate.call_args.kwargs
    assert request["model"] == "test-model"
    assert request["contents"][0] == module.PROMPT
    assert request["contents"][1].inline_data.data == b"image-bytes"
    assert request["contents"][1].inline_data.mime_type == "image/png"
    assert request["config"].response_schema is ProductAnalysis
    assert request["config"].response_mime_type == "application/json"
    factory.return_value.aio.aclose.assert_awaited_once()
    factory.return_value.close.assert_called_once()


def test_false_booleans_are_valid(provider, facts_data):
    _, generate = provider
    facts_data.update(single_product=False, clean_packshot=False)
    generate.return_value.text = json.dumps(facts_data)
    assert analyze(b"image", "image/png").model_dump() == facts_data


def test_missing_key_skips_client(monkeypatch, provider):
    monkeypatch.delenv("GEMINI_API_KEY")
    assert analyze(b"image", "image/png") is None
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
    result = analyze(b"image", "image/png")
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
    assert analyze(b"image", "image/png") is None
    generate.assert_called_once()


@pytest.mark.parametrize("text", [None, "", "not JSON", "{}", "[]", "null", "```json\n{}\n```"])
def test_invalid_response_returns_none_without_retry(provider, text):
    _, generate = provider
    generate.return_value.text = text
    assert analyze(b"image", "image/png") is None
    generate.assert_called_once()


@pytest.mark.parametrize("changes", [
    {"single_product": "false"}, {"confidence": 1.1},
    {"worn_on": "head"}, {"extra_field": True},
])
def test_schema_invalid_response(provider, facts_data, changes):
    _, generate = provider
    generate.return_value.text = json.dumps({**facts_data, **changes})
    assert analyze(b"image", "image/png") is None
    generate.assert_called_once()


@pytest.mark.parametrize("stage", ["create", "close_async", "close_sync"])
def test_client_lifecycle_errors_are_controlled(provider, stage):
    factory, _ = provider
    if stage == "create":
        factory.side_effect = RuntimeError("create")
    elif stage == "close_async":
        factory.return_value.aio.aclose.side_effect = RuntimeError(stage)
    else:
        factory.return_value.close.side_effect = RuntimeError(stage)
    result = analyze(b"image", "image/png")
    assert (result is None) == (stage == "create")


def test_error_logs_do_not_expose_provider_details(monkeypatch, provider):
    _, generate = provider
    generate.side_effect = RuntimeError("secret-key and private image data")
    warning = Mock()
    monkeypatch.setattr(module.logger, "warning", warning)
    assert analyze(b"private image data", "image/png") is None
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
    assert analyze(b"image", "image/png") is not None
    logs = [r for r in caplog.records if r.name == "image_api"]
    assert [r.getMessage() for r in logs] == [
        "llm.dispatched", "llm.attempt", "llm.retry", "llm.attempt", "llm.completed",
    ]
    assert logs[2].attempt == 1
    assert logs[2].error_type == "ReadTimeout"
    assert logs[-1].attempt == 2
    assert logs[-1].latency_ms >= 0


def test_invalid_output_log_identifies_validation_stage(monkeypatch, provider):
    warning = Mock()
    monkeypatch.setattr(module.logger, "warning", warning)
    provider[1].return_value.text = "not JSON"
    assert analyze(b"image", "image/png") is None
    assert warning.call_args.kwargs["extra"]["stage"] == "output_validation"
    assert warning.call_args.kwargs["extra"]["error_type"] == "ValidationError"


def test_client_is_reused_for_multiple_images(provider):
    async def run():
        async with resources.gemini_resources(Settings.from_env(env_file=None)) as shared:
            client = shared.client
            semaphore = asyncio.Semaphore(5)
            for _ in range(2):
                assert await module.analyze_image(
                    b"image", "image/png", client, semaphore, model=shared.settings.gemini_model,
                ) is not None
    asyncio.run(run())
    provider[0].assert_called_once()
    assert provider[1].await_count == 2
    provider[0].return_value.aio.aclose.assert_awaited_once()


def test_concurrency_queue_bound_and_cancellation(provider, facts_data):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        async def blocked(**kwargs):
            entered.set()
            await release.wait()
            return SimpleNamespace(text=json.dumps(facts_data))

        provider[1].side_effect = blocked
        semaphore, admission = asyncio.Semaphore(1), asyncio.Queue(maxsize=2)
        async with resources.gemini_resources(Settings.from_env(env_file=None)) as shared:
            client = shared.client
            def call():
                return module.analyze_image(b"image", "image/png", client, semaphore,
                                            model=shared.settings.gemini_model,
                                            admission=admission, retry_backoff=0)
            first = asyncio.create_task(call())
            await asyncio.wait_for(entered.wait(), timeout=1)
            second = asyncio.create_task(call())
            await asyncio.sleep(0)
            assert admission.qsize() == 2
            assert provider[1].await_count == 1
            with pytest.raises(module.GeminiBusy):
                await call()
            second.cancel()
            with pytest.raises(asyncio.CancelledError):
                await second
            assert admission.qsize() == 1
            release.set()
            assert await first is not None
            assert admission.empty()
            assert semaphore._value == 1
    asyncio.run(run())


def test_queue_wait_deadline_returns_busy_without_provider_call(provider):
    async def run():
        async with resources.gemini_resources(Settings.from_env(env_file=None)) as shared:
            client = shared.client
            admission = asyncio.Queue(maxsize=1)
            with pytest.raises(module.GeminiBusy):
                await module.analyze_image(b"image", "image/png", client, asyncio.Semaphore(0),
                                           model=shared.settings.gemini_model,
                                           admission=admission, queue_timeout=0.01)
            assert admission.empty()
    asyncio.run(run())
    provider[1].assert_not_awaited()


@pytest.mark.parametrize("mode", ["attempt", "total", "backoff", "rate_wait"])
def test_deadlines_cover_attempts_backoff_and_pacing(provider, mode):
    async def blocked(**kwargs):
        await asyncio.Event().wait()

    async def run():
        semaphore, admission = asyncio.Semaphore(1), asyncio.Queue(maxsize=1)
        options = {"retry_backoff": 0, "attempt_timeout": 1, "total_timeout": 1}
        provider[1].side_effect = blocked
        if mode == "attempt":
            options["attempt_timeout"] = 0.01
        else:
            options["total_timeout"] = 0.01
        if mode == "backoff":
            options["retry_backoff"] = 1
            provider[1].side_effect = httpx.ReadTimeout("temporary")
        if mode == "rate_wait":
            options["pacing"] = {"lock": asyncio.Lock(), "interval": 1,
                                  "next_at": asyncio.get_running_loop().time() + 100}
        async with resources.gemini_resources(Settings.from_env(env_file=None)) as shared:
            client = shared.client
            call = module.analyze_image(b"image", "image/png", client, semaphore,
                                        model=shared.settings.gemini_model,
                                        admission=admission, **options)
            if mode == "rate_wait":
                with pytest.raises(module.GeminiBusy):
                    await call
            else:
                assert await call is None
        assert admission.empty()
        assert semaphore._value == 1
    asyncio.run(run())
    assert provider[1].await_count == {"attempt": 2, "total": 1, "backoff": 1, "rate_wait": 0}[mode]


def test_cancelling_active_call_releases_capacity(provider):
    async def run():
        entered = asyncio.Event()
        async def blocked(**kwargs):
            entered.set()
            await asyncio.Event().wait()
        provider[1].side_effect = blocked
        semaphore, admission = asyncio.Semaphore(1), asyncio.Queue(maxsize=1)
        async with resources.gemini_resources(Settings.from_env(env_file=None)) as shared:
            client = shared.client
            task = asyncio.create_task(module.analyze_image(
                b"image", "image/png", client, semaphore,
                model=shared.settings.gemini_model, admission=admission,
            ))
            await asyncio.wait_for(entered.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert admission.empty()
            assert semaphore._value == 1
    asyncio.run(run())


def test_request_pacing(monkeypatch):
    async def run():
        clock = Mock()
        clock.time.side_effect = [10, 10, 10.5, 11]
        sleep = AsyncMock()
        with monkeypatch.context() as patch:
            patch.setattr(module.asyncio, "get_running_loop", lambda: clock)
            patch.setattr(module.asyncio, "sleep", sleep)
            pacing = {"lock": asyncio.Lock(), "interval": 1, "next_at": 0}
            await module.pace_request(pacing)
            await module.pace_request(pacing)
            sleep.assert_awaited_once_with(0.5)
            assert pacing["next_at"] == 12
    asyncio.run(run())


@pytest.mark.parametrize("header,minimum", [("2", 2), ("invalid", 0.5), ("nan", 0.5), ("-1", 0.5)])
def test_retry_after_and_jitter(monkeypatch, header, minimum):
    monkeypatch.setattr(module.random, "uniform", lambda low, high: 0.25)
    error = errors.APIError(429, {}, response=httpx.Response(429, headers={"Retry-After": header}))
    assert module.retry_delay(error, 0.5) == max(0.75, minimum)

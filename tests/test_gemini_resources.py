import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import gemini_resources as module
from settings import Settings


@pytest.mark.parametrize("fail_request", [False, True])
def test_component_reuses_capacity_forwards_settings_and_closes_on_exit(monkeypatch, fail_request):
    settings = Settings(
        gemini_api_key="test-key",
        gemini_max_concurrent=2, gemini_max_waiting=3, gemini_requests_per_minute=60,
        gemini_model="test-model", gemini_queue_timeout=1.5, gemini_total_timeout=12,
        gemini_attempt_timeout=4, gemini_retry_backoff=0.25,
    )
    client = SimpleNamespace(aclose=AsyncMock())
    owner = Mock(aio=client)
    factory = Mock(return_value=owner)
    analyze = AsyncMock(return_value=None)
    monkeypatch.setattr(module.genai, "Client", factory)
    monkeypatch.setattr(module, "analyze_image", analyze)

    async def run():
        async with module.gemini_resources(settings) as resources:
            assert resources.semaphore._value == 2
            assert resources.admission.maxsize == 5
            assert resources.pacing["interval"] == 1
            for _ in range(2):
                await resources.analyze(b"image", "image/png")
            for call in analyze.await_args_list:
                assert call.args == (b"image", "image/png", client, resources.semaphore)
                assert call.kwargs == {
                    "admission": resources.admission, "pacing": resources.pacing,
                    "model": "test-model", "queue_timeout": 1.5, "total_timeout": 12,
                    "attempt_timeout": 4, "retry_backoff": 0.25,
                }
            client.aclose.assert_not_awaited()
            owner.close.assert_not_called()
            if fail_request:
                raise RuntimeError("request failed")

    if fail_request:
        with pytest.raises(RuntimeError, match="request failed"):
            asyncio.run(run())
    else:
        asyncio.run(run())
    factory.assert_called_once()
    assert factory.call_args.kwargs["http_options"].timeout == 4000
    client.aclose.assert_awaited_once()
    owner.close.assert_called_once()


def test_resource_setup_failure_still_closes_both_transports(monkeypatch):
    client = SimpleNamespace(aclose=AsyncMock())
    owner = Mock(aio=client)
    monkeypatch.setattr(module.genai, "Client", Mock(return_value=owner))
    monkeypatch.setattr(module, "GeminiResources", Mock(side_effect=RuntimeError("setup failed")))

    async def run():
        async with module.gemini_resources(Settings(gemini_api_key="test-key", gemini_model="test-model")):
            pytest.fail("Failed resource setup must not enter the context body")

    with pytest.raises(RuntimeError, match="setup failed"):
        asyncio.run(run())
    client.aclose.assert_awaited_once()
    owner.close.assert_called_once()


def test_pacing_is_disabled_by_default():
    assert module.GeminiResources(Settings(gemini_model="test-model"), client=None).pacing is None

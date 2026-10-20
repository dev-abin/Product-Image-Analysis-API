"""Own the shared Gemini client and per-process analysis capacity."""

import asyncio
from contextlib import asynccontextmanager

from google import genai
from google.genai import types

from analyse_image import analyze_image
from request_logging import logger
from schemas import ProductAnalysis
from settings import Settings


class GeminiResources:
    def __init__(self, settings: Settings, client):
        self.settings = settings
        self.client = client
        self.semaphore = asyncio.Semaphore(settings.gemini_max_concurrent)
        self.admission = asyncio.Queue(
            maxsize=settings.gemini_max_concurrent + settings.gemini_max_waiting,
        )
        rpm = settings.gemini_requests_per_minute
        self.pacing = {"lock": asyncio.Lock(), "next_at": 0.0, "interval": 60 / rpm} if rpm else None

    async def analyze(self, image_bytes: bytes, mime_type: str) -> ProductAnalysis | None:
        return await analyze_image(
            image_bytes, mime_type, self.client, self.semaphore,
            admission=self.admission, pacing=self.pacing,
            model=self.settings.gemini_model,
            queue_timeout=self.settings.gemini_queue_timeout,
            total_timeout=self.settings.gemini_total_timeout,
            attempt_timeout=self.settings.gemini_attempt_timeout,
            retry_backoff=self.settings.gemini_retry_backoff,
        )


@asynccontextmanager
async def gemini_resources(settings: Settings):
    """Create shared analysis resources; close both SDK transports on exit."""
    owner = client = None
    try:
        key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else None
        if not key:
            raise ValueError("missing_api_key")
        owner = genai.Client(api_key=key, http_options=types.HttpOptions(
            timeout=max(1, int(settings.gemini_attempt_timeout * 1000)),
            retry_options=types.HttpRetryOptions(attempts=1),
        ))
        client = owner.aio
    except Exception as exc:
        logger.warning("llm.error", extra={
            "stage": "provider_initialization", "error_type": type(exc).__name__,
        })
    try:
        yield GeminiResources(settings, client)
    finally:
        try:
            if client is not None:
                await client.aclose()
        except Exception as exc:
            logger.warning("llm.cleanup_error", extra={"error_type": type(exc).__name__})
        finally:
            if owner is not None:
                try:
                    await asyncio.to_thread(owner.close)
                except Exception as exc:
                    logger.warning("llm.cleanup_error", extra={"error_type": type(exc).__name__})

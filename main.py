import asyncio
import math
import os
from concurrent.futures.process import BrokenProcessPool
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from analyse_image import GeminiBusy, analyze_image, shared_gemini_client
from request_logging import (
    RequestLoggingMiddleware, logger, request_id_context, start_queue_logging, stop_queue_logging,
)
from schemas import ProductAnalysis, UploadResponse
from validation_pool import ValidationBusy, ValidationPool


@asynccontextmanager
async def application_lifespan(api: FastAPI, log_runtime):
    pool = ValidationPool(
        workers=int(os.getenv("VALIDATION_WORKERS", str(min(4, os.cpu_count() or 1)))),
        max_waiting=int(os.getenv("VALIDATION_MAX_WAITING", "96")),
        wait_timeout=float(os.getenv("VALIDATION_WAIT_TIMEOUT", "3")),
        log_queue=log_runtime["queue"], dropped=log_runtime["dropped"],
    )
    api.state.validation_pool = pool
    try:
        concurrent = int(os.getenv("GEMINI_MAX_CONCURRENT", "5"))
        waiting = int(os.getenv("GEMINI_MAX_WAITING", "20"))
        rpm = float(os.getenv("GEMINI_REQUESTS_PER_MINUTE", "0"))
        options = {
            "queue_timeout": float(os.getenv("GEMINI_QUEUE_TIMEOUT", "3")),
            "total_timeout": float(os.getenv("GEMINI_TOTAL_TIMEOUT", "30")),
            "attempt_timeout": float(os.getenv("GEMINI_ATTEMPT_TIMEOUT", "10")),
            "retry_backoff": float(os.getenv("GEMINI_RETRY_BACKOFF", "0.5")),
        }
        if (concurrent < 1 or waiting < 0 or not math.isfinite(rpm) or rpm < 0
                or any(not math.isfinite(v) or v <= 0 for k, v in options.items() if k != "retry_backoff")
                or not math.isfinite(options["retry_backoff"]) or options["retry_backoff"] < 0):
            raise ValueError("Invalid Gemini configuration")
        async with shared_gemini_client(options["attempt_timeout"]) as client:
            api.state.gemini = {
                **options, "client": client,
                "model": os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
                "semaphore": asyncio.Semaphore(concurrent),
                "admission": asyncio.Queue(maxsize=concurrent + waiting),
                "pacing": {"lock": asyncio.Lock(), "next_at": 0.0, "interval": 60 / rpm} if rpm else None,
            }
            yield
    finally:
        await pool.close()


@asynccontextmanager
async def lifespan(api: FastAPI):
    runtime = start_queue_logging(capacity=int(os.getenv("LOG_QUEUE_CAPACITY", "1000")))
    api.state.log_runtime = runtime
    try:
        async with application_lifespan(api, runtime):
            yield
    finally:
        await asyncio.to_thread(stop_queue_logging, runtime)


app = FastAPI(lifespan=lifespan)


def decide(facts: ProductAnalysis | None) -> dict:
    """Apply business rules in order; the LLM supplies facts only."""
    if facts is None:
        return {"status": "needs_review", "reasons": ["llm_error"], "facts": None}

    if not facts.single_product:
        status, reasons = "fail", ["not_single_product"]
    elif not facts.clean_packshot:
        status, reasons = "fail", ["not_packshot"]
    elif facts.confidence < 0.6:
        status, reasons = "needs_review", ["low_confidence"]
    else:
        status, reasons = "pass", []

    return {"status": status, "reasons": reasons, "facts": facts.model_dump()}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/upload", response_model=UploadResponse)
async def upload_image(
    request: Request,
    file: UploadFile = File(..., json_schema_extra={"format": "binary"}),
):
    try:
        image_bytes = await file.read()
        logger.info("upload.received", extra={"stage": "upload", "size_bytes": len(image_bytes)})
        validation = await request.app.state.validation_pool.validate(
            image_bytes, request_id_context.get(),
        )

        if validation["status"] == "fail":
            logger.info("llm.skipped", extra={"stage": "llm", "reasons": validation["reasons"]})
            logger.info("decision.completed", extra={
                "stage": "decision", "status": "fail", "reasons": validation["reasons"],
            })
            return {**validation, "facts": None}

        facts = await analyze_image(
            image_bytes,
            file.content_type or "application/octet-stream",
            **request.app.state.gemini,
        )

        decision = decide(facts)
        logger.info("decision.completed", extra={
            "stage": "decision", "status": decision["status"], "reasons": decision["reasons"],
        })
        return {
            **decision,
            "scores": validation["scores"],
        }

    except GeminiBusy as exc:
        logger.warning("llm.unavailable", extra={"stage": "llm_queue", "error_type": type(exc).__name__})
        raise HTTPException(
            status_code=503, detail="Gemini analysis is busy; retry later", headers={"Retry-After": "1"},
        ) from exc
    except (ValidationBusy, BrokenProcessPool) as exc:
        logger.warning("validation.unavailable", extra={
            "stage": "validation_queue", "error_type": type(exc).__name__,
        })
        raise HTTPException(
            status_code=503, detail="Image validation is busy or unavailable; retry later",
            headers={"Retry-After": "1"},
        ) from exc
    except ValueError as exc:
        logger.warning("upload.invalid", extra={"stage": "upload", "error_type": type(exc).__name__})
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()


# Wrap the complete app so even unhandled 500 responses get a request ID.
app = RequestLoggingMiddleware(app)

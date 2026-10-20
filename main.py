import asyncio
from concurrent.futures.process import BrokenProcessPool
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from analyse_image import GeminiBusy
from gemini_resources import gemini_resources
from request_logging import (
    RequestLoggingMiddleware, logger, request_id_context, start_queue_logging, stop_queue_logging,
)
from schemas import ProductAnalysis, UploadResponse
from settings import Settings
from validation_pool import ValidationBusy, ValidationPool


@asynccontextmanager
async def lifespan(api: FastAPI):
    settings = Settings.from_env()
    api.state.settings = settings
    runtime = start_queue_logging(capacity=settings.log_queue_capacity)
    api.state.log_runtime = runtime
    try:
        pool = ValidationPool(
            workers=settings.validation_workers,
            max_waiting=settings.validation_max_waiting,
            wait_timeout=settings.validation_wait_timeout,
            log_queue=runtime["queue"],
        )
        api.state.validation_pool = pool
        try:
            async with gemini_resources(settings) as gemini:
                api.state.gemini = gemini
                yield
        finally:
            await pool.close()
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

        facts = await request.app.state.gemini.analyze(
            image_bytes,
            file.content_type or "application/octet-stream",
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

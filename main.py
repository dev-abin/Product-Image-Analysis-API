from fastapi import FastAPI, File, HTTPException, UploadFile

from analyse_image import analyze_image
from request_logging import RequestLoggingMiddleware, logger
from schemas import ProductAnalysis, UploadResponse
from validate_image import validate_image

app = FastAPI()


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
    file: UploadFile = File(..., json_schema_extra={"format": "binary"}),
):
    try:
        image_bytes = await file.read()
        logger.info("upload.received", extra={"stage": "upload", "size_bytes": len(image_bytes)})
        validation = validate_image(image_bytes)

        if validation["status"] == "fail":
            logger.info("llm.skipped", extra={"stage": "llm", "reasons": validation["reasons"]})
            logger.info("decision.completed", extra={
                "stage": "decision", "status": "fail", "reasons": validation["reasons"],
            })
            return {**validation, "facts": None}

        facts = analyze_image(
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

    except ValueError as exc:
        logger.warning("upload.invalid", extra={"stage": "upload", "error_type": type(exc).__name__})
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()


# Wrap the complete app so even unhandled 500 responses get a request ID.
app = RequestLoggingMiddleware(app)

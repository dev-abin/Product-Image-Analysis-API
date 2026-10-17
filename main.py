from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from analyse_image import ProductAnalysis, analyze_image
from request_logging import RequestLoggingMiddleware
from validate_image import validate_image

app = FastAPI()


class ImageScores(BaseModel):
    min_side: int | None
    sharpness: float | None


class UploadResponse(BaseModel):
    status: Literal["pass", "fail", "needs_review"]
    reasons: list[str]
    scores: ImageScores
    facts: ProductAnalysis | None


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
        validation = validate_image(image_bytes)

        if validation["status"] == "fail":
            return {**validation, "facts": None}

        facts = analyze_image(
            image_bytes,
            file.content_type or "application/octet-stream",
        )

        return {
            **decide(facts),
            "scores": validation["scores"],
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()


# Wrap the complete app so even unhandled 500 responses get a request ID.
app = RequestLoggingMiddleware(app)

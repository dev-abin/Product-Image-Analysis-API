from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ProductAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    single_product: bool
    clean_packshot: bool
    product_type: str
    worn_on: Literal["ear", "neck", "wrist", "finger", "face", "none"]
    sits_in_room: bool
    confidence: float = Field(ge=0, le=1)


class ImageScores(BaseModel):
    min_side: int | None
    sharpness: float | None


class UploadResponse(BaseModel):
    status: Literal["pass", "fail", "needs_review"]
    reasons: list[str]
    scores: ImageScores
    facts: ProductAnalysis | None

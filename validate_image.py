from io import BytesIO
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from request_logging import logger


def validate_image(image_bytes: bytes) -> dict:
    started = perf_counter()
    logger.info("validation.started", extra={"stage": "validation"})
    try:
        with Image.open(BytesIO(image_bytes)) as uploaded:
            image = ImageOps.exif_transpose(uploaded).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        logger.info("validation.completed", extra={
            "stage": "validation", "status": "fail", "reasons": ["invalid_image"],
            "latency_ms": round((perf_counter() - started) * 1000, 3),
        })
        return {
            "status": "fail",
            "reasons": ["invalid_image"],
            "scores": {
                "min_side": None,
                "sharpness": None,
            },
        }

    width, height = image.size
    shorter_side = min(width, height)

    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    sharpness_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    reasons = []

    if shorter_side < 1000:
        reasons.append("low_resolution")

    if sharpness_score < 50:
        reasons.append("blurry")

    logger.info("validation.completed", extra={
        "stage": "validation", "status": "fail" if reasons else "pass",
        "reasons": reasons, "min_side": shorter_side,
        "sharpness": round(sharpness_score, 2),
        "latency_ms": round((perf_counter() - started) * 1000, 3),
    })
    return {
        "status": "fail" if reasons else "pass",
        "reasons": reasons,
        "scores": {
            "min_side": shorter_side,
            "sharpness": round(sharpness_score, 2),
        },
    }

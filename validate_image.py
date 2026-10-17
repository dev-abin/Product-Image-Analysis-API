from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


def validate_image(image_bytes: bytes) -> dict:
    try:
        with Image.open(BytesIO(image_bytes)) as uploaded:
            image = ImageOps.exif_transpose(uploaded).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
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

    return {
        "status": "fail" if reasons else "pass",
        "reasons": reasons,
        "scores": {
            "min_side": shorter_side,
            "sharpness": round(sharpness_score, 2),
        },
    }

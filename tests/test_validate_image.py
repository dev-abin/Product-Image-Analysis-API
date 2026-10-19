from io import BytesIO
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from PIL import Image

import validate_image as module


@pytest.mark.parametrize("payload", [b"", b"not an image", b"\x89PNG\r\n\x1a\n"])
def test_invalid_image(payload):
    assert module.validate_image(payload) == {
        "status": "fail", "reasons": ["invalid_image"],
        "scores": {"min_side": None, "sharpness": None},
    }


@pytest.mark.parametrize("error", [OSError("decode"), ValueError("decode")])
def test_decode_errors_are_controlled(monkeypatch, error):
    monkeypatch.setattr(module.Image, "open", Mock(side_effect=error))
    assert module.validate_image(b"image")["reasons"] == ["invalid_image"]


@pytest.mark.parametrize("size,sharpness,reasons", [
    ((1000, 1200), 50, []),
    ((999, 1200), 50, ["low_resolution"]),
    ((1200, 999), 50, ["low_resolution"]),
    ((1001, 1000), 49.999, ["blurry"]),
    ((800, 1200), 0, ["low_resolution", "blurry"]),
    ((1000, 1000), 50.001, []),
])
def test_quality_boundaries(monkeypatch, make_image, size, sharpness, reasons):
    # Isolate exact thresholds; separate tests below exercise real OpenCV.
    laplacian = Mock()
    laplacian.var.return_value = sharpness
    monkeypatch.setattr(module.cv2, "Laplacian", Mock(return_value=laplacian))
    result = module.validate_image(make_image(size=size))
    assert result == {
        "status": "fail" if reasons else "pass", "reasons": reasons,
        "scores": {"min_side": min(size), "sharpness": round(sharpness, 2)},
    }


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P"])
def test_uniform_images_are_blurry(make_image, mode):
    result = module.validate_image(make_image(mode=mode))
    assert result["reasons"] == ["blurry"]
    assert result["scores"] == {"min_side": 1000, "sharpness": 0.0}


def test_real_sharp_image_passes():
    pixels = ((np.indices((1000, 1000)).sum(axis=0) % 2) * 255).astype(np.uint8)
    output = BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    result = module.validate_image(output.getvalue())
    assert result["status"] == "pass"
    assert result["scores"]["sharpness"] == round(
        float(cv2.Laplacian(pixels, cv2.CV_64F).var()), 2,
    )


def test_exif_orientation_is_applied(monkeypatch):
    image = Image.new("RGB", (1200, 1000))
    exif = Image.Exif()
    exif[274] = 6  # Rotate 90 degrees clockwise.
    output = BytesIO()
    image.save(output, format="JPEG", exif=exif)
    convert = Mock(wraps=module.cv2.cvtColor)
    monkeypatch.setattr(module.cv2, "cvtColor", convert)
    module.validate_image(output.getvalue())
    assert convert.call_args.args[0].shape == (1200, 1000, 3)

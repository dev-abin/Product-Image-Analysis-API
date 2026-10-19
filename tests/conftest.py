from io import BytesIO
from unittest.mock import Mock

import pytest
from PIL import Image

import analyse_image


@pytest.fixture(autouse=True)
def block_real_gemini(monkeypatch):
    """Never use local credentials or contact Gemini during a test."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        analyse_image.genai, "Client",
        Mock(side_effect=AssertionError("Unexpected Gemini client creation")),
    )


@pytest.fixture
def facts_data():
    return {
        "single_product": True,
        "clean_packshot": True,
        "product_type": "earrings",
        "worn_on": "ear",
        "sits_in_room": False,
        "confidence": 0.9,
    }


@pytest.fixture
def make_image():
    """Generate small reproducible fixtures without checking images into Git."""
    def make(size=(1000, 1000), mode="RGB", color=128, **save_options):
        output = BytesIO()
        Image.new(mode, size, color).save(output, format="PNG", **save_options)
        return output.getvalue()

    return make

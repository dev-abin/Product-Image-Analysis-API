import asyncio
import json
from io import BytesIO
from unittest.mock import Mock

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

import main
from schemas import ProductAnalysis
from request_logging import create_file_handler, logger


@pytest.fixture
def client():
    with TestClient(main.app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def pipeline(monkeypatch, facts_data):
    validation = Mock(return_value={
        "status": "pass", "reasons": [],
        "scores": {"min_side": 1000, "sharpness": 60.5},
    })
    analysis = Mock(return_value=ProductAnalysis(**facts_data))
    monkeypatch.setattr(main, "validate_image", validation)
    monkeypatch.setattr(main, "analyze_image", analysis)
    return validation, analysis


@pytest.mark.parametrize("single,packshot,confidence,status,reasons", [
    (False, False, 0.1, "fail", ["not_single_product"]),
    (False, True, 0.9, "fail", ["not_single_product"]),
    (True, False, 0.1, "fail", ["not_packshot"]),
    (True, False, 0.9, "fail", ["not_packshot"]),
    (True, True, 0.5999, "needs_review", ["low_confidence"]),
    (True, True, 0.6, "pass", []),
    (True, True, 1, "pass", []),
])
def test_decision_precedence(facts_data, single, packshot, confidence, status, reasons):
    facts = ProductAnalysis(**{**facts_data, "single_product": single,
                              "clean_packshot": packshot, "confidence": confidence})
    assert main.decide(facts) == {"status": status, "reasons": reasons,
                                 "facts": facts.model_dump()}


def test_missing_facts_need_review():
    assert main.decide(None) == {"status": "needs_review", "reasons": ["llm_error"], "facts": None}


def test_health(client, pipeline):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
    for function in pipeline:
        function.assert_not_called()


@pytest.mark.parametrize("reasons", [
    ["invalid_image"], ["low_resolution"], ["blurry"], ["low_resolution", "blurry"],
])
def test_local_rejection_skips_analysis(client, pipeline, reasons):
    validation, analysis = pipeline
    validation.return_value.update(status="fail", reasons=reasons)
    response = client.post("/upload", files={"file": ("test.png", b"bytes", "image/png")})
    assert response.status_code == 200
    assert response.json() == {**validation.return_value, "facts": None}
    validation.assert_called_once_with(b"bytes")
    analysis.assert_not_called()


@pytest.mark.parametrize("changes,status,reasons", [
    ({}, "pass", []),
    ({"single_product": False}, "fail", ["not_single_product"]),
    ({"clean_packshot": False}, "fail", ["not_packshot"]),
    ({"confidence": 0.5}, "needs_review", ["low_confidence"]),
    (None, "needs_review", ["llm_error"]),
])
def test_upload_final_decision(client, pipeline, facts_data, changes, status, reasons):
    validation, analysis = pipeline
    facts = ProductAnalysis(**{**facts_data, **changes}) if changes is not None else None
    analysis.return_value = facts
    response = client.post("/upload", files={"file": ("test.png", b"bytes", "image/png")})
    assert response.status_code == 200
    assert response.json() == {
        "status": status, "reasons": reasons, "scores": validation.return_value["scores"],
        "facts": facts.model_dump() if facts else None,
    }
    analysis.assert_called_once_with(b"bytes", "image/png")


def test_missing_file_is_422(client, pipeline):
    assert client.post("/upload").status_code == 422
    for function in pipeline:
        function.assert_not_called()


@pytest.mark.parametrize("stage", [0, 1])
def test_value_errors_are_http_400(client, pipeline, stage):
    pipeline[stage].side_effect = ValueError("invalid input")
    response = client.post("/upload", files={"file": ("test.png", b"bytes", "image/png")})
    assert response.status_code == 400
    assert response.json() == {"detail": "invalid input"}


@pytest.mark.parametrize("error", [None, ValueError("invalid"), RuntimeError("unexpected")])
def test_upload_file_is_closed_even_on_error(pipeline, error):
    file = UploadFile(file=BytesIO(b"bytes"), filename="test.png")
    if error:
        pipeline[1].side_effect = error
        expected = HTTPException if isinstance(error, ValueError) else RuntimeError
        with pytest.raises(expected):
            asyncio.run(main.upload_image(file))
    else:
        asyncio.run(main.upload_image(file))
        pipeline[1].assert_called_once_with(b"bytes", "application/octet-stream")
    assert file.file.closed


def test_real_invalid_image_skips_analysis(client, monkeypatch):
    analysis = Mock()
    monkeypatch.setattr(main, "analyze_image", analysis)
    response = client.post("/upload", files={"file": ("bad.png", b"invalid", "image/png")})
    assert response.json() == {"status": "fail", "reasons": ["invalid_image"],
                               "scores": {"min_side": None, "sharpness": None}, "facts": None}
    analysis.assert_not_called()


def test_upload_stage_logs_are_persisted_with_same_request_id(client, tmp_path):
    path = tmp_path / "app.log"
    handler = create_file_handler(path)
    logger.addHandler(handler)
    try:
        response = client.post("/upload", headers={"X-Request-ID": "upload-test"},
                               files={"file": ("bad.png", b"invalid", "image/png")})
    finally:
        logger.removeHandler(handler)
        handler.close()
    logs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert response.status_code == 200
    assert [r["event"] for r in logs] == [
        "upload.received", "validation.started", "validation.completed",
        "llm.skipped", "decision.completed", "request.completed",
    ]
    assert all(r["request_id"] == "upload-test" for r in logs)
    assert logs[-2]["status"] == "fail"
    assert logs[-2]["reasons"] == ["invalid_image"]

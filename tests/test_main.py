import asyncio
import json
import threading
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, Mock
from concurrent.futures.process import BrokenProcessPool
from contextlib import asynccontextmanager

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

import main
import validate_image as validation_module
from schemas import ProductAnalysis
from request_logging import create_file_handler, logger, request_id_context
from validation_pool import ValidationBusy
from analyse_image import GeminiBusy


@pytest.fixture
def gemini_resources(monkeypatch):
    client = object()

    @asynccontextmanager
    async def shared_client(attempt_timeout):
        yield client

    monkeypatch.setattr(main, "shared_gemini_client", shared_client)
    return {"client": client, "semaphore": asyncio.Semaphore(5)}


@pytest.fixture
def validation_service(monkeypatch):
    # Route tests isolate the process boundary. The real pool has its own tests.
    async def validate(image_bytes, request_id):
        return await asyncio.to_thread(validation_module.validate_image, image_bytes)

    service = SimpleNamespace(validate=AsyncMock(side_effect=validate), close=AsyncMock())
    monkeypatch.setattr(main, "ValidationPool", Mock(return_value=service))
    return service


def request_for(service, gemini):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(validation_pool=service, gemini=gemini)))


@pytest.fixture
def client(validation_service, gemini_resources):
    with TestClient(main.app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def pipeline(monkeypatch, facts_data):
    validation = Mock(return_value={
        "status": "pass", "reasons": [],
        "scores": {"min_side": 1000, "sharpness": 60.5},
    })
    analysis = AsyncMock(return_value=ProductAnalysis(**facts_data))
    monkeypatch.setattr(validation_module, "validate_image", validation)
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
    assert analysis.await_args.args == (b"bytes", "image/png")
    assert analysis.await_args.kwargs["client"] is not None


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
def test_upload_file_is_closed_even_on_error(pipeline, validation_service, gemini_resources, error):
    file = UploadFile(file=BytesIO(b"bytes"), filename="test.png")
    if error:
        pipeline[1].side_effect = error
        expected = HTTPException if isinstance(error, ValueError) else RuntimeError
        with pytest.raises(expected):
            asyncio.run(main.upload_image(request_for(validation_service, gemini_resources), file))
    else:
        asyncio.run(main.upload_image(request_for(validation_service, gemini_resources), file))
        assert pipeline[1].await_args.args == (b"bytes", "application/octet-stream")
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


def test_gemini_is_awaited_with_request_context(pipeline, validation_service, gemini_resources):
    async def run():
        event_loop_thread = threading.get_ident()
        result = pipeline[1].return_value

        async def network_call(*args, **kwargs):
            assert threading.get_ident() == event_loop_thread
            await asyncio.sleep(0)
            assert request_id_context.get() == "threaded-upload"
            return result

        pipeline[1].side_effect = network_call
        token = request_id_context.set("threaded-upload")
        file = UploadFile(file=BytesIO(b"bytes"), filename="test.png")
        try:
            response = await main.upload_image(request_for(validation_service, gemini_resources), file)
            assert response["status"] == "pass"
            assert request_id_context.get() == "threaded-upload"
        finally:
            request_id_context.reset(token)
        assert file.file.closed

    asyncio.run(run())


@pytest.mark.parametrize("error", [ValidationBusy("full"), BrokenProcessPool("broken")])
def test_unavailable_validation_returns_503_and_skips_gemini(client, pipeline, validation_service, error):
    validation_service.validate.side_effect = error
    response = client.post("/upload", files={"file": ("test.png", b"bytes", "image/png")})
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "1"
    assert response.headers["X-Request-ID"]
    pipeline[1].assert_not_called()


def test_lifespan_configures_and_closes_pool(monkeypatch, validation_service, gemini_resources):
    monkeypatch.setenv("VALIDATION_WORKERS", "2")
    monkeypatch.setenv("VALIDATION_MAX_WAITING", "10")
    monkeypatch.setenv("VALIDATION_WAIT_TIMEOUT", "1.5")
    with TestClient(main.app) as client:
        assert client.get("/health").status_code == 200
        main.ValidationPool.assert_called_once_with(
            workers=2, max_waiting=10, wait_timeout=1.5, log_queue=ANY, dropped=ANY,
        )
        validation_service.close.assert_not_awaited()
    validation_service.close.assert_awaited_once()


def test_gemini_overload_is_503(client, pipeline):
    pipeline[1].side_effect = GeminiBusy("full")
    response = client.post("/upload", files={"file": ("test.png", b"bytes", "image/png")})
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "1"


def test_lifespan_shares_gemini_limits(monkeypatch, validation_service, gemini_resources):
    monkeypatch.setenv("GEMINI_MAX_CONCURRENT", "2")
    monkeypatch.setenv("GEMINI_MAX_WAITING", "3")
    monkeypatch.setenv("GEMINI_REQUESTS_PER_MINUTE", "60")
    with TestClient(main.app) as client:
        state = client.app.app.state.gemini
        assert state["semaphore"]._value == 2
        assert state["admission"].maxsize == 5
        assert state["pacing"]["interval"] == 1


@pytest.mark.parametrize("key,value", [
    ("GEMINI_MAX_CONCURRENT", "0"), ("GEMINI_MAX_WAITING", "-1"),
    ("GEMINI_REQUESTS_PER_MINUTE", "nan"), ("GEMINI_TOTAL_TIMEOUT", "0"),
    ("GEMINI_RETRY_BACKOFF", "-1"),
])
def test_invalid_gemini_config_closes_validation_pool(monkeypatch, validation_service, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="Gemini configuration"):
        with TestClient(main.app):
            pass
    validation_service.close.assert_awaited_once()

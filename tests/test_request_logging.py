import asyncio
import json
import logging
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from request_logging import (
    JsonFormatter, RequestContextFilter, RequestLoggingMiddleware,
    create_file_handler, logger, request_id_context,
)


@pytest.fixture
def logged_client(monkeypatch, caplog):
    monkeypatch.setattr(logger, "propagate", True)
    caplog.set_level(logging.INFO, logger="image_api")
    api = FastAPI()

    @api.get("/health")
    async def health():
        logger.info("health.checked")
        return {"status": "healthy"}

    @api.get("/error")
    async def error():
        raise RuntimeError("test failure")

    @api.get("/unavailable", status_code=503)
    async def unavailable():
        return {"status": "unavailable"}

    with TestClient(RequestLoggingMiddleware(api), raise_server_exceptions=False) as client:
        yield client


def records(caplog):
    return [r for r in caplog.records if r.name == "image_api"]


def test_generated_id_status_and_latency(logged_client, caplog):
    response = logged_client.get("/health")
    assert response.status_code == 200
    request_id = response.headers["X-Request-ID"]
    UUID(request_id)
    logs = records(caplog)
    assert logs[0].getMessage() == "health.checked"
    assert logs[0].request_id == request_id
    record = logs[-1]
    assert record.request_id == request_id
    assert record.getMessage() == "request.completed"
    assert record.method == "GET"
    assert record.path == "/health"
    assert record.status_code == 200
    assert record.latency_ms >= 0
    assert request_id_context.get() is None


@pytest.mark.parametrize("request_id", ["client-123", "a.b_c-9", "x" * 128])
def test_supplied_id_and_query_exclusion(logged_client, caplog, request_id):
    response = logged_client.get(
        "/health?token=private", headers={"X-Request-ID": request_id},
    )
    assert response.headers["X-Request-ID"] == request_id
    assert records(caplog)[-1].request_id == request_id
    assert "private" not in "".join(JsonFormatter().format(r) for r in records(caplog))


@pytest.mark.parametrize("request_id", ["", "x" * 129, "bad id", "bad/id"])
def test_invalid_id_is_replaced(logged_client, request_id):
    response = logged_client.get("/health", headers={"X-Request-ID": request_id})
    UUID(response.headers["X-Request-ID"])


def test_not_found_is_logged(logged_client, caplog):
    assert logged_client.get("/missing").status_code == 404
    assert records(caplog)[-1].status_code == 404


def test_unhandled_exception_has_id_and_error_log(logged_client, caplog):
    response = logged_client.get("/error")
    assert response.status_code == 500
    record = records(caplog)[-1]
    assert record.request_id == response.headers["X-Request-ID"]
    assert record.getMessage() == "request.failed"
    assert record.status_code == 500
    assert record.levelno == logging.ERROR
    assert "RuntimeError: test failure" in json.loads(JsonFormatter().format(record))["exception"]
    assert request_id_context.get() is None


def test_returned_503_is_logged_at_error_level(logged_client, caplog):
    assert logged_client.get("/unavailable").status_code == 503
    record = records(caplog)[-1]
    assert record.levelno == logging.ERROR
    assert record.getMessage() == "request.completed"
    assert record.exc_info is None


def test_context_filter_adds_id_but_preserves_explicit_id():
    token = request_id_context.set("context-id")
    try:
        record = logging.makeLogRecord({"msg": "event"})
        assert RequestContextFilter().filter(record) is True
        assert record.request_id == "context-id"
        record.request_id = "explicit-id"
        RequestContextFilter().filter(record)
        assert record.request_id == "explicit-id"
    finally:
        request_id_context.reset(token)


def test_json_formatter_fields_and_unicode():
    record = logging.makeLogRecord({
        "name": "test", "msg": "image %s", "args": ("café",),
        "levelname": "INFO", "created": 0,
    })
    data = json.loads(JsonFormatter().format(record))
    assert data == {
        "timestamp": "1970-01-01T00:00:00.000+00:00", "level": "INFO",
        "logger": "test", "event": "image café", "request_id": None,
    }


def test_non_http_scope_passes_through():
    app = AsyncMock()
    receive, send = AsyncMock(), AsyncMock()
    scope = {"type": "websocket"}
    asyncio.run(RequestLoggingMiddleware(app)(scope, receive, send))
    app.assert_awaited_once_with(scope, receive, send)
    assert "state" not in scope


def test_request_context_is_restored_when_logging_raises(monkeypatch):
    def fail_log(*args, **kwargs):
        raise RuntimeError("logging failed")

    monkeypatch.setattr(logger, "log", fail_log)
    token = request_id_context.set("outer-context")
    try:
        scope = {"type": "http", "method": "GET", "path": "/", "headers": []}
        with pytest.raises(RuntimeError, match="logging failed"):
            asyncio.run(RequestLoggingMiddleware(AsyncMock())(scope, AsyncMock(), AsyncMock()))
        assert request_id_context.get() == "outer-context"
    finally:
        request_id_context.reset(token)


def test_concurrent_requests_keep_separate_context(monkeypatch):
    monkeypatch.setattr(logger, "log", lambda *args, **kwargs: None)
    seen = {}

    async def app(scope, receive, send):
        before = request_id_context.get()
        await asyncio.sleep(0)
        seen[scope["path"]] = (before, request_id_context.get(), scope["state"]["request_id"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def run():
        middleware = RequestLoggingMiddleware(app)
        await asyncio.gather(*[
            middleware({"type": "http", "method": "GET", "path": f"/{name}",
                        "headers": [(b"x-request-id", name.encode())]}, AsyncMock(), AsyncMock())
            for name in ["first", "second"]
        ])

    asyncio.run(run())
    assert seen == {"/first": ("first",) * 3, "/second": ("second",) * 3}
    assert request_id_context.get() is None


def test_json_formatter_includes_stage_fields_and_excludes_unknown_fields():
    record = logging.makeLogRecord({
        "msg": "validation.completed", "stage": "validation", "status": "fail",
        "reasons": ["blurry"], "min_side": 1000, "sharpness": 12.5,
        "api_key": "private", "image_bytes": b"private",
    })
    data = json.loads(JsonFormatter().format(record))
    assert data["status"] == "fail"
    assert data["reasons"] == ["blurry"]
    assert data["min_side"] == 1000
    assert data["sharpness"] == 12.5
    assert "private" not in json.dumps(data)


def test_file_handler_persists_json_with_request_context(tmp_path):
    path = tmp_path / "logs" / "app.log"
    handler = create_file_handler(path)
    local_logger = logging.Logger("persistence-test", level=logging.INFO)
    local_logger.addFilter(RequestContextFilter())
    local_logger.addHandler(handler)
    token = request_id_context.set("persisted-request")
    try:
        local_logger.info("decision.completed", extra={
            "stage": "decision", "status": "pass", "reasons": [],
        })
    finally:
        handler.close()
        request_id_context.reset(token)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["request_id"] == "persisted-request"
    assert data["event"] == "decision.completed"
    assert data["status"] == "pass"


def test_file_rotation_keeps_bounded_backups(tmp_path):
    path = tmp_path / "app.log"
    handler = create_file_handler(path, max_bytes=300, backup_count=2)
    local_logger = logging.Logger("rotation-test", level=logging.INFO)
    local_logger.addHandler(handler)
    try:
        for index in range(10):
            local_logger.info("event.%s", index)
    finally:
        handler.close()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["app.log", "app.log.1", "app.log.2"]
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[-1])["event"] == "event.9"
    for log_file in tmp_path.iterdir():
        for line in log_file.read_text(encoding="utf-8").splitlines():
            json.loads(line)

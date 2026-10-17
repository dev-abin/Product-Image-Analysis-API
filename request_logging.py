import json
import logging
import re
from contextvars import ContextVar
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send


request_id_context: ContextVar[str | None] = ContextVar("request_id", default=None)
VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,128}")


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = request_id_context.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "timestamp": datetime.fromtimestamp(
                record.created, timezone.utc
            ).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "request_id": getattr(record, "request_id", request_id_context.get()),
        }
        for field in ("method", "path", "status_code", "latency_ms"):
            if hasattr(record, field):
                data[field] = getattr(record, field)
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


logger = logging.getLogger("image_api")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    logger.addFilter(RequestContextFilter())
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)


class RequestLoggingMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        supplied_id = Headers(scope=scope).get("x-request-id", "")
        request_id = (
            supplied_id if VALID_REQUEST_ID.fullmatch(supplied_id) else str(uuid4())
        )
        scope.setdefault("state", {})["request_id"] = request_id
        token = request_id_context.set(request_id)
        started = perf_counter()
        status_code = 500
        error = None

        async def send_with_request_id(message: Message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        except Exception as exc:
            error = (type(exc), exc, exc.__traceback__)
            raise
        finally:
            try:
                level = (
                    logging.ERROR if error or status_code >= 500 else logging.INFO
                )
                logger.log(
                    level,
                    "request.failed" if error else "request.completed",
                    extra={
                        "request_id": request_id,
                        "method": scope["method"],
                        "path": scope["path"],
                        "status_code": status_code,
                        "latency_ms": round((perf_counter() - started) * 1000, 3),
                    },
                    exc_info=error,
                )
            finally:
                request_id_context.reset(token)

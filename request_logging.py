import json
import logging
import re
from copy import copy
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from multiprocessing import get_context
from pathlib import Path
from queue import Full
from contextvars import ContextVar
from datetime import datetime, timezone
from time import monotonic, perf_counter
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send


request_id_context: ContextVar[str | None] = ContextVar("request_id", default=None)
VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,128}")
LOG_FILE = Path(__file__).resolve().parent / "logs" / "app.log"


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
        # Only explicitly approved fields become part of the persisted log.
        for field in (
            "method", "path", "status_code", "latency_ms", "stage",
            "status", "reasons", "min_side", "sharpness", "attempt",
            "model", "error_type", "provider_status", "size_bytes", "dropped_count",
        ):
            if hasattr(record, field):
                data[field] = getattr(record, field)
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        elif record.exc_text:
            data["exception"] = record.exc_text
        return json.dumps(data, ensure_ascii=False)


def create_file_handler(
    path: Path = LOG_FILE, max_bytes: int = 5 * 1024 * 1024, backup_count: int = 3,
) -> RotatingFileHandler:
    """Persist JSON logs, keeping the active file and three bounded backups."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8",
    )
    handler.setFormatter(JsonFormatter())
    return handler


logger = logging.getLogger("image_api")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    logger.addFilter(RequestContextFilter())
    # Output handlers belong to the app lifespan, never to module imports.
    logger.addHandler(logging.NullHandler())


class LogQueueHandler(QueueHandler):
    def __init__(self, queue, dropped):
        super().__init__(queue)
        self.dropped = dropped

    def prepare(self, record):
        # Preserve exceptions as a separate JSON field and make them picklable.
        prepared = copy(record)
        prepared.msg = record.getMessage()
        prepared.args = None
        if record.exc_info:
            prepared.exc_text = logging.Formatter().formatException(record.exc_info)
        prepared.exc_info = None
        return prepared

    def enqueue(self, record):
        try:
            self.queue.put_nowait(record)
        except Full:
            # No synchronous stderr fallback on a request-processing thread.
            with self.dropped.get_lock():
                self.dropped.value += 1


class LogQueueListener(QueueListener):
    def __init__(self, queue, dropped, *handlers):
        super().__init__(queue, *handlers, respect_handler_level=True)
        self.dropped = dropped
        self._reported = 0
        self._last_report = 0.0

    def report_drops(self, force=False):
        total = self.dropped.value
        if total > self._reported and (force or monotonic() - self._last_report >= 5):
            record = logging.makeLogRecord({
                "name": "image_api", "levelno": logging.WARNING, "levelname": "WARNING",
                "msg": "logging.dropped", "request_id": None,
                "dropped_count": total - self._reported,
            })
            # Write directly from the listener; don't enqueue a new warning.
            super().handle(record)
            self._reported = total
            self._last_report = monotonic()

    def handle(self, record):
        super().handle(record)
        self.report_drops()

    def enqueue_sentinel(self):
        # stop() runs off the event loop; wait for room if the queue is full.
        self.queue.put(self._sentinel)

    def stop(self):
        super().stop()
        self.report_drops(force=True)


def start_queue_logging(capacity=1000, handlers=None):
    """Start one bounded queue/listener shared by the API and CPU workers."""
    if capacity < 1:
        raise ValueError("Log queue capacity must be positive")
    if getattr(logger, "queue_runtime", None) is not None:
        raise RuntimeError("Queue logging is already running")
    previous = logger.handlers[:]
    owned = []
    if handlers is None:
        console = logging.StreamHandler()
        console.setFormatter(JsonFormatter())
        owned = [console, create_file_handler()]
        handlers = owned + [h for h in previous if not isinstance(h, logging.NullHandler)]
    context = get_context("spawn")
    queue = context.Queue(maxsize=capacity)
    dropped = context.Value("Q", 0)
    handler = LogQueueHandler(queue, dropped)
    listener = LogQueueListener(queue, dropped, *handlers)
    runtime = {"queue": queue, "dropped": dropped, "listener": listener,
               "handler": handler, "previous": previous, "owned": owned}
    listener.start()
    logger.handlers = [handler]
    logger.queue_runtime = runtime
    return runtime


def stop_queue_logging(runtime):
    """Stop producers first, drain accepted records, then close output files."""
    logger.handlers = runtime["previous"]
    try:
        runtime["listener"].stop()
    finally:
        for handler in runtime["owned"]:
            handler.close()
        runtime["handler"].close()
        runtime["queue"].close()
        runtime["queue"].join_thread()
        logger.queue_runtime = None


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

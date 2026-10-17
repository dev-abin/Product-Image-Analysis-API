import asyncio
import json
import unittest
from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient

from request_logging import (
    JsonFormatter,
    RequestLoggingMiddleware,
    logger,
    request_id_context,
)


class RequestLoggingTests(unittest.TestCase):
    def setUp(self):
        api = FastAPI()

        @api.get("/health")
        async def health():
            await asyncio.sleep(0.01)
            logger.info("health.checked")
            return {"status": "healthy"}

        @api.get("/error")
        async def error():
            raise RuntimeError("test failure")

        self.client = TestClient(
            RequestLoggingMiddleware(api), raise_server_exceptions=False
        )

    def request_with_logs(self, path, **kwargs):
        with self.assertLogs(logger, level="INFO") as captured:
            response = self.client.get(path, **kwargs)
        logs = [json.loads(JsonFormatter().format(r)) for r in captured.records]
        return response, logs

    def test_generated_id_status_and_latency(self):
        response, logs = self.request_with_logs("/health")
        self.assertEqual(response.status_code, 200)
        request_id = response.headers["X-Request-ID"]
        UUID(request_id)
        self.assertEqual(logs[0]["event"], "health.checked")
        self.assertEqual(logs[0]["request_id"], request_id)
        record = logs[-1]
        self.assertEqual(record["request_id"], request_id)
        self.assertEqual(record["event"], "request.completed")
        self.assertEqual(record["method"], "GET")
        self.assertEqual(record["path"], "/health")
        self.assertEqual(record["status_code"], 200)
        self.assertGreaterEqual(record["latency_ms"], 10)
        self.assertIsNone(request_id_context.get())

    def test_supplied_id_and_query_is_not_logged(self):
        response, logs = self.request_with_logs(
            "/health?token=private", headers={"X-Request-ID": "client-123"}
        )
        self.assertEqual(response.headers["X-Request-ID"], "client-123")
        self.assertEqual(logs[-1]["request_id"], "client-123")
        self.assertNotIn("private", json.dumps(logs))

    def test_invalid_id_is_replaced(self):
        response, _ = self.request_with_logs(
            "/health", headers={"X-Request-ID": "x" * 129}
        )
        UUID(response.headers["X-Request-ID"])

    def test_not_found_is_logged(self):
        response, logs = self.request_with_logs("/missing")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(logs[-1]["status_code"], 404)

    def test_unhandled_exception_has_id_and_error_log(self):
        response, logs = self.request_with_logs("/error")
        self.assertEqual(response.status_code, 500)
        record = logs[-1]
        self.assertEqual(record["request_id"], response.headers["X-Request-ID"])
        self.assertEqual(record["event"], "request.failed")
        self.assertEqual(record["status_code"], 500)
        self.assertEqual(record["level"], "ERROR")
        self.assertIn("RuntimeError: test failure", record["exception"])


if __name__ == "__main__":
    unittest.main()

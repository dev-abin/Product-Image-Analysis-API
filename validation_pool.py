"""Bound CPU validation work without blocking the API event loop."""

import asyncio
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

import cv2

from request_logging import (
    LogQueueHandler, logger, request_id_context, start_queue_logging, stop_queue_logging,
)
from validate_image import validate_image


class ValidationBusy(Exception):
    """The validation queue is full or its waiting deadline expired."""


def initialize_worker(log_queue, dropped):
    # Only the API process writes/rotates app.log. Workers forward log records.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    logger.addHandler(LogQueueHandler(log_queue, dropped))
    # Avoid each process starting another large set of OpenCV CPU threads.
    cv2.setNumThreads(1)


def validate_in_worker(image_bytes: bytes, request_id: str | None) -> dict:
    # ContextVars do not automatically propagate across process boundaries.
    token = request_id_context.set(request_id)
    try:
        return validate_image(image_bytes)
    finally:
        request_id_context.reset(token)


class ValidationPool:
    def __init__(self, workers=4, max_waiting=96, wait_timeout=3.0, log_queue=None, dropped=None):
        if workers < 1 or max_waiting < 0 or not math.isfinite(wait_timeout) or wait_timeout <= 0:
            raise ValueError("Invalid validation pool configuration")
        self.workers = workers
        self.capacity = workers + max_waiting
        self.wait_timeout = wait_timeout
        self._slots = asyncio.Semaphore(workers)
        self._admitted = 0
        self._closing = False
        context = multiprocessing.get_context("spawn")
        self._logging = start_queue_logging() if log_queue is None else None
        if self._logging is not None:
            log_queue, dropped = self._logging["queue"], self._logging["dropped"]
        try:
            self._executor = ProcessPoolExecutor(
                max_workers=workers, mp_context=context,
                initializer=initialize_worker, initargs=(log_queue, dropped),
            )
        except Exception:
            if self._logging is not None:
                stop_queue_logging(self._logging)
            raise

    async def validate(self, image_bytes: bytes, request_id: str | None) -> dict:
        # No await between checking and reserving admission on this event loop.
        if self._closing or self._admitted >= self.capacity:
            raise ValidationBusy("Validation capacity is full; retry later")
        self._admitted += 1
        acquired = submitted = False
        started = perf_counter()
        try:
            try:
                await asyncio.wait_for(self._slots.acquire(), timeout=self.wait_timeout)
            except TimeoutError as exc:
                raise ValidationBusy("Timed out waiting for image validation") from exc
            acquired = True
            if self._closing:
                raise ValidationBusy("Validation service is shutting down")
            logger.info("validation.dispatched", extra={
                "stage": "validation_queue",
                "latency_ms": round((perf_counter() - started) * 1000, 3),
            })
            future = asyncio.get_running_loop().run_in_executor(
                self._executor, validate_in_worker, image_bytes, request_id,
            )
            # A disconnected caller must not free capacity while CPU work runs.
            future.add_done_callback(self._finished)
            submitted = True
            return await asyncio.shield(future)
        finally:
            if not submitted:
                self._admitted -= 1
                if acquired:
                    self._slots.release()

    def _finished(self, future):
        self._admitted -= 1
        self._slots.release()
        if not future.cancelled():
            future.exception()  # Consume failures even if the caller disconnected.

    async def close(self):
        self._closing = True
        # Finish worker producers before their logging listener is stopped.
        try:
            await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)
        finally:
            if self._logging is not None:
                await asyncio.to_thread(stop_queue_logging, self._logging)

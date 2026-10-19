import asyncio
import json
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import validation_pool as module
from request_logging import create_file_handler, logger, request_id_context


class ManualExecutor:
    """Controlled process futures make admission/cancellation tests deterministic."""
    def __init__(self):
        self.jobs = []
        self.shutdown = Mock()

    def submit(self, function, *args):
        future = Future()
        future.set_running_or_notify_cancel()
        self.jobs.append((future, function, args))
        return future


@pytest.fixture
def resources(monkeypatch):
    executor = ManualExecutor()
    queue, dropped = Mock(), Mock()
    context = SimpleNamespace()
    monkeypatch.setattr(module.multiprocessing, "get_context", Mock(return_value=context))
    factory = Mock(return_value=executor)
    monkeypatch.setattr(module, "ProcessPoolExecutor", factory)
    runtime = {"queue": queue, "dropped": dropped}
    monkeypatch.setattr(module, "start_queue_logging", Mock(return_value=runtime))
    stop = Mock()
    monkeypatch.setattr(module, "stop_queue_logging", stop)
    return executor, factory, runtime, stop


async def until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0)
    await asyncio.wait_for(wait(), timeout=2)


def test_bounds_active_work_and_waiting_queue(resources):
    async def run():
        executor, factory, _, _ = resources
        pool = module.ValidationPool(workers=1, max_waiting=1)
        try:
            first = asyncio.create_task(pool.validate(b"first", "request-1"))
            await until(lambda: len(executor.jobs) == 1)
            second = asyncio.create_task(pool.validate(b"second", "request-2"))
            await until(lambda: pool._admitted == 2)
            with pytest.raises(module.ValidationBusy, match="capacity"):
                await pool.validate(b"third", "request-3")
            assert len(executor.jobs) == 1
            executor.jobs[0][0].set_result({"status": "pass"})
            assert await first == {"status": "pass"}
            await until(lambda: len(executor.jobs) == 2)
            assert executor.jobs[1][1:] == (module.validate_in_worker, (b"second", "request-2"))
            executor.jobs[1][0].set_result({"status": "fail"})
            assert await second == {"status": "fail"}
            assert pool._admitted == 0
            assert factory.call_args.kwargs["max_workers"] == 1
        finally:
            await pool.close()
    asyncio.run(run())


def test_queue_wait_timeout_releases_admission(resources):
    async def run():
        executor = resources[0]
        pool = module.ValidationPool(workers=1, max_waiting=1, wait_timeout=0.01)
        try:
            first = asyncio.create_task(pool.validate(b"first", None))
            await until(lambda: len(executor.jobs) == 1)
            with pytest.raises(module.ValidationBusy, match="Timed out"):
                await pool.validate(b"waiting", None)
            assert pool._admitted == 1
            assert len(executor.jobs) == 1
            executor.jobs[0][0].set_result({})
            await first
            assert pool._admitted == 0
        finally:
            await pool.close()
    asyncio.run(run())


def test_100_upload_burst_uses_four_slots_and_rejects_overflow(resources):
    async def run():
        executor = resources[0]
        pool = module.ValidationPool(workers=4, max_waiting=96, wait_timeout=5)
        tasks = [asyncio.create_task(pool.validate(b"image", f"request-{i}")) for i in range(100)]
        try:
            await until(lambda: pool._admitted == 100 and len(executor.jobs) == 4)
            with pytest.raises(module.ValidationBusy):
                await pool.validate(b"overflow", "request-101")
            for end in range(4, 101, 4):
                await until(lambda: len(executor.jobs) == end)
                for future, _, _ in executor.jobs[end - 4:end]:
                    future.set_result({"status": "pass"})
            results = await asyncio.gather(*tasks)
            assert len(results) == 100
            assert pool._admitted == 0
        finally:
            await pool.close()
    asyncio.run(run())


def test_cancelling_running_request_keeps_slot_until_process_finishes(resources):
    async def run():
        executor = resources[0]
        pool = module.ValidationPool(workers=1, max_waiting=0)
        try:
            task = asyncio.create_task(pool.validate(b"running", None))
            await until(lambda: len(executor.jobs) == 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            with pytest.raises(module.ValidationBusy):
                await pool.validate(b"excess", None)
            assert not executor.jobs[0][0].cancelled()
            executor.jobs[0][0].set_exception(RuntimeError("worker failed after disconnect"))
            await until(lambda: pool._admitted == 0)
            next_task = asyncio.create_task(pool.validate(b"next", None))
            await until(lambda: len(executor.jobs) == 2)
            executor.jobs[1][0].set_result({})
            await next_task
        finally:
            await pool.close()
    asyncio.run(run())


def test_cancelling_waiting_request_releases_admission(resources):
    async def run():
        executor = resources[0]
        pool = module.ValidationPool(workers=1, max_waiting=1)
        try:
            first = asyncio.create_task(pool.validate(b"first", None))
            await until(lambda: len(executor.jobs) == 1)
            waiting = asyncio.create_task(pool.validate(b"waiting", None))
            await until(lambda: pool._admitted == 2)
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
            assert pool._admitted == 1
            executor.jobs[0][0].set_result({})
            await first
        finally:
            await pool.close()
    asyncio.run(run())


def test_submission_failure_releases_capacity(resources, monkeypatch):
    async def run():
        executor = resources[0]
        monkeypatch.setattr(executor, "submit", Mock(side_effect=RuntimeError("submit failed")))
        pool = module.ValidationPool(workers=1, max_waiting=0)
        try:
            with pytest.raises(RuntimeError, match="submit failed"):
                await pool.validate(b"image", None)
            assert pool._admitted == 0
            assert pool._slots._value == 1
        finally:
            await pool.close()
    asyncio.run(run())


def test_worker_failure_releases_capacity(resources):
    async def run():
        executor = resources[0]
        pool = module.ValidationPool(workers=1, max_waiting=0)
        try:
            task = asyncio.create_task(pool.validate(b"image", None))
            await until(lambda: len(executor.jobs) == 1)
            executor.jobs[0][0].set_exception(ValueError("worker failed"))
            with pytest.raises(ValueError, match="worker failed"):
                await task
            assert pool._admitted == 0
        finally:
            await pool.close()
    asyncio.run(run())


def test_shutdown_rejects_new_work_and_closes_resources(resources):
    async def run():
        executor, _, runtime, stop = resources
        pool = module.ValidationPool()
        await pool.close()
        with pytest.raises(module.ValidationBusy):
            await pool.validate(b"image", None)
        executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
        stop.assert_called_once_with(runtime)
    asyncio.run(run())


@pytest.mark.parametrize("options", [
    {"workers": 0}, {"max_waiting": -1}, {"wait_timeout": 0},
    {"wait_timeout": float("nan")}, {"wait_timeout": float("inf")},
])
def test_invalid_configuration(options):
    with pytest.raises(ValueError, match="configuration"):
        module.ValidationPool(**options)


def test_worker_sets_and_restores_request_context(monkeypatch):
    def validate(payload):
        assert payload == b"image"
        assert request_id_context.get() == "worker-request"
        return {"status": "pass"}

    monkeypatch.setattr(module, "validate_image", validate)
    token = request_id_context.set("outer")
    try:
        assert module.validate_in_worker(b"image", "worker-request") == {"status": "pass"}
        assert request_id_context.get() == "outer"
    finally:
        request_id_context.reset(token)


def test_worker_initializer_forwards_logs_and_limits_opencv_threads(monkeypatch):
    old_handler = Mock()
    monkeypatch.setattr(logger, "handlers", [old_handler])
    queue_handler = Mock()
    monkeypatch.setattr(module, "LogQueueHandler", Mock(return_value=queue_handler))
    monkeypatch.setattr(module.cv2, "setNumThreads", Mock())
    queue, dropped = Mock(), Mock()
    module.initialize_worker(queue, dropped)
    old_handler.close.assert_called_once()
    assert logger.handlers == [queue_handler]
    module.LogQueueHandler.assert_called_once_with(queue, dropped)
    module.cv2.setNumThreads.assert_called_once_with(1)


def test_real_spawned_process_validates_and_persists_correlated_logs(tmp_path):
    path = tmp_path / "process.log"
    handler = create_file_handler(path)
    logger.addHandler(handler)

    async def run():
        pool = module.ValidationPool(workers=1, max_waiting=0, wait_timeout=5)
        try:
            result = await asyncio.wait_for(pool.validate(b"invalid image", "process-test"), timeout=30)
            assert result["reasons"] == ["invalid_image"]
        finally:
            await pool.close()

    try:
        asyncio.run(run())
    finally:
        logger.removeHandler(handler)
        handler.close()
    logs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    completed = [r for r in logs if r["event"] == "validation.completed"]
    assert len(completed) == 1
    assert completed[0]["request_id"] == "process-test"


def test_external_logging_stays_owned_by_application(resources):
    async def run():
        runtime = resources[2]
        pool = module.ValidationPool(log_queue=runtime["queue"], dropped=runtime["dropped"])
        await pool.close()
        module.start_queue_logging.assert_not_called()
        module.stop_queue_logging.assert_not_called()
    asyncio.run(run())


def test_executor_start_failure_closes_owned_logging(resources):
    resources[1].side_effect = RuntimeError("executor initialization failed")
    with pytest.raises(RuntimeError, match="executor initialization"):
        module.ValidationPool()
    resources[3].assert_called_once_with(resources[2])

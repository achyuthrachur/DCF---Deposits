"""Background execution helpers for long-running analysis jobs."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Callable, Optional


@dataclass
class ProgressEvent:
    step: int
    total: int
    message: str
    timestamp: float


class AnalysisRunner:
    """Manage background execution of ALM engine runs with progress streaming."""

    def __init__(self, *, max_workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="analysis-runner")
        self._future: Optional[Future] = None
        self._progress: "Queue[ProgressEvent]" = Queue()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ status
    @property
    def running(self) -> bool:
        with self._lock:
            return self._future is not None and not self._future.done()

    @property
    def done(self) -> bool:
        with self._lock:
            return self._future is not None and self._future.done()

    # ------------------------------------------------------------------ control
    def start(self, task: Callable[[Callable[[int, int, str], None]], object]) -> None:
        """Submit a background task that accepts a progress reporter callable."""
        with self._lock:
            if self._future is not None and not self._future.done():
                raise RuntimeError("AnalysisRunner is already executing a task.")
            self._progress.queue.clear()

            def _invoke() -> object:
                return task(self._enqueue_progress)

            self._future = self._executor.submit(_invoke)

    def cancel(self) -> None:
        with self._lock:
            if self._future and not self._future.done():
                self._future.cancel()

    # ------------------------------------------------------------------ progress
    def _enqueue_progress(self, step: int, total: int, message: str) -> None:
        self._progress.put(ProgressEvent(step=step, total=total, message=message, timestamp=time.time()))

    def drain_progress(self) -> list[ProgressEvent]:
        updates: list[ProgressEvent] = []
        while True:
            try:
                updates.append(self._progress.get_nowait())
            except Empty:
                break
        return updates

    # ------------------------------------------------------------------ results
    def result(self, timeout: Optional[float] = None) -> object:
        with self._lock:
            if self._future is None:
                raise RuntimeError("AnalysisRunner has not started a task.")
            return self._future.result(timeout=timeout)

    def exception(self) -> Optional[BaseException]:
        with self._lock:
            if self._future is None:
                return None
            return self._future.exception()

    def reset(self) -> None:
        with self._lock:
            self._future = None
            self._progress.queue.clear()

    # ------------------------------------------------------------------- cleanup
    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait)

"""Batch scheduler — orchestrates read batching and dispatching.

Manages a queue of reads, batches them efficiently for GPU processing,
and dispatches batches to GPU/CPU workers with backpressure.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from obs.log import log


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class SchedulerConfig:
    """Scheduler tuning parameters."""
    batch_size: int = 4096          # reads per GPU batch
    max_queue_size: int = 32        # max pending batches
    read_len: int = 300             # uniform read length after padding
    cpu_fallback: bool = True       # use CPU worker when GPU is saturated
    num_cpu_workers: int = 2        # number of CPU worker threads


@dataclass
class Batch:
    """A batch of reads with metadata."""
    batch_id: int
    reads: List[str]
    ref_name: str
    read_len: int


@dataclass
class BatchResult:
    """Result from processing a batch."""
    batch_id: int
    scores: 'np.ndarray'
    read_start: Optional['np.ndarray'] = None
    read_end: Optional['np.ndarray'] = None
    ref_start: Optional['np.ndarray'] = None
    ref_end: Optional['np.ndarray'] = None
    elapsed_ms: float = 0.0
    worker_type: str = "unknown"  # "gpu" or "cpu"


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
class Scheduler:
    """Read batching and dispatch scheduler.

    Accepts reads from a FASTQ parser stream, groups them into
    fixed-size batches, and dispatches to GPU worker (primary)
    or CPU workers (overflow).
    """

    def __init__(self, config: Optional[SchedulerConfig] = None):
        self.config = config or SchedulerConfig()
        self._batch_counter = 0
        self._pending: queue.Queue[Batch] = queue.Queue(
            maxsize=self.config.max_queue_size
        )
        self._results: queue.Queue[BatchResult] = queue.Queue()
        self._running = False
        self._gpu_thread: Optional[threading.Thread] = None
        self._cpu_threads: List[threading.Thread] = []
        self._gpu_handler: Optional[Callable] = None
        self._cpu_handler: Optional[Callable] = None

    # ------------------------------------------------------------------
    # Handlers (set externally by Manager)
    # ------------------------------------------------------------------
    def set_gpu_handler(self, handler: Callable[[Batch], BatchResult]):
        """Set the GPU batch processing function."""
        self._gpu_handler = handler

    def set_cpu_handler(self, handler: Callable[[Batch], BatchResult]):
        """Set the CPU batch processing function."""
        self._cpu_handler = handler

    # ------------------------------------------------------------------
    # Feeding reads
    # ------------------------------------------------------------------
    def feed(self, reads: Iterator[str], ref_name: str = "unknown") -> int:
        """Feed reads from an iterator, grouping into batches.

        Returns total number of reads processed.
        """
        total = 0
        buf: List[str] = []

        for read in reads:
            buf.append(read.strip())
            total += 1

            if len(buf) >= self.config.batch_size:
                self._enqueue_batch(buf, ref_name)
                buf = []

        # Flush remaining
        if buf:
            self._enqueue_batch(buf, ref_name)

        log("scheduler_feed_done", total_reads=total, ref=ref_name)
        return total

    def feed_list(self, reads: List[str], ref_name: str = "unknown"):
        """Feed a pre-collected list of reads."""
        for i in range(0, len(reads), self.config.batch_size):
            batch_reads = reads[i:i + self.config.batch_size]
            self._enqueue_batch(batch_reads, ref_name)

    def _enqueue_batch(self, reads: List[str], ref_name: str):
        """Enqueue a batch, blocking if queue is full (backpressure)."""
        batch = Batch(
            batch_id=self._batch_counter,
            reads=reads,
            ref_name=ref_name,
            read_len=self.config.read_len,
        )
        self._batch_counter += 1
        self._pending.put(batch)  # blocks if queue full
        log("batch_enqueued",
            batch_id=batch.batch_id,
            size=len(reads),
            queue_depth=self._pending.qsize(),
        )

    # ------------------------------------------------------------------
    # Worker threads
    # ------------------------------------------------------------------
    def start(self):
        """Start worker threads."""
        self._running = True

        if self._gpu_handler:
            self._gpu_thread = threading.Thread(
                target=self._worker_loop,
                args=("gpu", self._gpu_handler),
                daemon=True,
                name="gpu-worker",
            )
            self._gpu_thread.start()

        if self.config.cpu_fallback and self._cpu_handler:
            for i in range(self.config.num_cpu_workers):
                t = threading.Thread(
                    target=self._worker_loop,
                    args=(f"cpu-{i}", self._cpu_handler),
                    daemon=True,
                    name=f"cpu-worker-{i}",
                )
                t.start()
                self._cpu_threads.append(t)

        log("scheduler_started",
            gpu_workers=1 if self._gpu_thread else 0,
            cpu_workers=len(self._cpu_threads),
        )

    def stop(self, timeout: float = 10.0):
        """Signal workers to stop and wait for completion."""
        # Put sentinel values first (while _running is still True so workers
        # can dequeue them)
        num_workers = (1 if self._gpu_thread else 0) + len(self._cpu_threads)
        for _ in range(num_workers):
            self._pending.put(None)  # type: ignore

        if self._gpu_thread:
            self._gpu_thread.join(timeout=timeout)
        for t in self._cpu_threads:
            t.join(timeout=timeout)

        self._running = False

        log("scheduler_stopped")

    def _worker_loop(self, worker_name: str, handler: Callable):
        """Main worker loop: dequeue batch → process → enqueue result."""
        log("worker_started", worker=worker_name)

        while self._running:
            try:
                batch = self._pending.get(timeout=1.0)
            except queue.Empty:
                continue

            if batch is None:  # sentinel
                break

            try:
                result = handler(batch)
                self._results.put(result)
            except Exception as e:
                log("worker_error", worker=worker_name, error=str(e))

        log("worker_stopped", worker=worker_name)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def results(self) -> Iterator[BatchResult]:
        """Yield results as they become available."""
        while True:
            try:
                yield self._results.get(timeout=0.5)
            except queue.Empty:
                if not self._running:
                    # Drain remaining
                    try:
                        yield self._results.get_nowait()
                    except queue.Empty:
                        break


# ---------------------------------------------------------------------------
# Convenience: batch iterator for streaming reads
# ---------------------------------------------------------------------------
def batch_reads(
    reads: Iterator[str],
    batch_size: int,
    ref_name: str = "unknown",
) -> Iterator[Batch]:
    """Generator that yields batches from a read stream.

    Useful for direct use without the full Scheduler.
    """
    batch_id = 0
    buf: List[str] = []
    for read in reads:
        buf.append(read.strip())
        if len(buf) >= batch_size:
            yield Batch(batch_id=batch_id, reads=buf, ref_name=ref_name, read_len=300)
            batch_id += 1
            buf = []
    if buf:
        yield Batch(batch_id=batch_id, reads=buf, ref_name=ref_name, read_len=300)

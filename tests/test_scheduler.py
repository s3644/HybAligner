"""Tests for runtime scheduler module."""

import pytest
import time
import numpy as np
from runtime.scheduler import (
    Scheduler, SchedulerConfig, Batch, BatchResult, batch_reads,
)


class TestSchedulerConfig:
    def test_defaults(self):
        cfg = SchedulerConfig()
        assert cfg.batch_size == 4096
        assert cfg.max_queue_size == 32
        assert cfg.cpu_fallback is True
        assert cfg.num_cpu_workers == 2

    def test_custom(self):
        cfg = SchedulerConfig(batch_size=100, max_queue_size=5)
        assert cfg.batch_size == 100
        assert cfg.max_queue_size == 5


class TestBatch:
    def test_create(self):
        b = Batch(batch_id=0, reads=["ACGT", "TGCA"], ref_name="chr1", read_len=4)
        assert b.batch_id == 0
        assert len(b.reads) == 2
        assert b.ref_name == "chr1"


class TestBatchResult:
    def test_create(self):
        scores = np.array([1.0, 2.0], dtype=np.float32)
        br = BatchResult(
            batch_id=0, scores=scores, elapsed_ms=10.0, worker_type="gpu",
        )
        assert br.batch_id == 0
        assert len(br.scores) == 2
        assert br.elapsed_ms == 10.0

    def test_with_bounds(self):
        scores = np.array([5.0], dtype=np.float32)
        rs = np.array([0], dtype=np.int32)
        re = np.array([10], dtype=np.int32)
        fs = np.array([100], dtype=np.int32)
        fe = np.array([110], dtype=np.int32)
        br = BatchResult(
            batch_id=1, scores=scores,
            read_start=rs, read_end=re,
            ref_start=fs, ref_end=fe,
            worker_type="gpu",
        )
        assert br.read_start[0] == 0
        assert br.ref_start[0] == 100


class TestBatchReads:
    def test_batches(self):
        reads = iter(["A" * 10] * 25)
        batches = list(batch_reads(reads, batch_size=10))
        assert len(batches) == 3  # 10 + 10 + 5
        assert batches[0].reads == ["A" * 10] * 10
        assert batches[1].reads == ["A" * 10] * 10
        assert batches[2].reads == ["A" * 10] * 5

    def test_empty(self):
        batches = list(batch_reads(iter([]), batch_size=10))
        assert len(batches) == 0

    def test_exact_batch(self):
        reads = iter(["A"] * 10)
        batches = list(batch_reads(reads, batch_size=10))
        assert len(batches) == 1


class TestSchedulerIntegration:
    def test_feed_and_drain(self):
        """Integration test: feed reads, process via handlers, collect results."""

        def handler(batch: Batch) -> BatchResult:
            scores = np.array([len(r) for r in batch.reads], dtype=np.float32)
            return BatchResult(
                batch_id=batch.batch_id,
                scores=scores,
                elapsed_ms=5.0,
                worker_type="test",
            )

        config = SchedulerConfig(batch_size=5, cpu_fallback=False)
        scheduler = Scheduler(config)
        scheduler.set_gpu_handler(handler)
        scheduler.start()

        # Feed 12 reads → 3 batches (5+5+2)
        reads = ["A" * 10] * 12
        scheduler.feed_list(reads)

        # Stop and wait
        time.sleep(0.5)
        scheduler.stop()

        # Collect results
        results = list(scheduler.results())
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        total_reads = sum(len(r.scores) for r in results)
        assert total_reads == 12

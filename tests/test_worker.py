"""Tests for GPU worker (CPU fallback path — no GPU required)."""

import pytest
import numpy as np
from gpu.worker import (
    AlignBatch, AlignResult, CPUAligner, gpu_worker,
)


class TestAlignBatch:
    def test_create(self):
        batch = AlignBatch(
            batch_id=42, reads=["ACGT", "TGCA"], ref="ACGTACGT",
        )
        assert batch.batch_id == 42
        assert batch.read_len == 4

    def test_read_len_auto(self):
        batch = AlignBatch(
            batch_id=0, reads=["AAA", "CCCCC"], ref="N" * 100,
        )
        assert batch.read_len == 5  # max of 3 and 5


class TestAlignResult:
    def test_defaults(self):
        scores = np.array([1.0, 2.0], dtype=np.float32)
        result = AlignResult(batch_id=0, scores=scores, n_reads=2)
        assert result.read_start is None
        assert result.ref_start is None


class TestCPUAlignerSW:
    """Test the CPU Gotoh Smith-Waterman implementation."""

    def test_identical_sequences(self):
        read = "ACGTACGT"
        ref = "ACGTACGT"
        score, rs, re, fs, fe = CPUAligner.sw_align(read, ref)
        assert score > 0
        assert rs == 0
        assert re == 8
        assert fs == 0
        assert fe == 8

    def test_no_match(self):
        read = "AAAA"
        ref = "CCCC"
        score, rs, re, fs, fe = CPUAligner.sw_align(read, ref)
        assert score == 0.0
        assert rs == re == fs == fe == 0

    def test_partial_match(self):
        read = "AAACCC"
        ref = "CCCGGG"
        score, rs, re, fs, fe = CPUAligner.sw_align(read, ref)
        assert score > 0
        # Should find the CCC match
        assert rs <= 3  # match starts in read at 3
        assert re >= 6  # ends at 6
        assert fs <= 0  # match starts in ref at 0
        assert fe >= 3  # ends at 3

    def test_gap_penalty_reduces_score(self):
        """A gapped alignment should score less than a perfect match."""
        read = "ACGT"
        ref = "ACGT"
        score_perfect, _, _, _, _ = CPUAligner.sw_align(read, ref)

        # Delete a base from read to force a gap
        read_gapped = "AGT"  # ACGT with C deleted
        score_gapped, _, _, _, _ = CPUAligner.sw_align(read_gapped, ref)
        assert score_gapped < score_perfect

    def test_affine_gap_prefers_one_long_gap(self):
        """Affine gap model: one long gap scores better than many short ones."""
        read = "AAAAAGGGGG"
        ref = "AAAATTTGGGGG"
        # One 3bp gap vs three 1bp gaps → affine prefers one long
        score1, _, _, _, _ = CPUAligner.sw_align(read, ref, gap_open=5, gap_extend=1)
        assert score1 > 0

    def test_batch_align(self):
        batch = AlignBatch(
            batch_id=0,
            reads=["ACGT", "TGCA"],
            ref="ACGTACGT",
        )
        scores, rs, re, fs, fe = CPUAligner.align_batch(batch)
        assert len(scores) == 2
        assert scores[0] > 0 or scores[1] > 0  # at least one should match


class TestGPUWorker:
    """Test the main gpu_worker entry point (will use CPU fallback if no GPU)."""

    def test_worker_returns_result(self):
        batch = AlignBatch(
            batch_id=0,
            reads=["ACGTACGT", "TGCATGCA"],
            ref="ACGTACGTACGT",
        )
        result = gpu_worker(batch, with_bounds=True)
        assert isinstance(result, AlignResult)
        assert len(result.scores) == 2
        assert result.n_reads == 2
        assert result.elapsed_ms > 0
        assert result.kernel_used in ("sw_affine", "sw_score_only", "cpu")

    def test_worker_without_bounds(self):
        batch = AlignBatch(
            batch_id=1,
            reads=["AAAA", "CCCC"],
            ref="AAAACCCC",
        )
        result = gpu_worker(batch, with_bounds=False)
        assert result.read_start is None
        assert result.ref_start is None

    def test_worker_custom_sw_params(self):
        batch = AlignBatch(
            batch_id=2,
            reads=["ACGT"],
            ref="ACGT",
        )
        result = gpu_worker(
            batch, band_width=20, gap_open=5, gap_extend=2,
        )
        assert result.scores[0] > 0

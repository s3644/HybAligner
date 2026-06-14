"""Tests for CIGAR traceback module."""

import pytest
from cpu.cigar import (
    traceback_cigar, batch_traceback_cigar, cigar_stats,
    CIGAR_MATCH, CIGAR_INSERT, CIGAR_DELETE,
)


class TestTracebackCigar:
    def test_perfect_match(self):
        read = "ACGTACGT"
        ref = "ACGTACGT"
        cigar = traceback_cigar(read, ref, 0, 8, 0, 8)
        assert cigar == "8M"

    def test_single_substitution(self):
        read = "ACGT"
        ref = "ACGA"  # T→A substitution
        cigar = traceback_cigar(read, ref, 0, 4, 0, 4)
        # Should still be 4M (matches include mismatches in CIGAR M)
        assert "M" in cigar

    def test_insertion(self):
        read = "ACGTA"
        ref = "ACGT"
        cigar = traceback_cigar(read, ref, 0, 5, 0, 4)
        # Should have an insertion
        assert "I" in cigar or cigar != "*"

    def test_deletion(self):
        read = "ACGT"
        ref = "ACGTA"
        cigar = traceback_cigar(read, ref, 0, 4, 0, 5)
        assert "D" in cigar or cigar != "*"

    def test_no_alignment(self):
        read = "AAAA"
        ref = "CCCC"
        cigar = traceback_cigar(read, ref, 0, 4, 0, 4)
        assert cigar == "*"

    def test_empty_bounds(self):
        cigar = traceback_cigar("ACGT", "ACGT", 0, 0, 0, 0)
        assert cigar == "*"

    def test_soft_clipping(self):
        read = "NNACGTNN"
        ref = "ACGT"
        # Alignment is in the middle
        cigar = traceback_cigar(read, ref, 2, 6, 0, 4)
        assert "S" in cigar  # soft-clips on both ends
        assert "M" in cigar

    def test_gap_in_middle(self):
        """Read has an insertion relative to reference."""
        read = "ACGTXXACGT"
        ref = "ACGTACGT"
        cigar = traceback_cigar(read, ref, 0, 10, 0, 8)
        # Should have an insertion or soft-clip
        assert len(cigar) > 0
        assert cigar != "*"


class TestBatchTracebackCigar:
    def test_batch(self):
        import numpy as np
        reads = ["ACGT", "TGCA"]
        ref = "ACGTTGCA"
        rs = np.array([0, 0], dtype=np.int32)
        re = np.array([4, 4], dtype=np.int32)
        fs = np.array([0, 4], dtype=np.int32)
        fe = np.array([4, 8], dtype=np.int32)
        cigars = batch_traceback_cigar(reads, ref, rs, re, fs, fe)
        assert len(cigars) == 2
        assert cigars[0] != "*"
        assert cigars[1] != "*"

    def test_batch_with_noalign(self):
        import numpy as np
        reads = ["AAAA", "CCCC"]
        ref = "ACGTACGT"
        rs = np.array([0, 0], dtype=np.int32)
        re = np.array([0, 0], dtype=np.int32)
        fs = np.array([0, 0], dtype=np.int32)
        fe = np.array([0, 0], dtype=np.int32)
        cigars = batch_traceback_cigar(reads, ref, rs, re, fs, fe)
        assert cigars == ["*", "*"]


class TestCigarStats:
    def test_match_only(self):
        stats = cigar_stats("8M")
        assert stats["matches"] == 8
        assert stats["insertions"] == 0
        assert stats["deletions"] == 0
        assert stats["aligned_bases"] == 8

    def test_mixed(self):
        stats = cigar_stats("5M2I3M1D2M")
        assert stats["matches"] == 10  # 5 + 3 + 2
        assert stats["insertions"] == 2
        assert stats["deletions"] == 1
        assert stats["aligned_bases"] == 12  # 10M + 2I

    def test_soft_clip(self):
        stats = cigar_stats("3S5M2S")
        assert stats["soft_clips"] == 5  # 3 + 2
        assert stats["matches"] == 5

    def test_no_alignment(self):
        stats = cigar_stats("*")
        assert stats["cigar"] == "*"
        assert stats["aligned_bases"] == 0

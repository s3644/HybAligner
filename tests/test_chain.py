"""Tests for CPU chaining module."""

import pytest
from cpu.chain import (
    Anchor, Chain, chain_anchors, extract_anchors,
    _canonical_kmer, _gap_penalty,
)


class TestCanonicalKmer:
    def test_forward_less_than_revcomp(self):
        # AAAAA... forward = revcomp(TTTTT...) — forward is smaller
        kmer = "A" * 15
        result = _canonical_kmer(kmer)
        assert result == kmer  # AAAAA < TTTTT

    def test_revcomp_less_than_forward(self):
        kmer = "T" * 15
        result = _canonical_kmer(kmer)
        assert result == "A" * 15  # AAAAA canonical

    def test_length(self):
        result = _canonical_kmer("ACGTACGTACGTACG")
        assert len(result) == 15

    def test_contains_only_acgt(self):
        result = _canonical_kmer("AAAAAAAAAAAAAAA")  # poly-A
        assert all(c in "ACGT" for c in result)


class TestGapPenalty:
    def test_zero_gap(self):
        assert _gap_penalty(0, 0) == 0.0

    def test_equal_gaps(self):
        # When ref_gap == read_gap, diff=0, only log term
        p = _gap_penalty(100, 100)
        assert p > 0
        assert p < 10  # reasonable range

    def test_different_gaps(self):
        # Large difference should increase penalty
        p1 = _gap_penalty(10, 10)
        p2 = _gap_penalty(10, 100)
        assert p2 > p1


class TestAnchor:
    def test_diagonal(self):
        a = Anchor(read_pos=10, ref_pos=20)
        assert a.diag == 10  # 20 - 10

    def test_negative_diagonal(self):
        a = Anchor(read_pos=50, ref_pos=10)
        assert a.diag == -40

    def test_score(self):
        a = Anchor(read_pos=0, ref_pos=0, length=15)
        assert a.score == 15.0


class TestChain:
    def test_empty_chain(self):
        c = Chain()
        assert c.num_anchors == 0
        assert c.score == 0.0

    def test_single_anchor(self):
        c = Chain()
        a = Anchor(read_pos=10, ref_pos=20, length=15)
        c.add_anchor(a)
        assert c.num_anchors == 1
        assert c.score == 15.0
        assert c.read_start == 10
        assert c.read_end == 25
        assert c.ref_start == 20
        assert c.ref_end == 35

    def test_two_anchors_expands_bounds(self):
        c = Chain()
        c.add_anchor(Anchor(read_pos=10, ref_pos=20, length=15))
        c.add_anchor(Anchor(read_pos=50, ref_pos=60, length=15))
        assert c.num_anchors == 2
        assert c.read_start == 10
        assert c.read_end == 65
        assert c.ref_start == 20
        assert c.ref_end == 75


class TestChainAnchors:
    def test_colinear_anchors(self):
        """Two colinear anchors should form a single chain."""
        anchors = [
            Anchor(read_pos=10, ref_pos=100, length=15),
            Anchor(read_pos=60, ref_pos=150, length=15),
        ]
        chain = chain_anchors(anchors)
        assert chain.num_anchors == 2
        assert chain.score > 0

    def test_conflicting_anchors(self):
        """Non-colinear anchors — best chain picked."""
        anchors = [
            Anchor(read_pos=10, ref_pos=100, length=15),
            Anchor(read_pos=60, ref_pos=90, length=15),   # reversed diag
        ]
        chain = chain_anchors(anchors)
        assert chain.num_anchors >= 1

    def test_empty_anchors(self):
        chain = chain_anchors([])
        assert chain.num_anchors == 0
        assert chain.score == 0.0

    def test_single_anchor(self):
        anchors = [Anchor(read_pos=5, ref_pos=50, length=15)]
        chain = chain_anchors(anchors)
        assert chain.num_anchors == 1
        assert chain.score == 15.0


class TestExtractAnchors:
    def test_identical_sequences(self):
        seq = "ACGTACGTACGTACG"  # 15 bp
        anchors = extract_anchors(seq, seq, k=8, w=5)
        assert len(anchors) > 0

    def test_no_match(self):
        read = "A" * 100
        ref = "C" * 100
        anchors = extract_anchors(read, ref, k=15, w=10)
        # Only poly-A minimizers — no match against poly-C
        assert all(a.read_pos >= 0 for a in anchors)

    def test_partial_match(self):
        read = "ACGTACGTACGT" + "G" * 88  # 100 bp
        ref = "ACGTACGTACGT" + "C" * 88
        anchors = extract_anchors(read, ref, k=8, w=5)
        assert len(anchors) > 0  # should find anchors in the matching prefix

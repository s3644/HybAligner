"""Tests for GPU seeder module (CPU fallback path)."""

import pytest
from gpu.seeder import GPUSeeder, RefMinimizerIndex, SeedResult


class TestRefMinimizerIndex:
    def test_create(self):
        import numpy as np
        idx = RefMinimizerIndex(
            hashes=np.zeros((1, 10), dtype=np.uint64),
            positions=np.zeros((1, 10), dtype=np.int32),
            counts=np.array([5], dtype=np.int32),
            n_total=5,
            k=15, w=10,
        )
        assert idx.n_total == 5
        assert idx.k == 15
        assert idx.w == 10


class TestGPUSeeder:
    def test_build_ref_index(self):
        seeder = GPUSeeder(k=8, w=5)
        ref_index = seeder.build_ref_index(["ACGTACGTACGT"], seq_len=12)
        assert ref_index.n_total > 0
        assert ref_index.hashes.shape == (1, 512)
        assert ref_index.positions.shape == (1, 512)

    def test_seed_batch(self):
        seeder = GPUSeeder(k=8, w=5)
        ref = "ACGTACGTACGTACGT"
        ref_index = seeder.build_ref_index([ref], seq_len=len(ref))

        reads = ["ACGTACGT", "TGCATGCA"]
        result = seeder.seed_batch(reads, read_len=8, ref_index=ref_index)

        assert isinstance(result, SeedResult)
        assert len(result.anchors) == 2
        assert result.n_total_anchors >= 0
        assert result.elapsed_ms > 0

    def test_seed_batch_empty_reads(self):
        seeder = GPUSeeder(k=8, w=5)
        ref_index = seeder.build_ref_index(["AAAAAAAAAAAA"], seq_len=12)
        result = seeder.seed_batch(["CCCCCCCC"], read_len=8, ref_index=ref_index)
        assert len(result.anchors) == 1
        # No matches expected between poly-A ref and poly-C read
        assert len(result.anchors[0]) == 0

    def test_get_seeder_singleton(self):
        from gpu.seeder import get_seeder
        s1 = get_seeder()
        s2 = get_seeder()
        assert s1 is s2

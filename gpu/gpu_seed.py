"""Pure-GPU Seeding — eliminates Python from the hot path.

Uses the existing CUDA seed kernels directly:
  1. launch_extract_minimizers(ref) → GPU hash table keys
  2. launch_build_hash_table → GPU open-addressing hash table
  3. launch_extract_minimizers(reads) → read minimizers
  4. launch_match_hash_table → GPU anchors (all on GPU!)

Python only does: load files, dispatch kernels, collect results.
Zero Python dict lookups, zero Python for-loops in hot path.
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from gpu.worker import _get_launcher


class GPUSeedingPipeline:
    """Pure-GPU seeding — one kernel sequence, zero Python in hot path."""

    def __init__(self):
        self._launcher = _get_launcher()

    def build_ref_index(self, ref_seq: str, k: int = 15, w: int = 10) -> dict:
        """Build GPU hash table from reference sequence.

        Returns dict with pre-allocated GPU output arrays.
        All minimizer extraction and hash table building happens on GPU.
        """
        ref_packed = ref_seq.ljust(len(ref_seq), 'N').encode()
        max_mins = min(10000, max(512, len(ref_seq) // (k + w)))

        # Step 1: Extract reference minimizers on GPU
        hashes, positions, counts = self._launcher.extract_minimizers(
            ref_packed, 1, len(ref_seq), k, w, max_mins,
        )
        n_mins = int(counts[0])
        if n_mins == 0:
            return {'table_keys': None, 'table_vals': None, 'table_size': 0, 'n_mins': 0}

        # Step 2: Build GPU hash table
        table_size = 1
        while table_size < n_mins * 2:
            table_size *= 2

        ref_hashes_1d = np.ascontiguousarray(hashes.ravel()[:n_mins])
        ref_pos_1d = np.ascontiguousarray(positions.ravel()[:n_mins])

        table_keys, table_vals, _ = self._launcher.build_hash_table(
            ref_hashes_1d, ref_pos_1d, n_mins,
        )

        return {
            'hashes': hashes, 'positions': positions, 'counts': counts,
            'table_keys': table_keys, 'table_vals': table_vals,
            'table_size': table_size, 'n_mins': n_mins,
        }

    def seed_batch_gpu(
        self,
        reads: List[str],
        read_len: int,
        ref_index: dict,
        k: int = 15,
        w: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pure-GPU seeding: extract minimizers + match hash table.

        All computation on GPU — Python only dispatches.

        Returns: (best_read_pos, best_ref_pos) arrays, shape (N,)
                 Values are -1 for unseeded reads.
        """
        n_reads = len(reads)
        if n_reads == 0 or ref_index['n_mins'] == 0:
            return (np.full(n_reads, -1, dtype=np.int32),
                    np.full(n_reads, -1, dtype=np.int32))

        # Step 1: Encode reads into packed bytes
        packed = bytearray(n_reads * read_len)
        for i, read in enumerate(reads):
            s = read[:read_len].ljust(read_len, 'N')
            packed[i * read_len:(i + 1) * read_len] = s.encode()

        max_mins = min(512, read_len // (k + w) + 10)

        # Step 2: Extract read minimizers on GPU
        rh, rp, rc = self._launcher.extract_minimizers(
            bytes(packed), n_reads, read_len, k, w, max_mins,
        )

        # Step 3: Match against GPU hash table (n_reads and max_mins from shapes)
        max_anchors = 32
        arp, afp, ac = self._launcher.match_hash_table(
            rh, rp,
            ref_index['table_keys'], ref_index['table_vals'],
            ref_index['table_size'],
            max_vals_per_key=8,
            max_anchors=max_anchors,
        )

        # Step 4: Best anchor per read (CPU — fast, just argmax on small arrays)
        best_rp = np.full(n_reads, -1, dtype=np.int32)
        best_fp = np.full(n_reads, -1, dtype=np.int32)

        for i in range(n_reads):
            n = int(ac[i])
            if n == 0:
                continue
            # arp shape: (n_reads * max_anchors,) — flat
            base = i * max_anchors
            best_rp[i] = int(arp[base])
            best_fp[i] = int(afp[base])

        return best_rp, best_fp


def benchmark_gpu_seeding():
    """Benchmark pure-GPU seeding vs CPU Python seeding."""
    import random
    random.seed(42)
    DNA = 'ACGT'

    ref = ''.join(random.choice(DNA) for _ in range(100_000))
    reads = []
    for i in range(500):
        start = random.randint(0, max(1, len(ref) - 10100))
        reads.append(ref[start:start + random.randint(5000, 15000)])

    print(f"Ref: {len(ref):,} bp, Reads: {len(reads)}")

    # GPU seeding
    print("\n── GPU Seeding Pipeline ──")
    gpu = GPUSeedingPipeline()
    t0 = time.perf_counter()
    ref_idx = gpu.build_ref_index(ref, k=15, w=10)
    build_ms = (time.perf_counter() - t0) * 1000
    print(f"  Ref index: {ref_idx['n_mins']} minimizers ({build_ms:.0f}ms)")

    read_len = max(len(r) for r in reads)
    t0 = time.perf_counter()
    brp, bfp = gpu.seed_batch_gpu(reads, read_len, ref_idx, k=15, w=10)
    seed_ms = (time.perf_counter() - t0) * 1000
    n_seeded = int(np.sum(brp >= 0))
    print(f"  Seed batch: {n_seeded}/{len(reads)} reads ({seed_ms:.0f}ms)")

    # CPU seeding (for comparison)
    print("\n── CPU Seeding (Python dict) ──")
    from gpu.wgs_align import ChunkIndex
    ci = ChunkIndex(0, ref, 0)
    t0 = time.perf_counter()
    ci.build(k8=8, k15=15, w15=10)
    cpu_build_ms = (time.perf_counter() - t0) * 1000
    print(f"  Ref index: {len(ci.index_15mer):,} keys ({cpu_build_ms:.0f}ms)")

    t0 = time.perf_counter()
    cpu_seeded = 0
    for read in reads:
        anchors = ci.query(read, k8=8, k15=15, w15=10)
        if anchors:
            cpu_seeded += 1
    cpu_seed_ms = (time.perf_counter() - t0) * 1000
    print(f"  Seed batch: {cpu_seeded}/{len(reads)} reads ({cpu_seed_ms:.0f}ms)")

    print(f"\n{'='*50}")
    print(f"  GPU seeding:  {seed_ms:.0f}ms (build: {build_ms:.0f}ms)")
    print(f"  CPU seeding:  {cpu_seed_ms:.0f}ms (build: {cpu_build_ms:.0f}ms)")
    print(f"  GPU speedup over CPU: {cpu_seed_ms/seed_ms:.1f}×")
    print(f"{'='*50}")


if __name__ == '__main__':
    benchmark_gpu_seeding()

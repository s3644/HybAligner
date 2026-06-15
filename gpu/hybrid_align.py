"""Hybrid Aligner — multi-core CPU seeding + batched GPU alignment.

Combines:
  - ThreadPoolExecutor for parallel seed matching across CPU cores
  - Per-chunk batched GPU Smith-Waterman (many reads per ctypes call)
  - Zero-copy result merging

Usage:
    ha = HybridAligner(n_workers=16)
    ha.build_index("hg38.fa")
    result = ha.align("reads.fastq")

Performance targets (DGX Spark GB10, 47 Mbp chr21):
  - 10K reads: ~800ms (12,500 r/s — competitive with minimap2)
  - CPU scaling: near-linear with core count for seeding phase
"""

from __future__ import annotations

import gzip
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from gpu.wgs_align import (
    WgsAligner, ChunkIndex, _load_fasta,
)


class HybridAligner(WgsAligner):
    """Multi-core CPU seeding + batched GPU SW alignment.

    Inherits chunked index building from WgsAligner.
    Adds parallel seed matching and batched GPU calls.
    """

    def __init__(
        self,
        chunk_size: int = 10_000_000,
        overlap: int = 1_000_000,
        n_workers: int = 0,
    ):
        super().__init__(chunk_size, overlap)
        self.n_workers = n_workers if n_workers > 0 else (os.cpu_count() or 4)

    # ------------------------------------------------------------------
    # Parallel alignment
    # ------------------------------------------------------------------
    def align(
        self,
        fastq_path: str,
        band_width: int = 50,
        gap_open: int = 5,
        gap_extend: int = 2,
        anchor_window: int = 5000,
        batch_size: int = 500,
    ) -> dict:
        """Hybrid alignment: parallel CPU seeding + batched GPU SW.

        Pipeline:
        1. Parse FASTQ (single-thread, fast)
        2. Split reads into batches
        3. ThreadPool: seed each batch in parallel → anchors
        4. Group seeded reads by reference chunk
        5. Per chunk: batched GPU SW (all reads in chunk, one ctypes call)
        6. Merge results by original read index
        """
        t_total = time.perf_counter()

        # ── Parse FASTQ ────────────────────────────
        t_parse = time.perf_counter()
        with open(fastq_path) as f:
            lines = f.read().splitlines()
        reads = [lines[i] for i in range(1, len(lines), 4)]
        n_reads = len(reads)
        read_len = max(len(r) for r in reads) if reads else 0
        parse_ms = (time.perf_counter() - t_parse) * 1000

        # ── Parallel Seeding (CPU) ─────────────────
        t_seed = time.perf_counter()
        read_batches = _split_batches(reads, batch_size)
        n_seeded = 0

        # Per-read results: (read_idx, anchor_read_pos, anchor_ref_pos)
        read_anchors: List[Optional[Tuple[int, int]]] = [None] * n_reads

        with ThreadPoolExecutor(max_workers=self.n_workers) as pool:
            futures = {}
            for batch_idx, batch_reads in enumerate(read_batches):
                start_idx = batch_idx * batch_size
                future = pool.submit(
                    _seed_batch,
                    batch_reads, start_idx, self.chunks, read_len,
                )
                futures[future] = batch_idx

            for future in as_completed(futures):
                batch_results = future.result()
                for read_idx, anchor in batch_results:
                    read_anchors[read_idx] = anchor
                    if anchor is not None:
                        n_seeded += 1

        seed_ms = (time.perf_counter() - t_seed) * 1000

        # ── Init GPU aligner ───────────────────────
        if self._aligner is None:
            from gpu.fast_align import FastAligner
            self._aligner = FastAligner(
                max_reads=n_reads + 10,
                max_read_len=read_len,
                max_ref_len=anchor_window * 2 + read_len + 100,
            )
            self._aligner.align(["A"], "N" * 100)

        # ── Batched GPU SW (per anchor cluster) ─────
        t_gpu = time.perf_counter()
        scores = np.zeros(n_reads, dtype=np.float32)
        read_starts = np.zeros(n_reads, dtype=np.int32)
        read_ends = np.zeros(n_reads, dtype=np.int32)
        ref_starts = np.zeros(n_reads, dtype=np.int32)
        ref_ends = np.zeros(n_reads, dtype=np.int32)

        # Cluster reads by nearby anchor positions within each chunk
        CLUSTER_RADIUS = anchor_window  # reads within ±anchor_window share a ref window

        for chunk in self.chunks:
            # Collect reads that anchor to this chunk
            chunk_data: List[Tuple[int, str, int, int]] = []  # (idx, read, rp, fp)
            for i, anchor in enumerate(read_anchors):
                if anchor is None:
                    continue
                rp, fp = anchor
                if chunk.ref_start <= fp < chunk.ref_end:
                    chunk_data.append((i, reads[i], rp, fp))

            if not chunk_data:
                continue

            # Sort by anchor ref position
            chunk_data.sort(key=lambda x: x[3])  # sort by fp (global ref pos)

            # Cluster reads with nearby anchors
            clusters = _cluster_by_position(chunk_data, CLUSTER_RADIUS)

            for cluster in clusters:
                # Find common ref window covering all anchors in cluster
                min_fp = min(c[3] for c in cluster)
                max_fp = max(c[3] for c in cluster)
                ref_start = max(0, min_fp - anchor_window)
                ref_end = min(self._ref_len, max_fp + read_len + anchor_window)
                ref_window = self._get_ref_slice(ref_start, ref_end)

                # Batch all reads in this cluster — ONE GPU call
                cluster_reads = [c[1] for c in cluster]
                cluster_indices = [c[0] for c in cluster]

                try:
                    s, rs, re, fs, fe = self._aligner.align(
                        cluster_reads, ref_window,
                        band_width=band_width,
                        gap_open=gap_open,
                        gap_extend=gap_extend,
                        zero_copy=True,
                    )
                    for k, idx in enumerate(cluster_indices):
                        scores[idx] = s[k]
                        read_starts[idx] = int(rs[k])
                        read_ends[idx] = int(re[k])
                        ref_starts[idx] = ref_start + int(fs[k])
                        ref_ends[idx] = ref_start + int(fe[k])
                except Exception:
                    continue  # skip failed cluster

        gpu_ms = (time.perf_counter() - t_gpu) * 1000
        total_ms = (time.perf_counter() - t_total) * 1000

        n_aligned = int(np.count_nonzero(scores))
        return {
            "pipeline": "HybAligner v0.9.0 (hybrid CPU+GPU)",
            "algorithm": "Parallel CPU seeding + batched GPU SW",
            "n_reads": n_reads,
            "n_aligned": n_aligned,
            "n_seeded": n_seeded,
            "pct_aligned": round(100.0 * n_aligned / max(1, n_reads), 2),
            "ref_len": self._ref_len,
            "read_len": read_len,
            "band_width": band_width,
            "n_workers": self.n_workers,
            "parse_ms": round(parse_ms, 2),
            "seed_ms": round(seed_ms, 2),
            "gpu_ms": round(gpu_ms, 2),
            "total_ms": round(total_ms, 2),
            "throughput_reads_per_sec": round(n_reads / max(0.001, total_ms / 1000.0), 1),
            "score_mean": round(float(np.mean(scores[scores > 0])), 4) if n_aligned else 0,
            "score_max": round(float(np.max(scores)), 4) if n_aligned else 0,
            "n_chunks": len(self.chunks),
        }


# ---------------------------------------------------------------------------
# Worker functions (module-level for pickling)
# ---------------------------------------------------------------------------
def _split_batches(items: list, batch_size: int) -> List[list]:
    """Split a list into fixed-size batches."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def _cluster_by_position(
    data: List[Tuple[int, str, int, int]],
    radius: int,
) -> List[List[Tuple[int, str, int, int]]]:
    """Cluster items by anchor position (fp, index 3) within radius.

    Greedy clustering: consecutive items within `radius` bp form a cluster.
    """
    if not data:
        return []
    clusters = []
    current = [data[0]]
    for item in data[1:]:
        # Distance from last item in current cluster
        if item[3] - current[-1][3] <= radius:
            current.append(item)
        else:
            clusters.append(current)
            current = [item]
    clusters.append(current)
    return clusters


def _seed_batch(
    reads: List[str],
    start_idx: int,
    chunks: List[ChunkIndex],
    read_len: int,
) -> List[Tuple[int, Optional[Tuple[int, int]]]]:
    """Seed a batch of reads against all chunk indexes.

    Returns list of (read_idx, (read_pos, ref_pos)) or (read_idx, None).
    """
    results = []
    for i, read in enumerate(reads):
        read_idx = start_idx + i
        all_anchors = []
        for chunk in chunks:
            anchors = chunk.query(read)
            all_anchors.extend(anchors)

        if not all_anchors:
            results.append((read_idx, None))
            continue

        # Best anchor via diagonal consensus
        diag_counts: Dict[int, int] = {}
        for rp, fp in all_anchors:
            d = fp - rp
            diag_counts[d] = diag_counts.get(d, 0) + 1
        if not diag_counts:
            results.append((read_idx, None))
            continue
        best_diag = max(diag_counts, key=diag_counts.get)
        best = next((a for a in all_anchors if a[1] - a[0] == best_diag), all_anchors[0])
        results.append((read_idx, best))

    return results


import os  # noqa: E402 (needed at module level for n_workers default)

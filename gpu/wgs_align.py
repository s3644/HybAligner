"""WGS Aligner — genome-scale alignment via chunked indexing + hierarchical seeds.

Unified architecture supporting human WGS (3.2 Gbp):
  1. Reference chunking (10 Mbp chunks, 1 Mbp overlap)
  2. Two-level seeding: 8-mer coarse + 15-mer minimizer fine
  3. Diagonal consensus anchor selection
  4. Per-chunk batched GPU banded SW
  5. Index serialization (save/load from disk)

Usage:
    wa = WgsAligner()
    wa.build_index("hg38.fa", chunk_size=10_000_000)
    wa.save_index("hg38.idx")
    # ... later ...
    wa.load_index("hg38.idx")
    result = wa.align("reads.fastq")

Performance targets (DGX Spark GB10, 3.2 Gbp):
  - Index build: ~60s (one-time)
  - Index load: ~2s
  - 500 reads: ~200ms (cached index)
  - Throughput: ~2,500 reads/s
"""

from __future__ import annotations

import gzip
import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Coarse 8-mer index — fast LUT-based lookup
# ---------------------------------------------------------------------------
def _build_8mer_index(ref_seq: str, k: int = 8, stride: int = 1) -> Dict[int, List[int]]:
    """Build 8-mer positional index for coarse filtering.

    Uses 2-bit encoding: A=00, C=01, G=10, T=11 → fits in 16 bits.
    Stride=1 stores ALL positions for maximum sensitivity.
    """
    ENC = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    index: Dict[int, List[int]] = {}
    n = len(ref_seq)

    for i in range(0, n - k + 1, stride):
        h = 0
        valid = True
        for j in range(k):
            c = ref_seq[i + j]
            if c not in ENC:
                valid = False
                break
            h = (h << 2) | ENC[c]
        if valid:
            index.setdefault(h, []).append(i)

    return index


def _query_8mer_index(read: str, index: Dict[int, List[int]], k: int = 8) -> List[int]:
    """Find candidate positions — query ALL 8-mers from the read."""
    ENC = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    candidates: set = set()
    n = len(read)

    for i in range(0, n - k + 1):  # ALL positions
        h = 0
        valid = True
        for j in range(k):
            c = read[i + j]
            if c not in ENC:
                valid = False
                break
            h = (h << 2) | ENC[c]
        if valid and h in index:
            candidates.update(index[h])
            if len(candidates) > 50:  # early exit: enough candidates
                break

    return list(candidates)


# ---------------------------------------------------------------------------
# ChunkIndex — one reference chunk with dual-level seeds
# ---------------------------------------------------------------------------
class ChunkIndex:
    """Dual-level seed index for one reference chunk (~10 Mbp)."""

    __slots__ = ('chunk_id', 'ref_start', 'ref_end', 'ref_seq',
                 'index_8mer', 'index_15mer')

    def __init__(self, chunk_id: int, ref_seq: str, ref_start: int):
        self.chunk_id = chunk_id
        self.ref_start = ref_start
        self.ref_end = ref_start + len(ref_seq)
        self.ref_seq = ref_seq
        self.index_8mer: Dict[int, List[int]] = {}
        self.index_15mer: Dict[int, List[int]] = {}

    def build(self, k8: int = 8, k15: int = 15, w15: int = 10):
        """Build both 8-mer coarse and 15-mer minimizer fine indexes."""
        self.index_8mer = _build_8mer_index(self.ref_seq, k=k8, stride=1)
        from gpu.fast_align import _build_cpu_seed_index
        self.index_15mer = _build_cpu_seed_index(self.ref_seq, k=k15, w=w15)

    def query(self, read: str, k8: int = 8, k15: int = 15, w15: int = 10) -> List[Tuple[int, int]]:
        """Find anchors: list of (read_pos, ref_pos_global).

        Two-level filtering (fast + accurate):
        1. 8-mer coarse → WHICH CHUNKS to search (fast chunk selection)
        2. 15-mer fine → anchors using FULL chunk index (no region restriction)
        """
        from gpu.fast_align import _find_anchors_cpu

        # Level 1: 8-mer coarse → select relevant chunks
        candidates = _query_8mer_index(read, self.index_8mer, k=k8)
        if not candidates:
            return []

        # Level 2: 15-mer fine → full chunk index search
        # (no region restriction — we trust the 15-mer index to find correct positions)
        chunk_anchors = _find_anchors_cpu(read, self.index_15mer, k=k15, w=w15)

        # Convert to global coordinates
        anchors = [(rp, self.ref_start + fp) for rp, fp in chunk_anchors]
        return anchors


def _merge_regions(positions: List[int], gap: int = 5000) -> List[Tuple[int, int]]:
    """Merge nearby positions into contiguous regions."""
    if not positions:
        return []
    regions = []
    start = positions[0]
    end = positions[0]
    for p in positions[1:]:
        if p - end <= gap:
            end = p
        else:
            regions.append((max(0, start - gap), end + gap))
            start = p
            end = p
    regions.append((max(0, start - gap), end + gap))
    return regions


# ---------------------------------------------------------------------------
# WgsAligner — main genome-scale aligner
# ---------------------------------------------------------------------------
class WgsAligner:
    """Genome-scale aligner with chunked indexing and hierarchical seeds.

    Usage:
        wa = WgsAligner(chunk_size=10_000_000)
        wa.build_index("hg38.fa")
        wa.save_index("hg38.idx")
        result = wa.align("reads.fastq")
    """

    def __init__(self, chunk_size: int = 10_000_000, overlap: int = 1_000_000):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: List[ChunkIndex] = []
        self._ref_len: int = 0
        self._aligner = None  # lazy init

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------
    def build_index(self, ref_path: str):
        """Build chunked reference index from FASTA file."""
        t0 = time.perf_counter()

        # Load reference
        ref_seq = _load_fasta(ref_path)
        self._ref_len = len(ref_seq)
        print(f"Reference: {self._ref_len:,} bp")

        # Split into overlapping chunks
        self.chunks = []
        chunk_id = 0
        for start in range(0, self._ref_len, self.chunk_size - self.overlap):
            end = min(start + self.chunk_size, self._ref_len)
            chunk_seq = ref_seq[start:end]
            if len(chunk_seq) < 1000:  # skip tiny final chunks
                break

            ci = ChunkIndex(chunk_id, chunk_seq, start)
            t_chunk = time.perf_counter()
            ci.build()
            chunk_ms = (time.perf_counter() - t_chunk) * 1000
            self.chunks.append(ci)

            if chunk_id % 10 == 0 or chunk_id == 0:
                print(f"  Chunk {chunk_id}: {start:,}-{end:,} "
                      f"({len(chunk_seq):,}bp, {chunk_ms:.0f}ms, "
                      f"8mer:{len(ci.index_8mer):,}keys, 15mer:{len(ci.index_15mer):,}keys)")
            chunk_id += 1

        total_ms = (time.perf_counter() - t0) * 1000
        print(f"Index built: {len(self.chunks)} chunks in {total_ms/1000:.1f}s")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def save_index(self, path: str):
        """Save chunked index to disk (pickle)."""
        data = {
            'chunk_size': self.chunk_size,
            'overlap': self.overlap,
            'ref_len': self._ref_len,
            'chunks': [
                {
                    'chunk_id': c.chunk_id,
                    'ref_start': c.ref_start,
                    'ref_end': c.ref_end,
                    'ref_seq': c.ref_seq,
                    'index_8mer': c.index_8mer,
                    'index_15mer': c.index_15mer,
                }
                for c in self.chunks
            ],
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"Index saved: {path} ({size_mb:.1f} MB)")

    def load_index(self, path: str):
        """Load chunked index from disk."""
        t0 = time.perf_counter()
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.chunk_size = data['chunk_size']
        self.overlap = data['overlap']
        self._ref_len = data['ref_len']
        self.chunks = []
        for cd in data['chunks']:
            ci = ChunkIndex(cd['chunk_id'], '', cd['ref_start'])
            ci.ref_end = cd['ref_end']
            ci.ref_seq = cd['ref_seq']
            ci.index_8mer = cd['index_8mer']
            ci.index_15mer = cd['index_15mer']
            self.chunks.append(ci)

        load_ms = (time.perf_counter() - t0) * 1000
        print(f"Index loaded: {len(self.chunks)} chunks in {load_ms:.0f}ms")

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------
    def align(
        self,
        fastq_path: str,
        band_width: int = 50,
        gap_open: int = 5,
        gap_extend: int = 2,
        anchor_window: int = 5000,
    ) -> dict:
        """Align reads from FASTQ against the chunked reference index."""
        t_total = time.perf_counter()

        # Parse FASTQ
        t_parse = time.perf_counter()
        with open(fastq_path) as f:
            lines = f.read().splitlines()
        reads = [lines[i] for i in range(1, len(lines), 4)]
        n_reads = len(reads)
        read_len = max(len(r) for r in reads) if reads else 0
        parse_ms = (time.perf_counter() - t_parse) * 1000

        # Lazy init FastAligner
        if self._aligner is None:
            from gpu.fast_align import FastAligner
            self._aligner = FastAligner(
                max_reads=n_reads + 10,
                max_read_len=read_len,
                max_ref_len=anchor_window * 2 + read_len + 100,
            )
            self._aligner.align(["A"], "N" * 100)  # warmup

        # Align each read
        t_align = time.perf_counter()
        scores = np.zeros(n_reads, dtype=np.float32)
        read_starts = np.zeros(n_reads, dtype=np.int32)
        read_ends = np.zeros(n_reads, dtype=np.int32)
        ref_starts = np.zeros(n_reads, dtype=np.int32)
        ref_ends = np.zeros(n_reads, dtype=np.int32)
        n_seeded = 0

        for i, read in enumerate(reads):
            if i > 0 and i % 100 == 0:
                elapsed = (time.perf_counter() - t_align) * 1000
                rate = i / max(0.001, elapsed / 1000)
                print(f"  Progress: {i}/{n_reads} reads ({rate:.0f} r/s)...", end='\r')

            # Query each chunk (8-mer pre-filter + 15-mer fine)
            all_anchors = []
            for chunk in self.chunks:
                anchors = chunk.query(read)
                all_anchors.extend(anchors)

            if not all_anchors:
                continue
            n_seeded += 1

            # Best anchor via diagonal consensus (inline)
            diag_counts: Dict[int, int] = {}
            for rp, fp in all_anchors:
                d = fp - rp
                diag_counts[d] = diag_counts.get(d, 0) + 1
            if not diag_counts:
                continue
            best_diag = max(diag_counts, key=diag_counts.get)
            best_anchor = next((a for a in all_anchors if a[1] - a[0] == best_diag), all_anchors[0])
            rp, fp = best_anchor

            # Extract ref window
            ref_start = max(0, fp - anchor_window)
            ref_end = min(self._ref_len, fp + read_len + anchor_window)

            # Fetch reference sequence for this window
            # (from the appropriate chunk)
            ref_window = self._get_ref_slice(ref_start, ref_end)

            # GPU banded SW
            try:
                s, rs, re, fs, fe = self._aligner.align(
                    [read], ref_window,
                    band_width=band_width,
                    gap_open=gap_open,
                    gap_extend=gap_extend,
                    zero_copy=True,
                )
                scores[i] = s[0]
                read_starts[i] = int(rs[0])
                read_ends[i] = int(re[0])
                ref_starts[i] = ref_start + int(fs[0])
                ref_ends[i] = ref_start + int(fe[0])
            except Exception:
                continue

        align_ms = (time.perf_counter() - t_align) * 1000
        total_ms = (time.perf_counter() - t_total) * 1000

        n_aligned = int(np.count_nonzero(scores))
        print(f"  Done: {n_reads} reads in {total_ms:.0f}ms "
              f"({n_reads/(total_ms/1000):.0f} r/s), "
              f"{n_aligned}/{n_reads} aligned ({n_seeded} seeded)")

        return {
            "pipeline": "HybAligner v0.8.0 (WGS chunked)",
            "algorithm": "Chunked index + hierarchical seeds + banded SW",
            "n_reads": n_reads,
            "n_aligned": n_aligned,
            "n_seeded": n_seeded,
            "pct_aligned": round(100.0 * n_aligned / max(1, n_reads), 2),
            "ref_len": self._ref_len,
            "read_len": read_len,
            "band_width": band_width,
            "total_ms": round(total_ms, 2),
            "parse_ms": round(parse_ms, 2),
            "align_ms": round(align_ms, 2),
            "throughput_reads_per_sec": round(n_reads / max(0.001, total_ms / 1000.0), 1),
            "score_mean": round(float(np.mean(scores[scores > 0])), 4) if n_aligned else 0,
            "score_max": round(float(np.max(scores)), 4) if n_aligned else 0,
            "n_chunks": len(self.chunks),
        }

    def _get_ref_slice(self, start: int, end: int) -> str:
        """Get reference sequence for a genomic interval."""
        # Find the chunk containing this interval
        for chunk in self.chunks:
            if chunk.ref_start <= start < chunk.ref_end:
                local_start = start - chunk.ref_start
                local_end = min(end, chunk.ref_end) - chunk.ref_start
                return chunk.ref_seq[local_start:local_end]
        # Fallback: not found in any chunk
        return "N" * (end - start)


def _load_fasta(path: str) -> str:
    """Load reference sequence from FASTA file (supports .gz)."""
    if path.endswith('.gz'):
        with gzip.open(path, 'rt') as f:
            return ''.join(l.strip() for l in f if not l.startswith('>'))
    else:
        with open(path) as f:
            return ''.join(l.strip() for l in f if not l.startswith('>'))

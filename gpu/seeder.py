"""GPU seeding — high-level minimizer extraction and anchor matching.

Orchestrates the CUDA seed kernels (extract_minimizers, match_seeds)
to produce anchors for chaining-based alignment. Falls back to CPU
minimizer extraction when CUDA is unavailable.

Designed to work with cpu/chain.py for downstream chaining.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from obs.log import log
from gpu.worker import _get_launcher, CUDALauncher
from cpu.chain import Anchor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_KMER       = 15
DEFAULT_WINDOW     = 10
DEFAULT_MAX_MINS   = 512    # max minimizers per sequence
DEFAULT_MAX_ANCHORS = 256   # max anchors per read


@dataclass
class SeedResult:
    """Result of GPU seeding for a batch of reads."""
    anchors: List[List[Anchor]]  # anchors[read_idx] = list of Anchor objects
    n_total_anchors: int
    elapsed_ms: float
    backend: str  # "gpu" or "cpu"


# ---------------------------------------------------------------------------
# Reference minimizer index (computed once, reused across reads)
# ---------------------------------------------------------------------------
@dataclass
class RefMinimizerIndex:
    """Pre-computed reference minimizer index for fast seed matching."""
    hashes: np.ndarray       # (n_seqs, max_mins) uint64
    positions: np.ndarray    # (n_seqs, max_mins) int32
    counts: np.ndarray       # (n_seqs,) int32
    n_total: int             # total minimizers across all ref seqs
    k: int = DEFAULT_KMER
    w: int = DEFAULT_WINDOW
    # Hash table (GPU open addressing, built when GPU available)
    table_keys: Optional[np.ndarray] = None    # [table_size] uint64
    table_vals: Optional[np.ndarray] = None    # [table_size * max_vals] int32
    table_size: int = 0
    max_vals_per_key: int = 8


# ---------------------------------------------------------------------------
# GPU Seeder
# ---------------------------------------------------------------------------
class GPUSeeder:
    """GPU-accelerated minimizer extraction and seed matching.

    Usage:
        seeder = GPUSeeder()
        ref_index = seeder.build_ref_index([ref_seq], seq_len)
        result = seeder.seed_batch(reads, read_len, ref_index)
        # result.anchors can be fed to cpu.chain.chain_anchors()
    """

    def __init__(self, k: int = DEFAULT_KMER, w: int = DEFAULT_WINDOW):
        self.k = k
        self.w = w
        self._launcher: Optional[CUDALauncher] = None
        self._gpu_available: Optional[bool] = None

    @property
    def gpu_available(self) -> bool:
        if self._gpu_available is None:
            self._launcher = _get_launcher()
            self._gpu_available = self._launcher.available
        return self._gpu_available

    def build_ref_index(
        self,
        ref_seqs: List[str],
        seq_len: int,
        max_mins: int = DEFAULT_MAX_MINS,
    ) -> RefMinimizerIndex:
        """Build a reference minimizer index (GPU or CPU).

        This is called once per reference — the index is reused
        for all read batches.

        Args:
            ref_seqs: Reference sequences (typically just one).
            seq_len: Padded length for each reference sequence.
            max_mins: Max minimizers per reference sequence.

        Returns:
            RefMinimizerIndex ready for seed matching.
        """
        start = time.perf_counter()
        n = len(ref_seqs)

        if self.gpu_available:
            packed = bytearray(n * seq_len)
            for i, seq in enumerate(ref_seqs):
                s = seq.ljust(seq_len, 'N')[:seq_len]
                packed[i * seq_len:(i + 1) * seq_len] = s.encode()

            hashes, positions, counts = self._launcher.extract_minimizers(
                bytes(packed), n, seq_len, self.k, self.w, max_mins,
            )
            n_total = int(counts.sum())
            backend = "gpu"
        else:
            # CPU fallback: use cpu/chain.py extract_anchors for minimizer extraction
            from cpu.chain import _canonical_kmer

            hashes = np.zeros((n, max_mins), dtype=np.uint64)
            positions = np.zeros((n, max_mins), dtype=np.int32)
            counts = np.zeros(n, dtype=np.int32)

            for idx, seq in enumerate(ref_seqs):
                prev_hash = None
                cnt = 0
                num_windows = len(seq) - self.k - self.w + 2
                for win_start in range(num_windows):
                    if cnt >= max_mins:
                        break
                    min_hash = None
                    min_pos = -1
                    for offset in range(self.w):
                        pos = win_start + offset
                        if pos + self.k > len(seq):
                            break
                        kmer = seq[pos:pos + self.k]
                        h = hash(_canonical_kmer(kmer))
                        if min_hash is None or h < min_hash:
                            min_hash = h
                            min_pos = pos
                    if min_hash is not None and min_hash != prev_hash:
                        hashes[idx, cnt] = min_hash
                        positions[idx, cnt] = min_pos
                        cnt += 1
                        prev_hash = min_hash
                counts[idx] = cnt
            n_total = int(counts.sum())
            backend = "cpu"

        elapsed = (time.perf_counter() - start) * 1000.0
        log("ref_index_built",
            n_seqs=n, n_mins=n_total, time_ms=round(elapsed, 2), backend=backend,
        )

        index = RefMinimizerIndex(
            hashes=hashes,
            positions=positions,
            counts=counts,
            n_total=n_total,
            k=self.k,
            w=self.w,
        )

        # Build GPU hash table for O(1) lookup
        if self.gpu_available and n_total > 0:
            t_ht = time.perf_counter()
            try:
                ref_hashes_1d = np.ascontiguousarray(hashes.ravel()[:n_total])
                ref_pos_1d   = np.ascontiguousarray(positions.ravel()[:n_total])
                tk, tv, ts = self._launcher.build_hash_table(
                    ref_hashes_1d, ref_pos_1d, n_total,
                )
                index.table_keys = tk
                index.table_vals = tv
                index.table_size = ts
                ht_ms = (time.perf_counter() - t_ht) * 1000.0
                log("hash_table_built", size=ts, time_ms=round(ht_ms, 2))
            except Exception as e:
                log("hash_table_build_failed", error=str(e))
                # Fall back to brute-force matching

        return index

    def seed_batch(
        self,
        reads: List[str],
        read_len: int,
        ref_index: RefMinimizerIndex,
        max_mins: int = DEFAULT_MAX_MINS,
        max_anchors: int = DEFAULT_MAX_ANCHORS,
    ) -> SeedResult:
        """Extract minimizers from reads and match against reference index.

        Args:
            reads: List of read sequences.
            read_len: Padded read length.
            ref_index: Pre-built reference minimizer index.
            max_mins: Max minimizers per read.
            max_anchors: Max anchors per read.

        Returns:
            SeedResult with anchors per read.
        """
        start = time.perf_counter()
        n = len(reads)

        if self.gpu_available:
            # --- GPU path ---
            packed = bytearray(n * read_len)
            for i, read in enumerate(reads):
                r = read.ljust(read_len, 'N')[:read_len]
                packed[i * read_len:(i + 1) * read_len] = r.encode()

            # Step 1: Extract read minimizers
            rh, rp, rc = self._launcher.extract_minimizers(
                bytes(packed), n, read_len, self.k, self.w, max_mins,
            )

            # Step 2: Match against reference minimizers
            if ref_index.table_keys is not None:
                # Hash table lookup (O(1) per probe)
                arp, afp, ac = self._launcher.match_hash_table(
                    rh, rp,
                    ref_index.table_keys,
                    ref_index.table_vals,
                    ref_index.table_size,
                    ref_index.max_vals_per_key,
                    max_anchors,
                )
                backend = "gpu_hash"
            else:
                # Brute-force fallback
                ref_hashes_flat = ref_index.hashes.ravel()
                ref_pos_flat    = ref_index.positions.ravel()
                n_ref_mins = ref_index.n_total

                arp, afp, ac = self._launcher.match_seeds(
                    rh, rp, ref_hashes_flat, ref_pos_flat,
                    n_ref_mins, max_anchors,
                )
                backend = "gpu"
        else:
            # --- CPU path ---
            rh = np.zeros((n, max_mins), dtype=np.uint64)
            rp = np.zeros((n, max_mins), dtype=np.int32)
            rc = np.zeros(n, dtype=np.int32)

            from cpu.chain import _canonical_kmer

            for idx, read in enumerate(reads):
                prev_hash = None
                cnt = 0
                num_windows = len(read) - self.k - self.w + 2
                for win_start in range(num_windows):
                    if cnt >= max_mins:
                        break
                    min_hash = None
                    min_pos = -1
                    for offset in range(self.w):
                        pos = win_start + offset
                        if pos + self.k > len(read):
                            break
                        kmer = read[pos:pos + self.k]
                        h = hash(_canonical_kmer(kmer))
                        if min_hash is None or h < min_hash:
                            min_hash = h
                            min_pos = pos
                    if min_hash is not None and min_hash != prev_hash:
                        rh[idx, cnt] = min_hash
                        rp[idx, cnt] = min_pos
                        cnt += 1
                        prev_hash = min_hash
                rc[idx] = cnt

            # CPU matching: build dict of ref hashes → positions
            ref_dict: dict = {}
            for si in range(ref_index.counts.shape[0]):
                for mi in range(int(ref_index.counts[si])):
                    h = ref_index.hashes[si, mi]
                    p = ref_index.positions[si, mi]
                    ref_dict.setdefault(int(h), []).append(int(p))

            arp = np.zeros((n, max_anchors), dtype=np.int32)
            afp = np.zeros((n, max_anchors), dtype=np.int32)
            ac = np.zeros(n, dtype=np.int32)

            for idx in range(n):
                cnt = 0
                for mi in range(int(rc[idx])):
                    if cnt >= max_anchors:
                        break
                    h = int(rh[idx, mi])
                    if h in ref_dict:
                        for fp in ref_dict[h]:
                            if cnt >= max_anchors:
                                break
                            arp[idx, cnt] = int(rp[idx, mi])
                            afp[idx, cnt] = fp
                            cnt += 1
                ac[idx] = cnt
            backend = "cpu"

        # Convert numpy arrays to Anchor objects
        all_anchors: List[List[Anchor]] = []
        total = 0
        for idx in range(n):
            anchors = []
            for ai in range(int(ac[idx])):
                anchors.append(Anchor(
                    read_pos=int(arp[idx, ai]),
                    ref_pos=int(afp[idx, ai]),
                    length=self.k,
                ))
            all_anchors.append(anchors)
            total += len(anchors)

        elapsed = (time.perf_counter() - start) * 1000.0
        log("seed_batch_done",
            n_reads=n,
            n_anchors=total,
            time_ms=round(elapsed, 2),
            backend=backend,
        )

        return SeedResult(
            anchors=all_anchors,
            n_total_anchors=total,
            elapsed_ms=elapsed,
            backend=backend,
        )


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------
_seeder: Optional[GPUSeeder] = None


def get_seeder(k: int = DEFAULT_KMER, w: int = DEFAULT_WINDOW) -> GPUSeeder:
    """Get or create the global GPUSeeder instance."""
    global _seeder
    if _seeder is None:
        _seeder = GPUSeeder(k=k, w=w)
    return _seeder

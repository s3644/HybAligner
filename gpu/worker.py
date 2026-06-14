"""GPU worker — Python launcher for CUDA alignment and seeding kernels.

Provides the bridge between the Python runtime and CUDA C++ kernels
via ctypes. Falls back to NumPy CPU implementation when CUDA is unavailable.
"""

from __future__ import annotations

import os
import ctypes
import ctypes.util
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np

from obs.log import log

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_BLOCK_SIZE = 256
MAX_READ_LEN = 300   # default max read length (bp)
MAX_REF_LEN  = 5000000  # ~5 Mbp reference chunks


@dataclass
class AlignBatch:
    """A batch of reads to align against a reference."""
    batch_id: int
    reads: List[str]
    ref: str
    read_len: int = 0
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.read_len == 0 and self.reads:
            self.read_len = max(len(r) for r in self.reads)


@dataclass
class AlignResult:
    """Result from GPU alignment of a batch."""
    batch_id: int
    scores: np.ndarray              # shape (n_reads,)
    read_start: Optional[np.ndarray] = None  # alignment start in read
    read_end: Optional[np.ndarray] = None    # alignment end in read
    ref_start: Optional[np.ndarray] = None   # alignment start in ref
    ref_end: Optional[np.ndarray] = None     # alignment end in ref
    elapsed_ms: float = 0.0
    n_reads: int = 0
    kernel_used: str = "cuda"


# ---------------------------------------------------------------------------
# CUDA kernel loader (via ctypes)
# ---------------------------------------------------------------------------
class CUDALauncher:
    """Loads and calls the compiled CUDA kernel shared library."""

    def __init__(self, lib_path: Optional[str] = None):
        self._lib = None
        self._available = False

        if lib_path is None:
            # Search standard locations
            candidates = [
                os.path.join(os.path.dirname(__file__), "..", "build", "libcuda_kernels.so"),
                "./build/libcuda_kernels.so",
            ]
            for c in candidates:
                if os.path.exists(c):
                    lib_path = c
                    break

        if lib_path and os.path.exists(lib_path):
            try:
                self._lib = ctypes.CDLL(lib_path)
                self._available = True
                self._setup_signatures()
                log("cuda_library_loaded", path=lib_path)
            except OSError as e:
                log("cuda_library_load_failed", error=str(e))
        else:
            log("cuda_library_not_found",
                searched=", ".join(candidates if 'candidates' in dir() else [str(lib_path)]))

    def _setup_signatures(self):
        """Define ctypes function signatures for the CUDA kernels."""
        if not self._lib:
            return

        # launch_sw_affine — full Smith-Waterman with alignment bounds
        self._lib.launch_sw_affine.argtypes = [
            ctypes.c_char_p,                     # reads
            ctypes.c_char_p,                     # ref
            ctypes.POINTER(ctypes.c_float),      # scores
            ctypes.POINTER(ctypes.c_int),        # read_start
            ctypes.POINTER(ctypes.c_int),        # read_end
            ctypes.POINTER(ctypes.c_int),        # ref_start
            ctypes.POINTER(ctypes.c_int),        # ref_end
            ctypes.c_int,                        # num_reads
            ctypes.c_int,                        # read_len
            ctypes.c_int,                        # ref_len
            ctypes.c_int,                        # band_width
            ctypes.c_int,                        # gap_open
            ctypes.c_int,                        # gap_extend
            ctypes.c_int,                        # block_size
        ]
        self._lib.launch_sw_affine.restype = ctypes.c_int

        # launch_sw_score_only — Smith-Waterman scores only (no bounds)
        self._lib.launch_sw_score_only.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.launch_sw_score_only.restype = ctypes.c_int

        # launch_extract_minimizers
        self._lib.launch_extract_minimizers.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.launch_extract_minimizers.restype = ctypes.c_int

        # launch_match_seeds
        self._lib.launch_match_seeds.argtypes = [
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.launch_match_seeds.restype = ctypes.c_int

        # launch_build_hash_table
        self._lib.launch_build_hash_table.argtypes = [
            ctypes.POINTER(ctypes.c_ulonglong),  # ref_hashes
            ctypes.POINTER(ctypes.c_int),         # ref_pos
            ctypes.c_int,                          # n_ref_mins
            ctypes.POINTER(ctypes.c_ulonglong),   # table_keys (in/out)
            ctypes.POINTER(ctypes.c_int),         # table_vals (in/out)
            ctypes.c_int,                          # table_size
            ctypes.c_int,                          # max_vals_per_key
            ctypes.c_int,                          # block_size
        ]
        self._lib.launch_build_hash_table.restype = ctypes.c_int

        # launch_match_hash_table
        self._lib.launch_match_hash_table.argtypes = [
            ctypes.POINTER(ctypes.c_ulonglong),  # read_hashes
            ctypes.POINTER(ctypes.c_int),         # read_pos
            ctypes.POINTER(ctypes.c_ulonglong),   # table_keys
            ctypes.POINTER(ctypes.c_int),         # table_vals
            ctypes.c_int,                          # table_size
            ctypes.c_int,                          # max_vals_per_key
            ctypes.POINTER(ctypes.c_int),         # anchor_rp
            ctypes.POINTER(ctypes.c_int),         # anchor_fp
            ctypes.POINTER(ctypes.c_int),         # anchor_counts
            ctypes.c_int, ctypes.c_int, ctypes.c_int,  # n_reads, max_mins, max_anchors
            ctypes.c_int,                          # block_size
        ]
        self._lib.launch_match_hash_table.restype = ctypes.c_int

    @property
    def available(self) -> bool:
        return self._available

    def sw_affine(
        self,
        reads_packed: bytes,
        ref: bytes,
        n_reads: int,
        read_len: int,
        ref_len: int,
        band_width: int = 50,
        gap_open: int = 5,
        gap_extend: int = 2,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run full Smith-Waterman with affine gaps and alignment bounds.

        Returns:
            scores, read_start, read_end, ref_start, ref_end — each shape (n_reads,)
        """
        scores     = np.zeros(n_reads, dtype=np.float32)
        read_start = np.zeros(n_reads, dtype=np.int32)
        read_end   = np.zeros(n_reads, dtype=np.int32)
        ref_start  = np.zeros(n_reads, dtype=np.int32)
        ref_end    = np.zeros(n_reads, dtype=np.int32)

        ret = self._lib.launch_sw_affine(
            reads_packed, ref,
            scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            read_start.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            read_end.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ref_start.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ref_end.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            n_reads, read_len, ref_len,
            band_width, gap_open, gap_extend, block_size,
        )
        if ret != 0:
            raise RuntimeError(f"SW affine kernel failed with code {ret}")
        return scores, read_start, read_end, ref_start, ref_end

    def sw_score_only(
        self,
        reads_packed: bytes,
        ref: bytes,
        n_reads: int,
        read_len: int,
        ref_len: int,
        band_width: int = 50,
        gap_open: int = 5,
        gap_extend: int = 2,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> np.ndarray:
        """Run Smith-Waterman score-only kernel (faster, no alignment bounds)."""
        scores = np.zeros(n_reads, dtype=np.float32)
        ret = self._lib.launch_sw_score_only(
            reads_packed, ref,
            scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n_reads, read_len, ref_len,
            band_width, gap_open, gap_extend, block_size,
        )
        if ret != 0:
            raise RuntimeError(f"SW score-only kernel failed with code {ret}")
        return scores

    # ------------------------------------------------------------------
    # Minimizer seeding kernels
    # ------------------------------------------------------------------
    def extract_minimizers(
        self,
        seq_packed: bytes,
        n_seqs: int,
        seq_len: int,
        k: int = 15,
        w: int = 10,
        max_mins: int = 512,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract minimizers from packed sequences on GPU.

        Returns:
            hashes:  (n_seqs, max_mins) uint64 — minimizer hash values
            positions: (n_seqs, max_mins) int32 — minimizer positions
            counts:   (n_seqs,) int32 — actual number of minimizers per seq
        """
        hashes    = np.zeros((n_seqs, max_mins), dtype=np.uint64)
        positions = np.zeros((n_seqs, max_mins), dtype=np.int32)
        counts    = np.zeros(n_seqs, dtype=np.int32)

        ret = self._lib.launch_extract_minimizers(
            seq_packed,
            hashes.ctypes.data_as(ctypes.POINTER(ctypes.c_ulonglong)),
            positions.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            n_seqs, seq_len, k, w, max_mins, block_size,
        )
        if ret != 0:
            raise RuntimeError(f"extract_minimizers kernel failed with code {ret}")
        return hashes, positions, counts

    def match_seeds(
        self,
        read_hashes: np.ndarray,
        read_positions: np.ndarray,
        ref_hashes: np.ndarray,
        ref_positions: np.ndarray,
        n_ref_mins: int,
        max_anchors: int = 256,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Match read minimizers against reference minimizers on GPU.

        Args:
            read_hashes:    (n_reads, max_mins) uint64
            read_positions: (n_reads, max_mins) int32
            ref_hashes:     (n_ref_seqs, max_mins) uint64 — flattened 1D
            ref_positions:  (n_ref_seqs, max_mins) int32 — flattened 1D
            n_ref_mins:     total reference minimizers (= n_ref_seqs * max_mins)

        Returns:
            anchor_read_pos: (n_reads, max_anchors) int32
            anchor_ref_pos:  (n_reads, max_anchors) int32
            anchor_counts:   (n_reads,) int32
        """
        n_reads = read_hashes.shape[0]
        max_mins = read_hashes.shape[1]

        anchor_read_pos = np.zeros((n_reads, max_anchors), dtype=np.int32)
        anchor_ref_pos  = np.zeros((n_reads, max_anchors), dtype=np.int32)
        anchor_counts   = np.zeros(n_reads, dtype=np.int32)

        # Flatten 2D arrays to 1D for the C interface
        rh_flat = read_hashes.ravel()
        rp_flat = read_positions.ravel()

        ret = self._lib.launch_match_seeds(
            rh_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_ulonglong)),
            rp_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ref_hashes.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_ulonglong)),
            ref_positions.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            n_ref_mins,
            anchor_read_pos.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            anchor_ref_pos.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            anchor_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            n_reads, max_mins, max_anchors, block_size,
        )
        if ret != 0:
            raise RuntimeError(f"match_seeds kernel failed with code {ret}")
        return anchor_read_pos, anchor_ref_pos, anchor_counts

    def build_hash_table(
        self,
        ref_hashes: np.ndarray,
        ref_positions: np.ndarray,
        n_ref_mins: int,
        max_vals_per_key: int = 8,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Build GPU hash table from reference minimizers (open addressing).

        Args:
            ref_hashes: (n_ref_seqs, max_mins) uint64 — can be flattened
            ref_positions: (n_ref_seqs, max_mins) int32
            n_ref_mins: Actual number of reference minimizers.
            max_vals_per_key: Max positions per hash slot (for collisions).

        Returns:
            table_keys: Hash table keys array
            table_vals: Hash table values array
            table_size: Size of hash table
        """
        # Table size: power of 2, ~2x the number of elements (load factor ~0.5)
        table_size = 1
        while table_size < n_ref_mins * 2:
            table_size <<= 1

        table_keys = np.full(table_size, 0xFFFFFFFFFFFFFFFF, dtype=np.uint64)
        table_vals = np.full(table_size * max_vals_per_key, -1, dtype=np.int32)

        rh_flat = np.ascontiguousarray(ref_hashes.ravel()[:n_ref_mins])
        rp_flat = np.ascontiguousarray(ref_positions.ravel()[:n_ref_mins])

        ret = self._lib.launch_build_hash_table(
            rh_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_ulonglong)),
            rp_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            n_ref_mins,
            table_keys.ctypes.data_as(ctypes.POINTER(ctypes.c_ulonglong)),
            table_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            table_size, max_vals_per_key, DEFAULT_BLOCK_SIZE,
        )
        if ret != 0:
            raise RuntimeError(f"build_hash_table kernel failed with code {ret}")
        return table_keys, table_vals, table_size

    def match_hash_table(
        self,
        read_hashes: np.ndarray,
        read_positions: np.ndarray,
        table_keys: np.ndarray,
        table_vals: np.ndarray,
        table_size: int,
        max_vals_per_key: int = 8,
        max_anchors: int = 256,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Match read minimizers against a pre-built GPU hash table.

        Args:
            read_hashes: (n_reads, max_mins) uint64
            read_positions: (n_reads, max_mins) int32
            table_keys: Hash table keys [table_size] uint64
            table_vals: Hash table values [table_size * max_vals_per_key] int32
            table_size: Size of hash table
            max_vals_per_key: Max values per slot
            max_anchors: Max anchors per read

        Returns:
            anchor_rp, anchor_fp, anchor_counts
        """
        n_reads = read_hashes.shape[0]
        max_mins = read_hashes.shape[1]

        anchor_rp = np.zeros((n_reads, max_anchors), dtype=np.int32)
        anchor_fp = np.zeros((n_reads, max_anchors), dtype=np.int32)
        anchor_counts = np.zeros(n_reads, dtype=np.int32)

        rh_flat = np.ascontiguousarray(read_hashes.ravel())
        rp_flat = np.ascontiguousarray(read_positions.ravel())

        ret = self._lib.launch_match_hash_table(
            rh_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_ulonglong)),
            rp_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            table_keys.ctypes.data_as(ctypes.POINTER(ctypes.c_ulonglong)),
            table_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            table_size, max_vals_per_key,
            anchor_rp.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            anchor_fp.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            anchor_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            n_reads, max_mins, max_anchors, block_size,
        )
        if ret != 0:
            raise RuntimeError(f"match_hash_table kernel failed with code {ret}")
        return anchor_rp, anchor_fp, anchor_counts


# ---------------------------------------------------------------------------
# CPU fallback (pure NumPy)
# ---------------------------------------------------------------------------
class CPUAligner:
    """CPU-based Smith-Waterman alignment fallback (Gotoh affine gap)."""

    SCORE_MATRIX: dict = {
        ('A','A'): 2, ('A','C'):-3, ('A','G'):-1, ('A','T'):-3, ('A','N'):-1,
        ('C','A'):-3, ('C','C'): 2, ('C','G'):-3, ('C','T'):-1, ('C','N'):-1,
        ('G','A'):-1, ('G','C'):-3, ('G','G'): 2, ('G','T'):-3, ('G','N'):-1,
        ('T','A'):-3, ('T','C'):-1, ('T','G'):-3, ('T','T'): 2, ('T','N'):-1,
        ('N','A'):-1, ('N','C'):-1, ('N','G'):-1, ('N','T'):-1, ('N','N'): 0,
    }

    @staticmethod
    def sw_align(
        read: str,
        ref: str,
        gap_open: int = 5,
        gap_extend: int = 2,
    ) -> Tuple[float, int, int, int, int]:
        """Smith-Waterman with affine gaps (Gotoh) for a single read.

        Returns: (score, read_start, read_end, ref_start, ref_end)
        """
        n, m = len(read), len(ref)
        # Use python ints; for large sequences this is slow but serves as fallback
        M  = [[0] * (m + 1) for _ in range(n + 1)]
        Ix = [[0] * (m + 1) for _ in range(n + 1)]  # gap in read (vertical)
        Iy = [[0] * (m + 1) for _ in range(n + 1)]  # gap in ref (horizontal)

        best_score = 0
        best_i = best_j = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                s = CPUAligner.SCORE_MATRIX.get(
                    (read[i-1].upper(), ref[j-1].upper()), -1
                )

                # M: diagonal
                diag_best = M[i-1][j-1]
                if Ix[i-1][j-1] > diag_best: diag_best = Ix[i-1][j-1]
                if Iy[i-1][j-1] > diag_best: diag_best = Iy[i-1][j-1]
                M[i][j] = max(0, diag_best + s)

                # Ix: from above
                Ix[i][j] = max(0, M[i-1][j] - gap_open, Ix[i-1][j] - gap_extend)

                # Iy: from left
                Iy[i][j] = max(0, M[i][j-1] - gap_open, Iy[i][j-1] - gap_extend)

                cell_max = max(M[i][j], Ix[i][j], Iy[i][j])
                if cell_max > best_score:
                    best_score = cell_max
                    best_i, best_j = i, j

        if best_score == 0:
            return 0.0, 0, 0, 0, 0

        # Approximate alignment bounds by scanning back for positive scores
        r_start, r_end = best_i, 0
        f_start, f_end = best_j, 0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if max(M[i][j], Ix[i][j], Iy[i][j]) > 0:
                    if i < r_start: r_start = i
                    if j < f_start: f_start = j
                    if i > r_end:   r_end = i
                    if j > f_end:   f_end = j

        return float(best_score), r_start - 1, r_end, f_start - 1, f_end

    @staticmethod
    def align_batch(
        batch: AlignBatch,
        gap_open: int = 5,
        gap_extend: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run Smith-Waterman on all reads in a batch (CPU)."""
        n = len(batch.reads)
        scores     = np.zeros(n, dtype=np.float32)
        read_start = np.zeros(n, dtype=np.int32)
        read_end   = np.zeros(n, dtype=np.int32)
        ref_start  = np.zeros(n, dtype=np.int32)
        ref_end    = np.zeros(n, dtype=np.int32)

        for idx, read in enumerate(batch.reads):
            sc, rs, re, fs, fe = CPUAligner.sw_align(
                read, batch.ref, gap_open, gap_extend
            )
            scores[idx]     = sc
            read_start[idx] = rs
            read_end[idx]   = re
            ref_start[idx]  = fs
            ref_end[idx]    = fe

        return scores, read_start, read_end, ref_start, ref_end


# ---------------------------------------------------------------------------
# Main GPU worker entry point
# ---------------------------------------------------------------------------
_launcher: Optional[CUDALauncher] = None


def _get_launcher() -> CUDALauncher:
    global _launcher
    if _launcher is None:
        _launcher = CUDALauncher()
    return _launcher


def gpu_worker(
    batch: AlignBatch,
    band_width: int = 50,
    gap_open: int = 5,
    gap_extend: int = 2,
    with_bounds: bool = True,
) -> AlignResult:
    """Main GPU worker: Smith-Waterman affine-gap alignment of a batch.

    Uses CUDA kernels when available, falls back to CPU Gotoh implementation.

    Args:
        batch: AlignBatch with reads and reference.
        band_width: Half-band width for banded DP.
        gap_open: Gap opening penalty.
        gap_extend: Gap extension penalty.
        with_bounds: If True, return alignment coordinates (slower).

    Returns:
        AlignResult with scores and optional alignment bounds.
    """
    start = time.perf_counter()
    n = len(batch.reads)
    read_len = batch.read_len
    ref_bytes = batch.ref.encode()

    # Pack reads into contiguous bytes array
    packed = bytearray(n * read_len)
    for i, read in enumerate(batch.reads):
        r = read.ljust(read_len, 'N')[:read_len]
        packed[i * read_len:(i + 1) * read_len] = r.encode()

    launcher = _get_launcher()

    if launcher.available:
        try:
            if with_bounds:
                scores, rs, re, fs, fe = launcher.sw_affine(
                    bytes(packed), ref_bytes,
                    n, read_len, len(batch.ref),
                    band_width, gap_open, gap_extend,
                )
                kernel_used = "sw_affine"
            else:
                scores = launcher.sw_score_only(
                    bytes(packed), ref_bytes,
                    n, read_len, len(batch.ref),
                    band_width, gap_open, gap_extend,
                )
                rs = re = fs = fe = None
                kernel_used = "sw_score_only"
        except Exception as e:
            log("cuda_kernel_fallback", error=str(e))
            scores, rs, re, fs, fe = CPUAligner.align_batch(
                batch, gap_open, gap_extend,
            )
            kernel_used = "cpu_fallback"
    else:
        scores, rs, re, fs, fe = CPUAligner.align_batch(
            batch, gap_open, gap_extend,
        )
        kernel_used = "cpu"

    elapsed = (time.perf_counter() - start) * 1000.0

    log("sw_batch_done",
        batch_id=batch.batch_id,
        reads=n,
        time_ms=round(elapsed, 2),
        kernel=kernel_used,
        band_width=band_width,
        gap_open=gap_open,
        gap_extend=gap_extend,
    )

    return AlignResult(
        batch_id=batch.batch_id,
        scores=scores,
        read_start=rs,
        read_end=re,
        ref_start=fs,
        ref_end=fe,
        elapsed_ms=elapsed,
        n_reads=n,
        kernel_used=kernel_used,
    )

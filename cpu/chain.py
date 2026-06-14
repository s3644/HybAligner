"""CPU chaining — anchor-based read-to-reference chaining logic.

Implements Minimap2-style chaining of seed matches (anchors)
to produce approximate alignment coordinates. Runs on CPU as
a complement to GPU alignment kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(order=True)
class Anchor:
    """A seed match (anchor) between a read and reference."""
    read_pos: int
    ref_pos: int
    length: int = 15  # k-mer length

    @property
    def diag(self) -> int:
        """Diagonal in the DP matrix: ref_pos - read_pos."""
        return self.ref_pos - self.read_pos

    @property
    def score(self) -> float:
        """Anchor score based on k-mer length."""
        return float(self.length)


@dataclass
class Chain:
    """A chain of colinear anchors forming an approximate alignment."""
    anchors: List[Anchor] = field(default_factory=list)
    score: float = 0.0
    read_start: int = 0
    read_end: int = 0
    ref_start: int = 0
    ref_end: int = 0

    def add_anchor(self, a: Anchor, gap_score: float = 0.0):
        """Add an anchor to the chain, updating bounds and score."""
        if not self.anchors:
            self.read_start = a.read_pos
            self.read_end = a.read_pos + a.length
            self.ref_start = a.ref_pos
            self.ref_end = a.ref_pos + a.length
        else:
            self.read_start = min(self.read_start, a.read_pos)
            self.read_end = max(self.read_end, a.read_pos + a.length)
            self.ref_start = min(self.ref_start, a.ref_pos)
            self.ref_end = max(self.ref_end, a.ref_pos + a.length)

        self.anchors.append(a)
        self.score += a.score + gap_score

    @property
    def num_anchors(self) -> int:
        return len(self.anchors)

    @property
    def read_span(self) -> int:
        return self.read_end - self.read_start

    @property
    def ref_span(self) -> int:
        return self.ref_end - self.ref_start


# ---------------------------------------------------------------------------
# Anchor extraction (simple k-mer matching without GPU)
# ---------------------------------------------------------------------------
KMER_SIZE = 15
WINDOW_SIZE = 10


def _canonical_kmer(kmer: str) -> str:
    """Return the canonical (lexicographically smaller) of k-mer and its
    reverse complement."""
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    revcomp = ''.join(comp.get(b, 'N') for b in reversed(kmer))
    return kmer if kmer <= revcomp else revcomp


def extract_anchors(
    read: str,
    ref: str,
    k: int = KMER_SIZE,
    w: int = WINDOW_SIZE,
) -> List[Anchor]:
    """Extract minimizer-based anchors between a single read and reference.

    Uses minimizer (k,w) scheme: for each window of w consecutive k-mers,
    select the one with the minimum hash as the minimizer.
    """
    # Build reference minimizer index
    ref_mins: dict = {}  # kmer_hash -> list of positions
    prev_min_hash = None
    for win_start in range(len(ref) - k - w + 2):
        min_hash = None
        min_pos = -1
        for offset in range(w):
            pos = win_start + offset
            if pos + k > len(ref):
                break
            kmer = ref[pos:pos + k]
            h = hash(_canonical_kmer(kmer))
            if min_hash is None or h < min_hash:
                min_hash = h
                min_pos = pos
        if min_hash is not None and min_hash != prev_min_hash:
            ref_mins.setdefault(min_hash, []).append(min_pos)
            prev_min_hash = min_hash

    # Extract read minimizers and match
    anchors: List[Anchor] = []
    prev_min_hash = None
    for win_start in range(len(read) - k - w + 2):
        min_hash = None
        min_pos = -1
        for offset in range(w):
            pos = win_start + offset
            if pos + k > len(read):
                break
            kmer = read[pos:pos + k]
            h = hash(_canonical_kmer(kmer))
            if min_hash is None or h < min_hash:
                min_hash = h
                min_pos = pos
        if min_hash is not None and min_hash != prev_min_hash:
            if min_hash in ref_mins:
                for ref_pos in ref_mins[min_hash]:
                    anchors.append(Anchor(read_pos=min_pos, ref_pos=ref_pos, length=k))
            prev_min_hash = min_hash

    return anchors


# ---------------------------------------------------------------------------
# Chaining algorithm (Minimap2-style 1D DP chaining)
# ---------------------------------------------------------------------------
def chain_anchors(
    anchors: List[Anchor],
    max_gap: int = 5000,
    bandwidth: int = 500,
) -> Chain:
    """Chain anchors using a 1D DP over diagonals.

    Implements the Minimap2 chaining algorithm:
    For each anchor, find the best predecessor within max_gap and
    consistent diagonal ordering.
    """
    if not anchors:
        return Chain()

    # Sort by reference position
    sorted_anchors = sorted(anchors, key=lambda a: a.ref_pos)
    n = len(sorted_anchors)

    # DP: best chain score ending at each anchor
    dp = [a.score for a in sorted_anchors]
    prev = [-1] * n
    best_idx = 0  # index of best chain

    for i in range(n):
        ai = sorted_anchors[i]
        # Look back within bandwidth
        j_start = max(0, i - bandwidth - 1)
        for j in range(i - 1, j_start - 1, -1):
            aj = sorted_anchors[j]

            # Gap in reference
            ref_gap = ai.ref_pos - (aj.ref_pos + aj.length)
            if ref_gap < 0:
                continue
            if ref_gap > max_gap:
                break  # anchors sorted by ref_pos, gaps only increase

            # Gap in read
            read_gap = ai.read_pos - (aj.read_pos + aj.length)

            # Colinearity check
            if read_gap < 0 or read_gap > max_gap:
                continue

            # Gap penalty (log affine)
            gap_penalty = _gap_penalty(ref_gap, read_gap)
            candidate = dp[j] + ai.score - gap_penalty

            if candidate > dp[i]:
                dp[i] = candidate
                prev[i] = j

        if dp[i] > dp[best_idx]:
            best_idx = i

    # Traceback
    chain = Chain()
    idx = best_idx
    while idx >= 0:
        chain.anchors.insert(0, sorted_anchors[idx])
        idx = prev[idx]

    chain.score = dp[best_idx]
    # Compute bounds
    if chain.anchors:
        chain.read_start = chain.anchors[0].read_pos
        chain.read_end = chain.anchors[-1].read_pos + chain.anchors[-1].length
        chain.ref_start = chain.anchors[0].ref_pos
        chain.ref_end = chain.anchors[-1].ref_pos + chain.anchors[-1].length

    return chain


def _gap_penalty(ref_gap: int, read_gap: int, w: float = 0.01) -> float:
    """Log-affine gap penalty: w * |ref_gap - read_gap| + 0.5 * log2(min(gap)+1)."""
    gap_diff = abs(ref_gap - read_gap)
    min_gap = min(ref_gap, read_gap)
    # Avoid log(0)
    log_term = math.log2(min_gap + 1) if min_gap > 0 else 0
    return w * gap_diff + 0.5 * log_term


# ---------------------------------------------------------------------------
# High-level chaining entry point
# ---------------------------------------------------------------------------
def chain_reads(
    reads: List[str],
    ref: str,
    k: int = KMER_SIZE,
    w: int = WINDOW_SIZE,
) -> List[Chain]:
    """Chain each read against the reference, returning a Chain per read."""
    chains = []
    for read in reads:
        anchors = extract_anchors(read, ref, k, w)
        chain = chain_anchors(anchors)
        chains.append(chain)
    return chains


def chain_reads_parallel(
    reads: List[str],
    ref: str,
    k: int = KMER_SIZE,
    w: int = WINDOW_SIZE,
    n_workers: int = 0,
) -> List[Chain]:
    """Parallel chain_reads using all CPU cores.

    Each read's anchor extraction + chaining is independent.
    Uses ProcessPoolExecutor to bypass the GIL.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor

    if n_workers <= 0:
        n_workers = os.cpu_count() or 4

    if len(reads) < 50:
        return chain_reads(reads, ref, k, w)

    def _chain_one(read: str) -> Chain:
        anchors = extract_anchors(read, ref, k, w)
        return chain_anchors(anchors)

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_chain_one, read) for read in reads]
        chains = []
        for fut in futures:
            try:
                chains.append(fut.result())
            except Exception:
                chains.append(Chain())
    return chains


def chain_to_alignment(chain: Chain, read_len: int, ref_len: int) -> dict:
    """Convert a chain to an approximate alignment summary."""
    if not chain.anchors:
        return {
            "mapped": False,
            "score": 0.0,
            "n_anchors": 0,
        }

    return {
        "mapped": True,
        "score": chain.score,
        "n_anchors": chain.num_anchors,
        "read_start": chain.read_start,
        "read_end": chain.read_end,
        "ref_start": chain.ref_start,
        "ref_end": chain.ref_end,
        "read_cov": (chain.read_end - chain.read_start) / read_len if read_len else 0,
        "ref_cov": (chain.ref_end - chain.ref_start) / ref_len if ref_len else 0,
    }

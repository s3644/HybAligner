"""CIGAR string generation — traceback from Smith-Waterman alignment bounds.

Given alignment boundaries (read_start/end, ref_start/end) and sequences,
performs a banded traceback through the Gotoh DP to reconstruct the CIGAR
string (compact representation of match/insert/delete operations).

CIGAR format: e.g., "5M2I3M1D10M" = 5 matches, 2 insertions, 3 matches,
1 deletion, 10 matches.
"""

from __future__ import annotations

from typing import List, Tuple


# ---------------------------------------------------------------------------
# Scoring matrix (same as CUDA kernel + CPUAligner)
# ---------------------------------------------------------------------------
SCORE_MATRIX: dict = {
    ('A','A'): 2, ('A','C'):-3, ('A','G'):-1, ('A','T'):-3, ('A','N'):-1,
    ('C','A'):-3, ('C','C'): 2, ('C','G'):-3, ('C','T'):-1, ('C','N'):-1,
    ('G','A'):-1, ('G','C'):-3, ('G','G'): 2, ('G','T'):-3, ('G','N'):-1,
    ('T','A'):-3, ('T','C'):-1, ('T','G'):-3, ('T','T'): 2, ('T','N'):-1,
    ('N','A'):-1, ('N','C'):-1, ('N','G'):-1, ('N','T'):-1, ('N','N'): 0,
}

# ---------------------------------------------------------------------------
# CIGAR operation encoding
# ---------------------------------------------------------------------------
CIGAR_MATCH    = 0  # M
CIGAR_INSERT   = 1  # I — base in read, gap in ref
CIGAR_DELETE   = 2  # D — gap in read, base in ref
CIGAR_SOFTCLIP = 3  # S — soft-clipped (outside alignment bounds)

GOTOH_M = 0  # match/mismatch state
GOTOH_X = 1  # gap-in-read (insertion in CIGAR terms = vertical gap)
GOTOH_Y = 2  # gap-in-ref (deletion in CIGAR terms = horizontal gap)


def _score(r_char: str, f_char: str) -> int:
    return SCORE_MATRIX.get((r_char.upper(), f_char.upper()), -1)


# ---------------------------------------------------------------------------
# CIGAR traceback
# ---------------------------------------------------------------------------
def traceback_cigar(
    read: str,
    ref: str,
    read_start: int,
    read_end: int,
    ref_start: int,
    ref_end: int,
    gap_open: int = 5,
    gap_extend: int = 2,
) -> str:
    """Reconstruct CIGAR string via banded Gotoh DP traceback.

    Args:
        read: Full read sequence.
        ref: Full reference sequence.
        read_start, read_end: Alignment bounds in read (0-based, exclusive end).
        ref_start, ref_end: Alignment bounds in ref (0-based, exclusive end).
        gap_open, gap_extend: Gap penalties (must match forward pass).

    Returns:
        CIGAR string, e.g. "10M2I5M" or "*" if no alignment.
    """
    if read_start >= read_end or ref_start >= ref_end:
        return "*"

    # Extract the aligned subsequences
    sub_read = read[read_start:read_end]
    sub_ref  = ref[ref_start:ref_end]
    n, m = len(sub_read), len(sub_ref)

    if n == 0 or m == 0:
        return "*"

    # Full Gotoh DP (small region, O(nm) is fine)
    # M[i][j] = best ending in match at (i,j)
    # X[i][j] = best ending in gap-in-read (vertical)
    # Y[i][j] = best ending in gap-in-ref (horizontal)
    M  = [[0] * (m + 1) for _ in range(n + 1)]
    X  = [[0] * (m + 1) for _ in range(n + 1)]
    Y  = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[-1] * (m + 1) for _ in range(n + 1)]  # backpointer: 0=M,1=X,2=Y
    st = [[0] * (m + 1) for _ in range(n + 1)]   # which state: 0=M,1=X,2=Y

    best_score = -1
    best_i = best_j = 0
    best_state = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = _score(sub_read[i-1], sub_ref[j-1])

            # M(i,j): from diag
            from_M = M[i-1][j-1]
            from_X = X[i-1][j-1]
            from_Y = Y[i-1][j-1]
            best = max(from_M, from_X, from_Y)
            M[i][j] = max(0, best + s)

            if M[i][j] > 0:
                if best == from_M:
                    bt[i][j] = GOTOH_M
                elif best == from_X:
                    bt[i][j] = GOTOH_X
                else:
                    bt[i][j] = GOTOH_Y
            st[i][j] = GOTOH_M

            # X(i,j): gap in read (vertical)
            from_Mx = M[i-1][j] - gap_open
            from_Xx = X[i-1][j] - gap_extend
            X[i][j] = max(0, from_Mx, from_Xx)

            # Y(i,j): gap in ref (horizontal)
            from_My = M[i][j-1] - gap_open
            from_Yy = Y[i][j-1] - gap_extend
            Y[i][j] = max(0, from_My, from_Yy)

            # Track best overall
            for val, s_type in [(M[i][j], GOTOH_M), (X[i][j], GOTOH_X), (Y[i][j], GOTOH_Y)]:
                if val > best_score:
                    best_score = val
                    best_i, best_j = i, j
                    best_state = s_type

    if best_score <= 0:
        return "*"

    # Traceback from best cell
    cigar_ops: List[Tuple[str, int]] = []  # (op, count)
    i, j = best_i, best_j
    state = best_state

    while i > 0 and j > 0:
        s = _score(sub_read[i-1], sub_ref[j-1])

        if state == GOTOH_M:
            # Coming from diagonal — must be a match/mismatch
            cigar_ops.append(('M', 1))
            # Determine predecessor state from backpointer
            state = bt[i][j]
            i -= 1
            j -= 1

        elif state == GOTOH_X:
            # Gap in read = insertion in CIGAR (read has base, ref doesn't)
            cigar_ops.append(('I', 1))
            # Predecessor: check M[i-1][j] vs X[i-1][j]
            from_Mx = M[i-1][j] - gap_open
            from_Xx = X[i-1][j] - gap_extend
            if from_Mx >= from_Xx:
                state = GOTOH_M
            else:
                state = GOTOH_X
            i -= 1

        elif state == GOTOH_Y:
            # Gap in ref = deletion in CIGAR (ref has base, read doesn't)
            cigar_ops.append(('D', 1))
            from_My = M[i][j-1] - gap_open
            from_Yy = Y[i][j-1] - gap_extend
            if from_My >= from_Yy:
                state = GOTOH_M
            else:
                state = GOTOH_Y
            j -= 1

        else:
            break

    # Add soft-clipping for the regions before/after alignment
    if read_start > 0:
        cigar_ops.append(('S', read_start))
    # Note: soft-clip at end is handled after reversing

    # Reverse to get 5'→3' order
    cigar_ops.reverse()

    # Soft-clip at 3' end
    remaining = len(read) - read_end
    if remaining > 0:
        cigar_ops.append(('S', remaining))

    # Compress runs of same operation
    compressed: List[Tuple[str, int]] = []
    for op, count in cigar_ops:
        if compressed and compressed[-1][0] == op:
            compressed[-1] = (op, compressed[-1][1] + count)
        else:
            compressed.append((op, count))

    return ''.join(f"{cnt}{op}" for op, cnt in compressed)


def batch_traceback_cigar(
    reads: List[str],
    ref: str,
    read_starts: 'np.ndarray',
    read_ends: 'np.ndarray',
    ref_starts: 'np.ndarray',
    ref_ends: 'np.ndarray',
    gap_open: int = 5,
    gap_extend: int = 2,
) -> List[str]:
    """Generate CIGAR strings for a batch of aligned reads.

    Args:
        reads: List of read sequences.
        ref: Reference sequence.
        read_starts, read_ends: Alignment bounds per read (numpy arrays).
        ref_starts, ref_ends: Alignment bounds per read (numpy arrays).
        gap_open, gap_extend: Gap penalties.

    Returns:
        List of CIGAR strings, one per read.
    """
    cigars = []
    for i, read in enumerate(reads):
        rs = int(read_starts[i])
        re = int(read_ends[i])
        fs = int(ref_starts[i])
        fe = int(ref_ends[i])

        if rs >= re or fs >= fe:
            cigars.append("*")
        else:
            cigars.append(traceback_cigar(
                read, ref, rs, re, fs, fe, gap_open, gap_extend,
            ))
    return cigars


def batch_traceback_cigar_parallel(
    reads: List[str],
    ref: str,
    read_starts: 'np.ndarray',
    read_ends: 'np.ndarray',
    ref_starts: 'np.ndarray',
    ref_ends: 'np.ndarray',
    gap_open: int = 5,
    gap_extend: int = 2,
    n_workers: int = 0,
) -> List[str]:
    """Parallel CIGAR traceback using all CPU cores.

    Each read's CIGAR is computed independently — perfect for
    embarrassingly parallel execution. Uses ProcessPoolExecutor
    to bypass the GIL.

    Args:
        n_workers: Number of processes (0 = auto = cpu_count).
    """
    import os
    from concurrent.futures import ProcessPoolExecutor

    if n_workers <= 0:
        n_workers = os.cpu_count() or 4

    # Build task list: (read, rs, re, fs, fe, gap_open, gap_extend)
    tasks = []
    for i, read in enumerate(reads):
        rs = int(read_starts[i])
        re = int(read_ends[i])
        fs = int(ref_starts[i])
        fe = int(ref_ends[i])
        if rs < re and fs < fe:
            tasks.append((read, ref, rs, re, fs, fe, gap_open, gap_extend))
        else:
            tasks.append(None)  # sentinel for no-alignment

    # Process small batches serially (overhead not worth it)
    if len(reads) < 200:
        return batch_traceback_cigar(
            reads, ref, read_starts, read_ends, ref_starts, ref_ends,
            gap_open, gap_extend,
        )

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        # Submit only the non-None tasks
        futures = {}
        result_idx = {}
        next_idx = 0
        for i, task in enumerate(tasks):
            if task is not None:
                fut = ex.submit(traceback_cigar, *task)
                futures[fut] = i

        # Collect results
        results = ["*"] * len(reads)
        for fut in futures:
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception:
                results[idx] = "*"

    return results


# ---------------------------------------------------------------------------
# CIGAR stats
# ---------------------------------------------------------------------------
def cigar_stats(cigar: str) -> dict:
    """Parse a CIGAR string and return operation counts and aligned length."""
    import re
    if cigar == "*":
        return {
            "cigar": "*",
            "matches": 0, "insertions": 0, "deletions": 0,
            "soft_clips": 0, "aligned_bases": 0, "total_ops": 0,
        }

    stats = {"cigar": cigar, "matches": 0, "insertions": 0,
             "deletions": 0, "soft_clips": 0, "aligned_bases": 0}
    total_ops = 0

    for match in re.finditer(r'(\d+)([MIDNSHP=X])', cigar):
        count = int(match.group(1))
        op = match.group(2)
        total_ops += 1

        if op == 'M':
            stats["matches"] += count
            stats["aligned_bases"] += count
        elif op == 'I':
            stats["insertions"] += count
            stats["aligned_bases"] += count
        elif op == 'D':
            stats["deletions"] += count
        elif op == 'S':
            stats["soft_clips"] += count

    stats["total_ops"] = total_ops
    return stats

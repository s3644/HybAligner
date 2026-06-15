#!/usr/bin/env python3
"""Benchmark HybAligner vs Minimap2 on real FASTQ data.

Reads: SRR077487 (human exome, paired-end Illumina)
Reference: chr21 (hg38)

Usage:
    python benchmark/bench_real.py --fastq data/SRR077487_1.fastq.gz --ref data/chr21.fa.gz
    python benchmark/bench_real.py --fastq data/SRR077487_1.fastq.gz --ref data/chr21.fa.gz --max-reads 10000
"""

from __future__ import annotations

import argparse
import gzip
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpu.fast_align import FastPipeline


def count_lines_gz(path: str) -> int:
    """Count lines in a gzipped file."""
    count = 0
    with gzip.open(path, 'rt') as f:
        for _ in f:
            count += 1
    return count


def parse_fastq_gz(path: str, max_reads: int = 0) -> Tuple[List[str], int]:
    """Parse gzipped FASTQ, returning reads and total line count.

    If max_reads > 0, only return first max_reads reads.
    """
    reads = []
    with gzip.open(path, 'rt') as f:
        for i, line in enumerate(f):
            if i % 4 == 1:  # sequence line
                reads.append(line.strip())
                if max_reads > 0 and len(reads) >= max_reads:
                    break
    return reads, (i + 1)


def write_fastq(reads: List[str], path: str):
    """Write reads to uncompressed FASTQ."""
    with open(path, 'w') as f:
        for i, read in enumerate(reads):
            f.write(f"@read_{i}\n{read}\n+\n{'I' * len(read)}\n")


def parse_fasta_gz(path: str) -> str:
    """Parse gzipped FASTA, return concatenated sequence."""
    seq_parts = []
    with gzip.open(path, 'rt') as f:
        for line in f:
            if not line.startswith('>'):
                seq_parts.append(line.strip())
    return ''.join(seq_parts)


def run_minimap2(fastq_path: str, ref_path: str, n_reads: int, threads: int = 16) -> dict:
    """Run minimap2 and return timing/results."""
    print(f"  Running minimap2 (-t {threads})...")
    t0 = time.perf_counter()
    result = subprocess.run(
        ["minimap2", "-c", "-t", str(threads), ref_path, fastq_path],
        capture_output=True, text=True, timeout=600,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    n_aligned = 0
    scores = []
    for line in result.stdout.strip().split('\n'):
        if not line:
            continue
        fields = line.split('\t')
        if len(fields) >= 10:
            n_aligned += 1
            try:
                matches = int(fields[9])
                read_len = len(fields[9]) if fields[9].isdigit() else 150
                score = matches * 2 - (int(fields[1]) - matches) * 3
                scores.append(float(score))
            except (ValueError, IndexError):
                scores.append(0.0)

    return {
        "tool": "minimap2",
        "threads": threads,
        "elapsed_ms": round(elapsed_ms, 1),
        "reads_per_sec": round(n_reads / (elapsed_ms / 1000.0), 1),
        "n_aligned": n_aligned,
        "score_mean": round(float(np.mean(scores)), 2) if scores else 0,
    }


def run_hybaligner(fastq_path: str, ref_path: str) -> dict:
    """Run HybAligner FastPipeline."""
    print("  Running HybAligner FastPipeline...")
    fp = FastPipeline()
    t0 = time.perf_counter()
    result = fp.run(fastq_path, ref_path)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    fp.free()
    return {
        "tool": "hybaligner",
        "elapsed_ms": round(elapsed_ms, 1),
        "reads_per_sec": result["throughput_reads_per_sec"],
        "n_aligned": result["n_aligned"],
        "score_mean": result["score_mean"],
        "kernel_ms": result.get("align_ms", 0),
        "parse_ms": result.get("parse_ms", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark on real FASTQ data")
    parser.add_argument("--fastq", required=True, help="Input FASTQ (gzipped)")
    parser.add_argument("--ref", required=True, help="Reference FASTA (gzipped)")
    parser.add_argument("--max-reads", type=int, default=0,
                        help="Max reads to use (0=all)")
    parser.add_argument("--threads", type=int, default=16,
                        help="Minimap2 threads (default: 16)")
    args = parser.parse_args()

    print(f"\n{'='*72}")
    print(f"  HybAligner vs Minimap2 — Real Data Benchmark")
    print(f"{'='*72}")
    print(f"  FASTQ:  {args.fastq}")
    print(f"  Ref:    {args.ref}")
    print()

    # Parse reference
    print("Loading reference...")
    t0 = time.perf_counter()
    ref_seq = parse_fasta_gz(args.ref)
    ref_load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  Reference: {len(ref_seq):,} bp ({ref_load_ms:.0f} ms)")
    print()

    # Parse reads
    print("Loading reads...")
    t0 = time.perf_counter()
    reads, total_lines = parse_fastq_gz(args.fastq, args.max_reads)
    read_load_ms = (time.perf_counter() - t0) * 1000.0
    n_reads = len(reads)
    read_len = max(len(r) for r in reads) if reads else 0
    print(f"  Reads: {n_reads:,} × {read_len}bp avg ({read_load_ms:.0f} ms)")
    print()

    # Write uncompressed temp files (minimap2 needs uncompressed)
    with tempfile.TemporaryDirectory() as tmpdir:
        fastq_tmp = os.path.join(tmpdir, "reads.fastq")
        fasta_tmp = os.path.join(tmpdir, "ref.fasta")

        print("Writing temp files...")
        write_fastq(reads, fastq_tmp)
        with open(fasta_tmp, 'w') as f:
            f.write(f">chr21\n")
            for i in range(0, len(ref_seq), 80):
                f.write(ref_seq[i:i+80] + "\n")
        print(f"  FASTQ: {os.path.getsize(fastq_tmp)/1024/1024:.1f} MB")
        print(f"  FASTA: {os.path.getsize(fasta_tmp)/1024/1024:.1f} MB")
        print()

        results = []

        # --- Minimap2 ---
        try:
            r = run_minimap2(fastq_tmp, fasta_tmp, n_reads, args.threads)
            results.append(r)
            print(f"  Result: {r['reads_per_sec']:,.0f} reads/s, "
                  f"{r['n_aligned']}/{n_reads} aligned, "
                  f"{r['elapsed_ms']:.0f} ms")
        except Exception as e:
            print(f"  minimap2 FAILED: {e}")

        print()

        # --- HybAligner ---
        try:
            r = run_hybaligner(fastq_tmp, fasta_tmp)
            results.append(r)
            print(f"  Result: {r['reads_per_sec']:,.0f} reads/s, "
                  f"{r['n_aligned']}/{n_reads} aligned, "
                  f"{r['elapsed_ms']:.0f} ms "
                  f"(kernel: {r['kernel_ms']:.1f}ms, parse: {r['parse_ms']:.1f}ms)")
        except Exception as e:
            print(f"  HybAligner FAILED: {e}")
            import traceback
            traceback.print_exc()

        # Report
        print(f"\n{'='*72}")
        print(f"  Benchmark Results ({n_reads:,} reads, {len(ref_seq):,}bp ref)")
        print(f"{'='*72}")
        print(f"{'Tool':<20} {'Time(ms)':>10} {'Reads/s':>12} {'Aligned':>10} {'Score':>10}")
        print(f"{'-'*62}")
        for r in results:
            print(f"{r['tool']:<20} {r['elapsed_ms']:>10.1f} {r['reads_per_sec']:>12,.1f} "
                  f"{r['n_aligned']:>10} {r['score_mean']:>10.1f}")

        if len(results) == 2:
            speedup = results[0]['reads_per_sec'] / max(results[1]['reads_per_sec'], 1)
            if speedup > 1:
                print(f"\n  minimap2 is {speedup:.1f}× faster than HybAligner")
            else:
                print(f"\n  HybAligner is {1/speedup:.1f}× faster than minimap2")

        print(f"{'='*72}\n")


if __name__ == "__main__":
    main()

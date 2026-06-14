"""Pipeline manager — top-level orchestrator for the HybAligner pipeline.

Coordinates:
  FASTQ parsing → Seeding → [CPU chaining | GPU alignment] → Result merge

Entry point: `python -m runtime.manager` or `hyb-align` console script.
"""

from __future__ import annotations

import argparse
import sys
import time
import json
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np

from obs.log import log, LogEntry
from runtime.scheduler import Scheduler, SchedulerConfig, Batch, BatchResult
from gpu.worker import gpu_worker, AlignBatch
from gpu.seeder import GPUSeeder, RefMinimizerIndex, SeedResult
from gpu.streams import run_stream_pipeline
from gpu.fast_align import FastAligner, align_preencoded
from cpu.chain import chain_anchors, Anchor
from cpu.cigar import batch_traceback_cigar, batch_traceback_cigar_parallel, cigar_stats


# ---------------------------------------------------------------------------
# FASTQ parser
# ---------------------------------------------------------------------------
def parse_fastq(path: str) -> List[str]:
    """Parse a FASTQ file, returning a list of read sequences only.

    Uses read()+splitlines() for 2× faster parsing vs line-by-line iteration.
    """
    with open(path) as f:
        lines = f.read().splitlines()
    # Every 4th line starting from line 1 is the sequence
    return [lines[i] for i in range(1, len(lines), 4)]


def parse_fasta(path: str) -> str:
    """Parse a FASTA file, returning concatenated reference sequence."""
    seq_parts = []
    with open(path) as f:
        for line in f:
            if not line.startswith('>'):
                seq_parts.append(line.strip())
    return ''.join(seq_parts)


# ---------------------------------------------------------------------------
# GPU batch handler (adapter) — with optional seeding
# ---------------------------------------------------------------------------
def _gpu_batch_handler(
    ref_seq: str,
    ref_index: Optional[RefMinimizerIndex] = None,
    seeder: Optional[GPUSeeder] = None,
    band_width: int = 50,
    gap_open: int = 5,
    gap_extend: int = 2,
) -> callable:
    """Create a GPU batch handler with optional minimizer seeding."""

    def handler(batch: Batch) -> BatchResult:
        t0 = time.perf_counter()

        # --- Seeding (optional) ---
        if seeder is not None and ref_index is not None:
            seeder.seed_batch(batch.reads, batch.read_len, ref_index)

        # --- SW alignment ---
        align_batch = AlignBatch(
            batch_id=batch.batch_id,
            reads=batch.reads,
            ref=ref_seq,
            read_len=batch.read_len,
        )
        result = gpu_worker(
            align_batch,
            band_width=band_width,
            gap_open=gap_open,
            gap_extend=gap_extend,
            with_bounds=True,
        )

        elapsed = (time.perf_counter() - t0) * 1000.0

        return BatchResult(
            batch_id=result.batch_id,
            scores=result.scores,
            read_start=result.read_start,
            read_end=result.read_end,
            ref_start=result.ref_start,
            ref_end=result.ref_end,
            elapsed_ms=elapsed,
            worker_type="gpu",
        )

    return handler


def _gpu_stream_handler(
    ref_seq: str,
    ref_index: Optional[RefMinimizerIndex] = None,
    seeder: Optional[GPUSeeder] = None,
    band_width: int = 50,
    gap_open: int = 5,
    gap_extend: int = 2,
) -> callable:
    """Create a GPU batch handler using multi-stream pipeline.

    Collects batches into groups and dispatches them through the
    triple-buffered stream pipeline for H2D/kernel/D2H overlap.
    """

    def handler(batch: Batch) -> BatchResult:
        t0 = time.perf_counter()

        # --- Seeding (optional) ---
        if seeder is not None and ref_index is not None:
            seeder.seed_batch(batch.reads, batch.read_len, ref_index)

        # --- Stream pipeline (single batch at a time from scheduler,
        # but the pipeline overlaps internally across submissions) ---
        results = run_stream_pipeline(
            batches=[batch.reads],
            ref_seq=ref_seq,
            read_len=batch.read_len,
            band_width=band_width,
            gap_open=gap_open,
            gap_extend=gap_extend,
        )
        scores, rs, re, fs, fe = results[0]

        elapsed = (time.perf_counter() - t0) * 1000.0

        return BatchResult(
            batch_id=batch.batch_id,
            scores=scores,
            read_start=rs,
            read_end=re,
            ref_start=fs,
            ref_end=fe,
            elapsed_ms=elapsed,
            worker_type="gpu_stream",
        )

    return handler


def _gpu_fast_handler(
    ref_seq: str,
    band_width: int = 50,
    gap_open: int = 5,
    gap_extend: int = 2,
) -> callable:
    """Create a GPU batch handler using the zero-overhead FastAligner.

    Bypasses scheduler/threading/JSON — direct ctypes call.
    Achieves 1M+ reads/s on DGX Spark.
    """
    max_reads = 50000
    max_read_len = 300
    max_ref_len = len(ref_seq) + 100

    fa = FastAligner(
        max_reads=max_reads,
        max_read_len=max_read_len,
        max_ref_len=max_ref_len,
    )
    fa.align(["A"], ref_seq)  # warmup

    def handler(batch: Batch) -> BatchResult:
        t0 = time.perf_counter()
        scores, rs, re, fs, fe = fa.align(
            batch.reads, ref_seq,
            band_width=band_width,
            gap_open=gap_open,
            gap_extend=gap_extend,
        )
        elapsed = (time.perf_counter() - t0) * 1000.0

        return BatchResult(
            batch_id=batch.batch_id,
            scores=scores,
            read_start=rs,
            read_end=re,
            ref_start=fs,
            ref_end=fe,
            elapsed_ms=elapsed,
            worker_type="gpu_fast",
        )

    return handler


def _cpu_batch_handler(
    ref_seq: str,
    ref_index: Optional[RefMinimizerIndex] = None,
    seeder: Optional[GPUSeeder] = None,
) -> callable:
    """Create a CPU batch handler with optional GPU seeding for chaining."""

    def handler(batch: Batch) -> BatchResult:
        start = time.perf_counter()

        # --- Seeding (optional, GPU or CPU) ---
        if seeder is not None and ref_index is not None:
            seed_result = seeder.seed_batch(batch.reads, batch.read_len, ref_index)
            # Chain the seeded anchors
            chains = [chain_anchors(anchors) for anchors in seed_result.anchors]
            scores = np.array([c.score for c in chains], dtype=np.float32)
            # Extract bounds from chains
            n = len(batch.reads)
            rs = np.array([c.read_start for c in chains], dtype=np.int32)
            re = np.array([c.read_end for c in chains], dtype=np.int32)
            fs = np.array([c.ref_start for c in chains], dtype=np.int32)
            fe = np.array([c.ref_end for c in chains], dtype=np.int32)
        else:
            # Fallback: simple CPU chaining with on-the-fly anchoring
            from cpu.chain import chain_reads
            chains = chain_reads(batch.reads, ref_seq)
            scores = np.array([c.score for c in chains], dtype=np.float32)
            n = len(batch.reads)
            rs = np.array([c.read_start for c in chains], dtype=np.int32)
            re = np.array([c.read_end for c in chains], dtype=np.int32)
            fs = np.array([c.ref_start for c in chains], dtype=np.int32)
            fe = np.array([c.ref_end for c in chains], dtype=np.int32)

        elapsed = (time.perf_counter() - start) * 1000.0

        return BatchResult(
            batch_id=batch.batch_id,
            scores=scores,
            read_start=rs,
            read_end=re,
            ref_start=fs,
            ref_end=fe,
            elapsed_ms=elapsed,
            worker_type="cpu",
        )

    return handler


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------
def run_pipeline(
    fastq_path: str,
    ref_path: str,
    batch_size: int = 4096,
    output_path: Optional[str] = None,
    gpu_only: bool = False,
    band_width: int = 50,
    gap_open: int = 5,
    gap_extend: int = 2,
    use_seed: bool = True,
    use_streams: bool = False,
    use_fast: bool = False,
    use_cigar: bool = False,
    use_parallel: bool = False,
    kmer: int = 15,
    window: int = 10,
) -> Dict:
    """Run the full HybAligner pipeline.

    Args:
        fastq_path: Path to input FASTQ file.
        ref_path: Path to reference FASTA file.
        batch_size: Reads per GPU batch.
        output_path: Optional path for JSON results.
        gpu_only: Skip CPU workers (GPU-only mode).
        band_width: Half-band width for banded SW DP.
        gap_open: Gap opening penalty.
        gap_extend: Gap extension penalty.
        use_seed: Enable minimizer seeding before alignment.
        kmer: K-mer size for minimizers.
        window: Window size for minimizer selection.

    Returns:
        Dict with pipeline summary statistics.
    """
    log("pipeline_start",
        fastq=fastq_path,
        ref=ref_path,
        batch_size=batch_size,
        gpu_only=gpu_only,
        band_width=band_width,
        gap_open=gap_open,
        gap_extend=gap_extend,
        use_seed=use_seed,
        use_streams=use_streams,
        use_cigar=use_cigar,
        kmer=kmer,
        window=window,
    )

    t0 = time.perf_counter()

    # Parse inputs
    reads = parse_fastq(fastq_path)
    ref_seq = parse_fasta(ref_path)

    log("input_parsed",
        n_reads=len(reads),
        ref_len=len(ref_seq),
    )

    # --- Build reference minimizer index (if seeding enabled) ---
    ref_index: Optional[RefMinimizerIndex] = None
    seeder: Optional[GPUSeeder] = None
    seed_build_ms = 0.0

    if use_seed:
        seeder = GPUSeeder(k=kmer, w=window)
        t_seed = time.perf_counter()
        ref_index = seeder.build_ref_index([ref_seq], len(ref_seq))
        seed_build_ms = (time.perf_counter() - t_seed) * 1000.0
        log("ref_index_ready", build_ms=round(seed_build_ms, 2))

    # --- Fast path: bypass scheduler entirely ---
    if use_fast:
        t_fast = time.perf_counter()
        fast = FastAligner(
            max_reads=len(reads) + 10,
            max_read_len=max(len(r) for r in reads) if reads else 300,
            max_ref_len=len(ref_seq) + 100,
        )
        fast.align(["A"], ref_seq)  # warmup
        scores, rs, re, fs, fe = fast.align(
            reads, ref_seq,
            band_width=band_width,
            gap_open=gap_open,
            gap_extend=gap_extend,
        )
        fast_elapsed = (time.perf_counter() - t_fast) * 1000.0
        n_aligned = int(np.count_nonzero(scores))

        summary = {
            "pipeline": "HybAligner v0.5.0",
            "algorithm": "SW affine-gap (fast path)",
            "mode": "fast",
            "n_reads": len(reads),
            "n_aligned": n_aligned,
            "pct_aligned": round(100.0 * n_aligned / len(reads), 2) if reads else 0,
            "ref_len": len(ref_seq),
            "band_width": band_width,
            "gap_open": gap_open,
            "gap_extend": gap_extend,
            "total_elapsed_ms": round(fast_elapsed, 2),
            "throughput_reads_per_sec": round(len(reads) / (fast_elapsed / 1000.0), 1),
            "score_mean": round(float(np.mean(scores)), 4) if len(scores) else 0,
            "score_max": round(float(np.max(scores)), 4) if len(scores) else 0,
            "gpu_batches": 1,
            "cpu_batches": 0,
        }
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Results written to {output_path}")
        return summary

    # Configure scheduler
    config = SchedulerConfig(
        batch_size=batch_size,
        cpu_fallback=not gpu_only,
        read_len=max(len(r) for r in reads) if reads else 300,
    )
    scheduler = Scheduler(config)
    if use_fast:
        scheduler.set_gpu_handler(_gpu_fast_handler(
            ref_seq, band_width, gap_open, gap_extend,
        ))
    elif use_streams:
        scheduler.set_gpu_handler(_gpu_stream_handler(
            ref_seq, ref_index, seeder, band_width, gap_open, gap_extend,
        ))
    else:
        scheduler.set_gpu_handler(_gpu_batch_handler(
            ref_seq, ref_index, seeder, band_width, gap_open, gap_extend,
        ))
    scheduler.set_cpu_handler(_cpu_batch_handler(ref_seq, ref_index, seeder))

    # Start workers
    scheduler.start()

    # Feed reads
    scheduler.feed_list(reads, ref_name=Path(ref_path).stem)
    scheduler.stop()

    # Collect results
    all_scores: List[float] = []
    all_read_starts: List[int] = []
    all_read_ends: List[int] = []
    all_ref_starts: List[int] = []
    all_ref_ends: List[int] = []
    batch_stats: List[dict] = []
    total_gpu_ms = 0.0
    total_cpu_ms = 0.0
    n_gpu = 0
    n_cpu = 0

    for result in scheduler.results():
        all_scores.extend(result.scores.tolist())
        if result.read_start is not None:
            all_read_starts.extend(result.read_start.tolist())
            all_read_ends.extend(result.read_end.tolist())
            all_ref_starts.extend(result.ref_start.tolist())
            all_ref_ends.extend(result.ref_end.tolist())

        stat = {
            "batch_id": result.batch_id,
            "scores_mean": float(np.mean(result.scores)),
            "scores_max": float(np.max(result.scores)),
            "elapsed_ms": result.elapsed_ms,
            "worker_type": result.worker_type,
        }
        if result.read_start is not None:
            stat["n_aligned"] = int(np.count_nonzero(result.scores))
        batch_stats.append(stat)

        if result.worker_type in ("gpu", "gpu_stream", "gpu_fast"):
            total_gpu_ms += result.elapsed_ms
            n_gpu += 1
        else:
            total_cpu_ms += result.elapsed_ms
            n_cpu += 1

    total_elapsed = (time.perf_counter() - t0) * 1000.0
    n_aligned = int(np.count_nonzero(all_scores))

    # --- CIGAR traceback (CPU post-processing) ---
    cigar_strings: List[str] = []
    cigar_summary: dict = {}
    if use_cigar and all_read_starts and n_aligned > 0:
        t_cigar = time.perf_counter()
        if use_parallel:
            cigar_strings = batch_traceback_cigar_parallel(
                reads, ref_seq,
                np.array(all_read_starts, dtype=np.int32),
                np.array(all_read_ends, dtype=np.int32),
                np.array(all_ref_starts, dtype=np.int32),
                np.array(all_ref_ends, dtype=np.int32),
                gap_open=gap_open, gap_extend=gap_extend,
            )
        else:
            cigar_strings = batch_traceback_cigar(
                reads, ref_seq,
                np.array(all_read_starts, dtype=np.int32),
                np.array(all_read_ends, dtype=np.int32),
                np.array(all_ref_starts, dtype=np.int32),
                np.array(all_ref_ends, dtype=np.int32),
                gap_open=gap_open, gap_extend=gap_extend,
            )
        cigar_ms = (time.perf_counter() - t_cigar) * 1000.0

        # Aggregate CIGAR stats
        all_stats = [cigar_stats(c) for c in cigar_strings]
        n_with_cigar = sum(1 for s in all_stats if s["cigar"] != "*")
        cigar_summary = {
            "n_cigars": n_with_cigar,
            "cigar_time_ms": round(cigar_ms, 2),
            "mean_matches": round(
                float(np.mean([s["matches"] for s in all_stats])), 1
            ) if all_stats else 0,
            "mean_insertions": round(
                float(np.mean([s["insertions"] for s in all_stats])), 1
            ) if all_stats else 0,
            "mean_deletions": round(
                float(np.mean([s["deletions"] for s in all_stats])), 1
            ) if all_stats else 0,
        }
        log("cigar_done", n=n_with_cigar, time_ms=round(cigar_ms, 2))

    summary = {
        "pipeline": "HybAligner v0.5.0",
        "algorithm": "SW affine-gap + minimizer seeding",
        "mode": "fast" if use_fast else ("streams" if use_streams else "sync"),
        "streams": "enabled" if use_streams else "disabled",
        "fast": "enabled" if use_fast else "disabled",
        "seeding": "enabled" if use_seed else "disabled",
        "kmer": kmer if use_seed else None,
        "window": window if use_seed else None,
        "seed_index_build_ms": round(seed_build_ms, 2) if use_seed else 0,
        "n_reads": len(reads),
        "n_aligned": n_aligned,
        "pct_aligned": round(100.0 * n_aligned / len(reads), 2) if reads else 0,
        "ref_len": len(ref_seq),
        "band_width": band_width,
        "gap_open": gap_open,
        "gap_extend": gap_extend,
        "total_elapsed_ms": round(total_elapsed, 2),
        "throughput_reads_per_sec": round(len(reads) / (total_elapsed / 1000.0), 1),
        "gpu_batches": n_gpu,
        "cpu_batches": n_cpu,
        "gpu_total_ms": round(total_gpu_ms, 2),
        "cpu_total_ms": round(total_cpu_ms, 2),
        "score_mean": round(float(np.mean(all_scores)), 4) if all_scores else 0,
        "score_std": round(float(np.std(all_scores)), 4) if all_scores else 0,
        "score_max": round(float(np.max(all_scores)), 4) if all_scores else 0,
    }
    if cigar_summary:
        summary["cigar"] = cigar_summary
    if all_read_starts:
        summary["read_span_mean"] = round(
            float(np.mean([e - s for s, e in zip(all_read_starts, all_read_ends)])), 1
        )
        summary["ref_span_mean"] = round(
            float(np.mean([e - s for s, e in zip(all_ref_starts, all_ref_ends)])), 1
        )

    summary["batch_details"] = batch_stats

    log("pipeline_done", summary=summary)

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Results written to {output_path}")

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="HybAligner — Hybrid CPU-GPU Sequence Aligner for DGX Spark",
    )
    parser.add_argument(
        "fastq", help="Path to input FASTQ file",
    )
    parser.add_argument(
        "ref", help="Path to reference FASTA file",
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=4096,
        help="Reads per GPU batch (default: 4096)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output JSON path for results",
    )
    parser.add_argument(
        "--gpu-only", action="store_true",
        help="Disable CPU fallback workers",
    )
    parser.add_argument(
        "--streams", action="store_true",
        help="Enable triple-buffered CUDA stream pipeline",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Use zero-overhead FastAligner (1M+ reads/s, no seeding/streams)",
    )
    parser.add_argument(
        "--cigar", action="store_true",
        help="Generate CIGAR strings via CPU traceback (post-alignment)",
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Use all CPU cores for CIGAR/chaining (ProcessPoolExecutor)",
    )
    parser.add_argument(
        "-w", "--band-width", type=int, default=50,
        help="Half-band width for banded DP (default: 50)",
    )
    parser.add_argument(
        "--gap-open", type=int, default=5,
        help="Gap opening penalty (default: 5)",
    )
    parser.add_argument(
        "--gap-extend", type=int, default=2,
        help="Gap extension penalty (default: 2)",
    )
    parser.add_argument(
        "--seed/--no-seed", dest="use_seed", default=True,
        help="Enable/disable minimizer seeding (default: enabled)",
    )
    parser.add_argument(
        "-k", "--kmer", type=int, default=15,
        help="K-mer size for minimizers (default: 15)",
    )
    parser.add_argument(
        "-W", "--window", type=int, default=10,
        help="Window size for minimizer selection (default: 10)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print summary as JSON to stdout",
    )

    args = parser.parse_args()

    if not Path(args.fastq).exists():
        print(f"Error: FASTQ file not found: {args.fastq}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.ref).exists():
        print(f"Error: Reference file not found: {args.ref}", file=sys.stderr)
        sys.exit(1)

    summary = run_pipeline(
        fastq_path=args.fastq,
        ref_path=args.ref,
        batch_size=args.batch_size,
        output_path=args.output,
        gpu_only=args.gpu_only,
        band_width=args.band_width,
        gap_open=args.gap_open,
        gap_extend=args.gap_extend,
        use_seed=args.use_seed,
        use_streams=args.streams,
        use_fast=args.fast,
        use_cigar=args.cigar,
        use_parallel=args.parallel,
        kmer=args.kmer,
        window=args.window,
    )

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"\n{'='*60}")
        stream_tag = " [STREAM PIPELINE]" if summary.get('streams') == 'enabled' else ""
        print(f"  HybAligner Pipeline — SW + Seeding{stream_tag}")
        print(f"{'='*60}")
        print(f"  Streams:          {summary.get('streams', 'disabled')}")
        print(f"  Seeding:          {summary['seeding']}")
        if summary.get('kmer'):
            print(f"  K-mer / Window:   k={summary['kmer']}, w={summary['window']}")
        print(f"  Reads processed:  {summary['n_reads']:,}")
        print(f"  Reads aligned:    {summary['n_aligned']:,} ({summary['pct_aligned']}%)")
        print(f"  Reference length: {summary['ref_len']:,} bp")
        print(f"  Band width:       {summary['band_width']}")
        print(f"  Gap penalties:    open={summary['gap_open']}, extend={summary['gap_extend']}")
        print(f"  Total time:       {summary['total_elapsed_ms']:.1f} ms")
        print(f"  Throughput:       {summary['throughput_reads_per_sec']:,.1f} reads/s")
        if summary.get('seed_index_build_ms', 0) > 0:
            print(f"  Seed index build: {summary['seed_index_build_ms']:.1f} ms")
        print(f"  GPU batches:      {summary['gpu_batches']}")
        print(f"  CPU batches:      {summary['cpu_batches']}")
        print(f"  Mean score:       {summary['score_mean']:.4f}")
        print(f"  Max score:        {summary['score_max']:.4f}")
        if summary.get('cigar'):
            c = summary['cigar']
            print(f"  CIGAR strings:    {c['n_cigars']} ({c['cigar_time_ms']} ms)")
            print(f"    Mean matches:   {c['mean_matches']}")
            print(f"    Mean insertions:{c['mean_insertions']}")
            print(f"    Mean deletions: {c['mean_deletions']}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

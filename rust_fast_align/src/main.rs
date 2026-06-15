//! HybAlign CLI — GPU-accelerated sequence alignment.
//!
//! ```bash
//! # Build
//! cd rust_fast_align && cargo build --release
//!
//! # Run
//! ./target/release/hyb-align reads.fastq ref.fasta
//! ./target/release/hyb-align reads.fastq ref.fasta -w 50 --json
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context;
use clap::Parser;

use hyb_align::{encode_from_fastq, encode_reference, AlignParams, FastAligner, FastqFile};

#[derive(Parser)]
#[command(name = "hyb-align")]
#[command(version = "0.1.0")]
#[command(about = "GPU-accelerated sequence aligner (Rust fast path)")]
struct Cli {
    /// Input FASTQ file (uncompressed)
    #[arg(value_name = "FASTQ")]
    fastq: PathBuf,

    /// Reference FASTA file (uncompressed)
    #[arg(value_name = "FASTA")]
    reference: PathBuf,

    /// Band width for Smith-Waterman (default: 50)
    #[arg(short = 'w', long, default_value = "50")]
    band_width: i32,

    /// Gap opening penalty (default: 5)
    #[arg(long, default_value = "5")]
    gap_open: i32,

    /// Gap extension penalty (default: 2)
    #[arg(long, default_value = "2")]
    gap_extend: i32,

    /// CUDA block size (default: 256)
    #[arg(long, default_value = "256")]
    block_size: i32,

    /// Path to libcuda_kernels.so (auto-detect if not specified)
    #[arg(long)]
    lib_path: Option<String>,

    /// Output results as JSON
    #[arg(long)]
    json: bool,

    /// Quiet mode — only print JSON
    #[arg(short = 'q', long)]
    quiet: bool,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // --- Parse inputs ---
    if !cli.quiet {
        eprintln!("HybAlign v0.1.0 (Rust)");
    }

    let t_total = Instant::now();

    // FASTQ
    let t_fq = Instant::now();
    let fq = FastqFile::open(&cli.fastq)
        .with_context(|| format!("Failed to open FASTQ: {}", cli.fastq.display()))?;
    let n_reads = fq.n_reads();
    let read_len = fq.max_len();
    let fq_ms = t_fq.elapsed().as_secs_f64() * 1000.0;

    // FASTA reference
    let t_ref = Instant::now();
    let ref_seq = std::fs::read_to_string(&cli.reference)
        .with_context(|| format!("Failed to open FASTA: {}", cli.reference.display()))?;
    // Strip header line and newlines
    let ref_clean: String = ref_seq
        .lines()
        .filter(|l| !l.starts_with('>'))
        .collect();
    let ref_bytes = encode_reference(&ref_clean);
    let ref_len = ref_bytes.len();
    let ref_ms = t_ref.elapsed().as_secs_f64() * 1000.0;

    if !cli.quiet {
        eprintln!(
            "  FASTQ: {} reads × {}bp max ({:.1} ms)",
            n_reads, read_len, fq_ms
        );
        eprintln!(
            "  Ref:   {} bp ({:.1} ms)",
            ref_len, ref_ms
        );
    }

    // --- Encode reads ---
    let t_enc = Instant::now();
    let reads_bytes = encode_from_fastq(&fq, read_len);
    let enc_ms = t_enc.elapsed().as_secs_f64() * 1000.0;

    if !cli.quiet {
        eprintln!(
            "  Encode: {:.1} KB ({:.1} ms)",
            reads_bytes.len() as f64 / 1024.0,
            enc_ms
        );
    }

    // --- GPU alignment ---
    let mut fa = if let Some(ref p) = cli.lib_path {
        FastAligner::with_lib_path(n_reads + 10, read_len, ref_len + 100, p)?
    } else {
        FastAligner::new(n_reads + 10, read_len, ref_len + 100)?
    };

    // Warmup: first CUDA call initializes context (~200ms). Do it once.
    let warmup_reads = vec![b'A'; read_len];
    let _ = fa.align(
        &warmup_reads, &ref_bytes,
        1, read_len, ref_len,
        AlignParams::default(),
    );

    let params = AlignParams {
        band_width: cli.band_width,
        gap_open: cli.gap_open,
        gap_extend: cli.gap_extend,
        block_size: cli.block_size,
    };

    let t_gpu = Instant::now();
    let result = fa.align(
        &reads_bytes,
        &ref_bytes,
        n_reads,
        read_len,
        ref_len,
        params,
    )?;
    let gpu_ms = t_gpu.elapsed().as_secs_f64() * 1000.0;

    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;

    let n_aligned = result.scores.iter().filter(|&&s| s > 0.0).count();
    let score_mean: f64 = if n_aligned > 0 {
        result.scores.iter().filter(|&&s| s > 0.0).map(|&s| s as f64).sum::<f64>()
            / n_aligned as f64
    } else {
        0.0
    };
    let throughput = n_reads as f64 / (total_ms / 1000.0);

    if cli.json {
        let json = serde_json::json!({
            "tool": "hyb-align-rs",
            "version": "0.1.0",
            "n_reads": n_reads,
            "read_len": read_len,
            "ref_len": ref_len,
            "band_width": cli.band_width,
            "n_aligned": n_aligned,
            "pct_aligned": if n_reads > 0 { 100.0 * n_aligned as f64 / n_reads as f64 } else { 0.0 },
            "score_mean": score_mean,
            "score_max": result.scores.iter().cloned().fold(0.0f32, f32::max),
            "total_ms": total_ms,
            "gpu_ms": gpu_ms,
            "parse_ms": fq_ms + ref_ms,
            "encode_ms": enc_ms,
            "throughput_reads_per_sec": throughput,
        });
        println!("{}", serde_json::to_string_pretty(&json)?);
    } else {
        println!();
        println!("═══════════════════════════════════════════════════════════");
        println!("  HybAlign Rust — Benchmark Results");
        println!("═══════════════════════════════════════════════════════════");
        println!("  Reads:    {}", n_reads);
        println!("  Ref:      {} bp", ref_len);
        println!("  Band:     ±{}", cli.band_width);
        println!("  ─────────────────────────────────────────────");
        println!("  Aligned:  {}/{} ({:.1}%)", n_aligned, n_reads,
                 if n_reads > 0 { 100.0 * n_aligned as f64 / n_reads as f64 } else { 0.0 });
        println!("  Score:    mean={:.1}, max={:.0}",
                 score_mean,
                 result.scores.iter().cloned().fold(0.0f32, f32::max));
        println!("  ─────────────────────────────────────────────");
        println!("  Total:    {:.1} ms", total_ms);
        println!("  GPU:      {:.1} ms", gpu_ms);
        println!("  Parse:    {:.1} ms", fq_ms + ref_ms);
        println!("  Encode:   {:.1} ms", enc_ms);
        println!("  ─────────────────────────────────────────────");
        println!("  Throughput: {:.0} reads/s", throughput);
        println!("═══════════════════════════════════════════════════════════");
    }

    Ok(())
}

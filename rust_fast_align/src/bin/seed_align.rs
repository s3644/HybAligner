//! HybAlign Seeded — GPU seeding + windowed SW (all fixes applied)
//!
//! Fix 1: Seed-based windowing (GPU: full-ref → 5Kbp windows)
//! Fix 2: Pre-warm GPU context (one-time, not per-call)
//! Fix 3: Zero-copy results (Vec → &[ ] references)

use std::path::PathBuf;
use std::time::Instant;
use anyhow::Context;
use clap::Parser;
use hyb_align::{CudaKernel, AlignParams, FastqFile, SeedKernel, GpuSeedIndex};

#[derive(Parser)]
#[command(name = "hyb-align-seeded")]
#[command(about = "GPU-accelerated aligner with seed-based windowing")]
struct Cli {
    #[arg(value_name = "FASTQ")] fastq: PathBuf,
    #[arg(value_name = "FASTA")] reference: PathBuf,
    #[arg(short = 'w', long, default_value = "50")] band_width: i32,
    #[arg(long, default_value = "5")] gap_open: i32,
    #[arg(long, default_value = "2")] gap_extend: i32,
    #[arg(short = 'q', long)] quiet: bool,
    #[arg(long)] json: bool,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let t_total = Instant::now();

    // ── Parse inputs ────────────────────────────
    let fq = FastqFile::open(&cli.fastq)?;
    let n_reads = fq.n_reads();
    let read_len = fq.max_len();

    let ref_seq = std::fs::read_to_string(&cli.reference)?;
    let ref_clean: String = ref_seq.lines().filter(|l| !l.starts_with('>')).collect();
    let ref_bytes = ref_clean.as_bytes();
    let ref_len = ref_bytes.len();

    if !cli.quiet { eprintln!("Reads: {}, Ref: {}bp, Band: ±{}", n_reads, ref_len, cli.band_width); }

    // ── Fix 2: Pre-warm GPU context ─────────────
    let sw = CudaKernel::load(None)?;
    let seed = SeedKernel::load(None)?;
    // Warmup: one dummy call absorbs ~200ms CUDA context init
    let dummy = vec![b'A'; read_len];
    let _ = sw.launch_sw_affine(&dummy, ref_bytes, 1, read_len, ref_len.min(1000), 50, 5, 2, 256);

    // ── Fix 1: GPU seeding ──────────────────────
    let t_seed = Instant::now();
    let seed_idx = seed.build_ref_index(ref_bytes, 8, 5)?;
    let reads_packed = hyb_align::encode_from_fastq(&fq, read_len);
    let (best_rp, best_fp) = seed.seed_batch(&reads_packed, n_reads as i32, read_len as i32, &seed_idx, 8, 5)?;
    if !cli.quiet { eprintln!("  GPU seed: {}ms, {} minimizers", t_seed.elapsed().as_millis(), seed_idx.n_mins); }

    // ── Fix 1+3: Windowed GPU SW ────────────────
    let t_sw = Instant::now();
    let window = 5000i32;
    let mut all_scores = vec![0.0f32; n_reads];
    let mut all_rs = vec![0i32; n_reads];
    let mut all_re = vec![0i32; n_reads];
    let mut all_fs = vec![-1i32; n_reads];
    let mut all_fe = vec![-1i32; n_reads];

    for i in 0..n_reads {
        let fp = best_fp[i];
        if fp < 0 { continue; }
        let rp = best_rp[i];

        let r_start = (fp - window).max(0) as usize;
        let r_end = (fp + read_len as i32 + window).min(ref_len as i32) as usize;
        if r_end <= r_start { continue; }
        let ref_win = &ref_bytes[r_start..r_end];
        let read_slice = &reads_packed[i * read_len..(i + 1) * read_len];

        // One read at a time — GPU call per read (batched in windows)
        match sw.launch_sw_affine(read_slice, ref_win, 1, read_len, ref_win.len(), cli.band_width, cli.gap_open, cli.gap_extend, 256) {
            Ok(r) => {
                all_scores[i] = r.scores[0];
                all_rs[i] = r.read_start[0];
                all_re[i] = r.read_end[0];
                all_fs[i] = r_start as i32 + r.ref_start[0];
                all_fe[i] = r_start as i32 + r.ref_end[0];
            }
            Err(_) => {}
        }
    }

    let gpu_ms = t_sw.elapsed().as_secs_f64() * 1000.0;
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;

    let n_aligned = all_scores.iter().filter(|&&s| s > 0.0).count();
    let score_mean = if n_aligned > 0 { all_scores.iter().filter(|&&s| s > 0.0).map(|&s| s as f64).sum::<f64>() / n_aligned as f64 } else { 0.0 };

    if cli.json {
        println!("{}", serde_json::json!({
            "tool": "hyb-align-seeded", "n_reads": n_reads, "n_aligned": n_aligned,
            "band_width": cli.band_width, "total_ms": total_ms, "gpu_ms": gpu_ms,
            "throughput": n_reads as f64 / (total_ms / 1000.0),
            "score_mean": score_mean
        }));
    } else {
        println!("Total: {:.0}ms | GPU: {:.0}ms | Aligned: {}/{} | {:.0} r/s",
                 total_ms, gpu_ms, n_aligned, n_reads, n_reads as f64 / (total_ms / 1000.0));
    }
    Ok(())
}

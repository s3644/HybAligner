//! WGS Aligner — genome-scale alignment via chunked reference indexing.
//!
//! Architecture mirrors Python `gpu/wgs_align.py`:
//!   1. Split reference into ~10 Mbp chunks with 1 Mbp overlap
//!   2. Per-chunk: 8-mer coarse index + 15-mer minimizer fine index
//!   3. Per-read: 8-mer chunk selection → 15-mer full search → anchor → SW
//!   4. Index serialization via serde (bincode)
//!
//! TODO: Full implementation — currently skeleton, see Python for reference.
//! The Python implementation already achieves 1,103 r/s on 47 Mbp chr21.

use std::collections::HashMap;

/// 8-mer coarse index: 2-bit encoded 8-mer → list of positions.
/// Uses u16 for the hash (2 bits × 8 = 16 bits).
pub type CoarseIndex = HashMap<u16, Vec<usize>>;

/// 15-mer minimizer fine index: minimizer hash → list of positions.
pub type FineIndex = HashMap<u64, Vec<usize>>;

/// One reference chunk (~10 Mbp) with dual-level indexes.
pub struct ChunkIndex {
    pub chunk_id: usize,
    pub ref_start: usize,
    pub ref_end: usize,
    pub ref_seq: Vec<u8>,
    pub index_8mer: CoarseIndex,
    pub index_15mer: FineIndex,
}

impl ChunkIndex {
    /// Build both coarse and fine indexes for this chunk.
    pub fn build(ref_seq: &[u8], chunk_id: usize, ref_start: usize) -> Self {
        let index_8mer = build_8mer_index(ref_seq, 8, 1);
        let index_15mer = crate::seed::SeedIndex::build(ref_seq, 15, 10);

        // Extract the internal HashMap from SeedIndex
        // TODO: expose SeedIndex's internal index
        let _ = index_15mer; // placeholder

        Self {
            chunk_id,
            ref_start,
            ref_end: ref_start + ref_seq.len(),
            ref_seq: ref_seq.to_vec(),
            index_8mer,
            index_15mer: HashMap::new(), // TODO
        }
    }
}

/// Build 8-mer coarse index (2-bit encoding, stride=1 for max sensitivity).
fn build_8mer_index(ref_seq: &[u8], k: usize, stride: usize) -> CoarseIndex {
    let enc = |b: u8| -> u16 {
        match b {
            b'A' | b'a' => 0,
            b'C' | b'c' => 1,
            b'G' | b'g' => 2,
            b'T' | b't' => 3,
            _ => 4, // invalid
        }
    };

    let mut index: CoarseIndex = HashMap::new();
    let n = ref_seq.len();

    for i in (0..n.saturating_sub(k)).step_by(stride) {
        let mut h: u16 = 0;
        let mut valid = true;
        for j in 0..k {
            let e = enc(ref_seq[i + j]);
            if e > 3 {
                valid = false;
                break;
            }
            h = (h << 2) | e;
        }
        if valid {
            index.entry(h).or_default().push(i);
        }
    }
    index
}

/// WGS Aligner — manages chunked reference index.
pub struct WgsAligner {
    pub chunks: Vec<ChunkIndex>,
    pub chunk_size: usize,
    pub overlap: usize,
    pub ref_len: usize,
}

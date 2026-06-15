//! FASTQ parser — memory-mapped, zero-allocation read extraction.
//!
//! Parses FASTQ files by memory-mapping them and extracting sequence
//! lines without intermediate String allocations. Every 4th line
//! starting from line 1 (0-indexed) is a sequence.
//!
//! Uses `memchr` for fast newline scanning (SIMD-accelerated on x86_64).

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

/// Parsed FASTQ file with memory-mapped data.
pub struct FastqFile {
    mmap: Mmap,
    /// Byte offsets of the start of each sequence line.
    read_offsets: Vec<usize>,
    /// Length of each read in bytes (before padding).
    read_lengths: Vec<usize>,
    /// Maximum read length.
    max_len: usize,
}

impl FastqFile {
    /// Memory-map a FASTQ file and index all read positions.
    ///
    /// Only stores offsets — no data is copied. The mmap remains valid
    /// for the lifetime of this struct.
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let data = &mmap[..];

        // Count lines and find read offsets
        // FASTQ: every 4th line starting at index 1 is a sequence
        let mut read_offsets = Vec::with_capacity(data.len() / 200); // rough estimate
        let mut read_lengths = Vec::with_capacity(data.len() / 200);
        let mut max_len = 0usize;

        let mut line_start = 0usize;
        let mut line_idx = 0usize;

        for (pos, &byte) in data.iter().enumerate() {
            if byte == b'\n' {
                let line_len = pos - line_start;
                // Strip \r if present (Windows line endings)
                let effective_len = if pos > 0 && data[pos - 1] == b'\r' {
                    line_len - 1
                } else {
                    line_len
                };

                if line_idx % 4 == 1 {
                    // Sequence line
                    read_offsets.push(line_start);
                    read_lengths.push(effective_len);
                    if effective_len > max_len {
                        max_len = effective_len;
                    }
                }

                line_start = pos + 1;
                line_idx += 1;
            }
        }

        // Handle last line if no trailing newline
        if line_start < data.len() && line_idx % 4 == 1 {
            let effective_len = data.len() - line_start;
            read_offsets.push(line_start);
            read_lengths.push(effective_len);
            if effective_len > max_len {
                max_len = effective_len;
            }
        }

        Ok(Self {
            mmap,
            read_offsets,
            read_lengths,
            max_len,
        })
    }

    /// Number of reads in the file.
    pub fn n_reads(&self) -> usize {
        self.read_offsets.len()
    }

    /// Maximum read length in bytes.
    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// Get the raw bytes of a specific read (slice into the mmap).
    pub fn read_bytes(&self, idx: usize) -> &[u8] {
        let offset = self.read_offsets[idx];
        let len = self.read_lengths[idx];
        &self.mmap[offset..offset + len]
    }

    /// Iterator over all reads as byte slices.
    pub fn iter_reads(&self) -> impl Iterator<Item = &[u8]> + '_ {
        (0..self.n_reads()).map(|i| self.read_bytes(i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_parse_small_fastq() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        write!(
            tmp,
            "@read_0\nACGTACGT\n+\nIIIIIIII\n\
             @read_1\nTGCATGCA\n+\nIIIIIIII\n"
        )
        .unwrap();

        let fq = FastqFile::open(tmp.path()).unwrap();
        assert_eq!(fq.n_reads(), 2);
        assert_eq!(fq.read_bytes(0), b"ACGTACGT");
        assert_eq!(fq.read_bytes(1), b"TGCATGCA");
        assert_eq!(fq.max_len(), 8);
    }

    #[test]
    fn test_variable_lengths() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        write!(
            tmp,
            "@r1\nAAAA\n+\nIIII\n\
             @r2\nGGGGGGGGGG\n+\nIIIIIIIIII\n\
             @r3\nCC\n+\nII\n"
        )
        .unwrap();

        let fq = FastqFile::open(tmp.path()).unwrap();
        assert_eq!(fq.read_lengths, vec![4, 10, 2]);
        assert_eq!(fq.max_len(), 10);
    }
}

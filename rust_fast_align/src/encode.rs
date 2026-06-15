//! Read encoder — converts variable-length reads into a flat, padded byte buffer.
//!
//! Each read is padded/truncated to exactly `read_len` with 'N' bytes.
//! The output is one contiguous `Vec<u8>` suitable for passing directly
//! to the CUDA kernel via FFI.
//!
//! Performance: single-pass, pre-allocated buffer, no per-read allocations.

/// Encode a slice of read byte-slices into a flat padded buffer.
///
/// Each read is copied into the buffer and right-padded with `b'N'` to
/// exactly `read_len`. Reads longer than `read_len` are truncated.
///
/// Returns a contiguous `Vec<u8>` of size `n_reads * read_len`.
pub fn encode_reads(reads: &[&[u8]], read_len: usize) -> Vec<u8> {
    let n = reads.len();
    let total = n * read_len;
    let mut buf = vec![b'N'; total];

    for (i, read) in reads.iter().enumerate() {
        let copy_len = read.len().min(read_len);
        let dest_start = i * read_len;
        // SAFETY: bounds checked by min() and index calculation
        buf[dest_start..dest_start + copy_len].copy_from_slice(&read[..copy_len]);
        // Rest is already 'N' from initialization
    }

    buf
}

/// Encode from a FastqFile, avoiding intermediate Vec of slices.
///
/// More efficient than `encode_reads` because it copies directly from
/// the mmap without creating intermediate references.
pub fn encode_from_fastq(fq: &crate::fastq::FastqFile, read_len: usize) -> Vec<u8> {
    let n = fq.n_reads();
    let total = n * read_len;
    let mut buf = vec![b'N'; total];

    for i in 0..n {
        let read = fq.read_bytes(i);
        let copy_len = read.len().min(read_len);
        let dest_start = i * read_len;
        buf[dest_start..dest_start + copy_len].copy_from_slice(&read[..copy_len]);
    }

    buf
}

/// Encode a reference sequence into a byte buffer.
///
/// FASTA references are single sequences — just convert to bytes.
pub fn encode_reference(seq: &str) -> Vec<u8> {
    seq.as_bytes().to_vec()
}

/// Encode reference from bytes (e.g., from memory-mapped FASTA).
pub fn encode_reference_bytes(seq: &[u8]) -> Vec<u8> {
    seq.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_padding() {
        let reads: &[&[u8]] = &[b"ACGT", b"TG"];
        let encoded = encode_reads(reads, 5);
        assert_eq!(encoded.len(), 10);
        assert_eq!(&encoded[0..5], b"ACGTN");
        assert_eq!(&encoded[5..10], b"TGNNN");
    }

    #[test]
    fn test_encode_truncation() {
        let reads: &[&[u8]] = &[b"AAAAACCCCC"];
        let encoded = encode_reads(reads, 5);
        assert_eq!(encoded.len(), 5);
        assert_eq!(&encoded[..], b"AAAAA");
    }

    #[test]
    fn test_encode_empty() {
        let reads: &[&[u8]] = &[];
        let encoded = encode_reads(reads, 10);
        assert!(encoded.is_empty());
    }
}

/**
 * seed_kernel.cu — Minimizer seeding kernels for sequence alignment.
 *
 * Implements GPU-accelerated minimizer extraction and seed matching
 * for anchor-based alignment (Minimap2-style chaining seeds).
 *
 * Designed for DGX Spark (Blackwell sm_90, CUDA 13.x).
 */

#include <cuda_runtime.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Minimizer extraction constants
// ---------------------------------------------------------------------------
#define KMER_SIZE    15    // k-mer length for minimizers
#define WINDOW_SIZE  10    // sliding window size for minimizer selection

__device__ unsigned char base_to_2bit(char c) {
    switch (c) {
        case 'A': case 'a': return 0x00;  // 00
        case 'C': case 'c': return 0x01;  // 01
        case 'G': case 'g': return 0x02;  // 10
        case 'T': case 't': return 0x03;  // 11
        default:              return 0x00;  // treat N as A
    }
}

// ---------------------------------------------------------------------------
// Compute a 2-bit encoded k-mer hash (canonical)
// ---------------------------------------------------------------------------
__device__ unsigned long long compute_kmer_hash(
    const char* seq,
    int start,
    int k
) {
    unsigned long long fwd = 0;
    unsigned long long rev = 0;

    for (int i = 0; i < k; i++) {
        unsigned char b = base_to_2bit(seq[start + i]);
        fwd = (fwd << 2) | b;
        rev = (rev >> 2) | ((unsigned long long)(3 - b) << (2 * (k - 1)));
    }

    // Take canonical (minimum of forward and reverse complement)
    return (fwd < rev) ? fwd : rev;
}

// ---------------------------------------------------------------------------
// Extract minimizers from a single sequence (one thread per sequence)
//
// Parameters:
//   seq          — packed sequences, shape (num_seqs * seq_len)
//   minimizers   — output: (hash, position) pairs, padded to max_minimizers
//   num_seqs     — number of sequences
//   seq_len      — length of each sequence (padded)
//   k            — k-mer size
//   w            — window size
//   max_mins     — max minimizers per sequence (output padding)
// ---------------------------------------------------------------------------
__global__
void extract_minimizers_kernel(
    const char* __restrict__ seq,
    unsigned long long* __restrict__ minimizer_hashes,
    int* __restrict__ minimizer_positions,
    int* __restrict__ num_minimizers_out,
    int num_seqs,
    int seq_len,
    int k,
    int w,
    int max_mins
) {
    int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq_idx >= num_seqs) return;

    const char* my_seq = seq + seq_idx * seq_len;
    int num_windows = seq_len - k - w + 2;
    if (num_windows <= 0) {
        num_minimizers_out[seq_idx] = 0;
        return;
    }

    int count = 0;
    unsigned long long prev_min_hash = ~0ULL;

    for (int win_start = 0; win_start < num_windows && count < max_mins; win_start++) {
        unsigned long long min_hash = ~0ULL;
        int min_pos = -1;

        // Scan window for minimum k-mer
        for (int offset = 0; offset < w; offset++) {
            int pos = win_start + offset;
            if (pos + k > seq_len) break;

            unsigned long long h = compute_kmer_hash(my_seq, pos, k);
            if (h < min_hash) {
                min_hash = h;
                min_pos = pos;
            }
        }

        // Deduplicate: only store if minimizer changed
        if (min_hash != prev_min_hash && min_pos >= 0) {
            int out_idx = seq_idx * max_mins + count;
            minimizer_hashes[out_idx] = min_hash;
            minimizer_positions[out_idx] = min_pos;
            count++;
            prev_min_hash = min_hash;
        }
    }

    num_minimizers_out[seq_idx] = count;
}

// ---------------------------------------------------------------------------
// Match minimizers between reads and reference
//
// For each read minimizer, find matching positions in the reference minimizer
// set using a shared-memory hash lookup approach.
//
// Parameters:
//   read_mins     — read minimizer hashes, shape (num_reads * max_mins)
//   read_pos      — read minimizer positions, shape (num_reads * max_mins)
//   ref_mins      — reference minimizer hashes, shape max_ref_mins
//   ref_pos       — reference minimizer positions, shape max_ref_mins
//   num_ref_mins  — actual number of reference minimizers
//   anchors       — output: (read_pos, ref_pos) anchor pairs
//   anchor_counts — number of anchors per read
//   num_reads     — number of reads
//   max_mins      — max minimizers per read
//   max_anchors   — max anchors per read (output padding)
// ---------------------------------------------------------------------------
__global__
void match_seeds_kernel(
    const unsigned long long* __restrict__ read_mins,
    const int* __restrict__ read_pos,
    const unsigned long long* __restrict__ ref_mins,
    const int* __restrict__ ref_pos,
    int num_ref_mins,
    int* __restrict__ anchor_read_pos,
    int* __restrict__ anchor_ref_pos,
    int* __restrict__ anchor_counts,
    int num_reads,
    int max_mins,
    int max_anchors
) {
    int read_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (read_idx >= num_reads) return;

    int anchor_count = 0;

    // Brute-force match per read minimizer against all reference minimizers
    for (int mi = 0; mi < max_mins && anchor_count < max_anchors; mi++) {
        unsigned long long r_hash = read_mins[read_idx * max_mins + mi];
        if (r_hash == 0) continue;  // skip empty slot

        int r_pos = read_pos[read_idx * max_mins + mi];

        for (int rj = 0; rj < num_ref_mins && anchor_count < max_anchors; rj++) {
            if (ref_mins[rj] == r_hash) {
                int out_idx = read_idx * max_anchors + anchor_count;
                anchor_read_pos[out_idx] = r_pos;
                anchor_ref_pos[out_idx] = ref_pos[rj];
                anchor_count++;
            }
        }
    }

    anchor_counts[read_idx] = anchor_count;
}

// ---------------------------------------------------------------------------
// Hash-table based seed matching (replaces brute-force O(N²) scan)
//
// Builds an open-addressing hash table from reference minimizers,
// then probes each read minimizer in O(1) amortized per lookup.
// ---------------------------------------------------------------------------

#define HASH_EMPTY  0xFFFFFFFFFFFFFFFFULL
#define HASH_TABLE_SCALE 2  // table size = 2 * n_ref_mins (load factor ~0.5)

// Jenkins-style integer hash for secondary probing
__device__ unsigned long long rehash(unsigned long long key, int attempt) {
    key = (~key) + (attempt << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return key;
}

/**
 * Build hash table from reference minimizers.
 * One thread per reference minimizer — atomic insertion with linear probing.
 *
 * Parameters:
 *   ref_hashes   — reference minimizer hashes [n_ref_mins]
 *   ref_pos      — reference minimizer positions [n_ref_mins]
 *   n_ref_mins   — number of reference minimizers
 *   table_keys   — output hash table keys [table_size], init to HASH_EMPTY
 *   table_vals   — output hash table values (ref positions) [table_size]
 *   table_size   — size of hash table
 *   max_vals_per_key — max positions stored per key (chaining within bucket)
 */
__global__
void build_hash_table_kernel(
    const unsigned long long* __restrict__ ref_hashes,
    const int* __restrict__ ref_pos,
    int n_ref_mins,
    unsigned long long* __restrict__ table_keys,
    int* __restrict__ table_vals,
    int table_size,
    int max_vals_per_key
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_ref_mins) return;

    unsigned long long key = ref_hashes[idx];
    int val = ref_pos[idx];

    if (key == 0 || key == HASH_EMPTY) return;

    // Primary hash
    // Linear probing with rehash
    for (int attempt = 0; attempt < 32; attempt++) {
        unsigned long long probe_key = (attempt == 0) ? key : rehash(key, attempt);
        unsigned int probe_slot = (unsigned int)(probe_key % table_size);

        unsigned long long old = atomicCAS(&table_keys[probe_slot], HASH_EMPTY, key);

        if (old == HASH_EMPTY || old == key) {
            // Found slot — store value in the value array
            int base = probe_slot * max_vals_per_key;
            for (int vi = 0; vi < max_vals_per_key; vi++) {
                int old_val = atomicCAS(&table_vals[base + vi], -1, val);
                if (old_val == -1) break;  // stored successfully
            }
            return;
        }
        // Collision — probe next
    }
}

/**
 * Match read minimizers against hash table.
 * One thread per read — probes each minimizer in the hash table.
 *
 * Parameters:
 *   read_hashes   — read minimizer hashes [num_reads * max_mins]
 *   read_pos      — read minimizer positions [num_reads * max_mins]
 *   table_keys    — hash table keys [table_size]
 *   table_vals    — hash table values (positions array) [table_size * max_vals_per_key]
 *   table_size    — size of hash table
 *   max_vals_per_key — max values per key
 *   anchor_rp     — output: read positions of anchors [num_reads * max_anchors]
 *   anchor_fp     — output: ref positions of anchors [num_reads * max_anchors]
 *   anchor_counts — output: number of anchors per read [num_reads]
 *   num_reads     — number of reads
 *   max_mins      — max minimizers per read (input)
 *   max_anchors   — max anchors per read (output)
 */
__global__
void match_hash_table_kernel(
    const unsigned long long* __restrict__ read_hashes,
    const int* __restrict__ read_pos,
    const unsigned long long* __restrict__ table_keys,
    const int* __restrict__ table_vals,
    int table_size,
    int max_vals_per_key,
    int* __restrict__ anchor_rp,
    int* __restrict__ anchor_fp,
    int* __restrict__ anchor_counts,
    int num_reads,
    int max_mins,
    int max_anchors
) {
    int read_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (read_idx >= num_reads) return;

    int anchor_count = 0;

    for (int mi = 0; mi < max_mins && anchor_count < max_anchors; mi++) {
        unsigned long long r_hash = read_hashes[read_idx * max_mins + mi];
        if (r_hash == 0 || r_hash == HASH_EMPTY) continue;

        int r_pos = read_pos[read_idx * max_mins + mi];

        // Probe hash table
        for (int attempt = 0; attempt < 32; attempt++) {
            unsigned long long probe_key = (attempt == 0) ? r_hash : rehash(r_hash, attempt);
            unsigned int probe_slot = (unsigned int)(probe_key % table_size);

            unsigned long long stored_key = table_keys[probe_slot];

            if (stored_key == HASH_EMPTY) {
                // Key not found
                break;
            }

            if (stored_key == r_hash) {
                // Match found — retrieve all positions stored at this slot
                int base = probe_slot * max_vals_per_key;
                for (int vi = 0; vi < max_vals_per_key && anchor_count < max_anchors; vi++) {
                    int f_pos = table_vals[base + vi];
                    if (f_pos < 0) break;  // no more values

                    int out_idx = read_idx * max_anchors + anchor_count;
                    anchor_rp[out_idx] = r_pos;
                    anchor_fp[out_idx] = f_pos;
                    anchor_count++;
                }
                break;  // done with this minimizer
            }
            // Collision with different key — probe next
        }
    }

    anchor_counts[read_idx] = anchor_count;
}

// ---------------------------------------------------------------------------
// Host-callable wrappers (extern "C" for ctypes interop)
// ---------------------------------------------------------------------------
extern "C" {

int launch_extract_minimizers(
    const char* seq,
    unsigned long long* minimizer_hashes,
    int* minimizer_positions,
    int* num_minimizers_out,
    int num_seqs,
    int seq_len,
    int k,
    int w,
    int max_mins,
    int block_size
) {
    if (block_size <= 0) block_size = 256;
    int num_blocks = (num_seqs + block_size - 1) / block_size;

    extract_minimizers_kernel<<<num_blocks, block_size>>>(
        seq, minimizer_hashes, minimizer_positions,
        num_minimizers_out,
        num_seqs, seq_len, k, w, max_mins
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Minimizer kernel error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceSynchronize();
    return 0;
}

int launch_match_seeds(
    const unsigned long long* read_mins,
    const int* read_pos,
    const unsigned long long* ref_mins,
    const int* ref_pos,
    int num_ref_mins,
    int* anchor_read_pos,
    int* anchor_ref_pos,
    int* anchor_counts,
    int num_reads,
    int max_mins,
    int max_anchors,
    int block_size
) {
    if (block_size <= 0) block_size = 256;
    int num_blocks = (num_reads + block_size - 1) / block_size;

    match_seeds_kernel<<<num_blocks, block_size>>>(
        read_mins, read_pos,
        ref_mins, ref_pos,
        num_ref_mins,
        anchor_read_pos, anchor_ref_pos,
        anchor_counts,
        num_reads, max_mins, max_anchors
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Seed match kernel error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceSynchronize();
    return 0;
}

int launch_build_hash_table(
    const unsigned long long* ref_hashes,
    const int* ref_pos,
    int n_ref_mins,
    unsigned long long* table_keys,
    int* table_vals,
    int table_size,
    int max_vals_per_key,
    int block_size
) {
    if (block_size <= 0) block_size = 256;
    int num_blocks = (n_ref_mins + block_size - 1) / block_size;

    build_hash_table_kernel<<<num_blocks, block_size>>>(
        ref_hashes, ref_pos, n_ref_mins,
        table_keys, table_vals,
        table_size, max_vals_per_key
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "build_hash_table error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    cudaDeviceSynchronize();
    return 0;
}

int launch_match_hash_table(
    const unsigned long long* read_hashes,
    const int* read_pos,
    const unsigned long long* table_keys,
    const int* table_vals,
    int table_size,
    int max_vals_per_key,
    int* anchor_rp,
    int* anchor_fp,
    int* anchor_counts,
    int num_reads,
    int max_mins,
    int max_anchors,
    int block_size
) {
    if (block_size <= 0) block_size = 256;
    int num_blocks = (num_reads + block_size - 1) / block_size;

    match_hash_table_kernel<<<num_blocks, block_size>>>(
        read_hashes, read_pos,
        table_keys, table_vals,
        table_size, max_vals_per_key,
        anchor_rp, anchor_fp, anchor_counts,
        num_reads, max_mins, max_anchors
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "match_hash_table error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    cudaDeviceSynchronize();
    return 0;
}

} // extern "C"

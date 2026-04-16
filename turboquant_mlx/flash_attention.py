"""Single-kernel fused SDPA over packed TurboQuant K/V.

Design goal: eliminate the full-size fp16 dequant tensor that today's
turboquant_attention produces as intermediates (scores, weights, v_deq).
In a fused kernel, these stay in shared memory / registers — the only
KV bytes that ever live in device memory are the packed uint32 indices
and the fp32 norms.

Algorithm (per-head MHA; GQA left as a follow-up):

  1. Pre-rotate Q outside (q_rot = WHT(signs_k * q)) — same convention
     as prerot_fused_qk_scores.
  2. Walk K positions in blocks of size B_c. Inside each block:
       a. Compute B_c Q@K scores via centroid dot with q_rot.
       b. Online softmax: update running (m, l) and rescale the output
          accumulator by exp(m_old - m_new).
       c. For each position in the block, dequant V (WHT butterfly over
          centroids, apply signs and v_norm) and accumulate
          exp(score - m_new) * v_elem into each thread's output slot.
  3. After the full K pass, divide by l and write out.

Correctness anchor: at any (B_c, seq_len), the output of this kernel
must match the composed (softmax(QK * scale), @ dequant(V)) pipeline of
turboquant_attention to float32 noise. Tests live in
tests/test_flash_attention.py.
"""

from __future__ import annotations

import math

import mlx.core as mx

__all__ = ["flash_attention_turboquant"]

FLASH_ATTN_KERNEL = """
    // Single-kernel fused SDPA. Uses the butterfly-pulled-out trick:
    // V-side butterfly is linear, so
    //     sum_pos exp_score[pos] * butterfly(c_v[idx_pos])
    //   = butterfly(sum_pos exp_score[pos] * c_v[idx_pos])
    // which lets us accumulate per-thread S[elem] without any
    // threadgroup barrier per K position, then do exactly one butterfly
    // on S at the end. In online-softmax form, when m changes, we also
    // rescale S by the same alpha factor we apply to l.
    uint head = threadgroup_position_in_grid.x;       // q_head
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint seq_len = dims[1];
    uint bits = dims[2];
    uint k_vpw = dims[3];
    uint k_pdim = dims[4];
    uint v_vpw = dims[5];
    uint v_pdim = dims[6];
    uint B_c = dims[7];
    uint n_rep = dims[8];
    uint bit_mask = (1u << bits) - 1u;
    T attn_scale = scales[0];
    T inv_dim = scales[1];
    uint kv_head = head / n_rep;

    threadgroup T bfly[128];          // reduction + final butterfly workspace
    threadgroup T scores[32];         // up to B_c=32
    threadgroup T m_state[1];
    threadgroup T l_state[1];
    threadgroup T m_new_s[1];
    threadgroup T alpha_s[1];

    if (elem == 0) {
        m_state[0] = (T)(-INFINITY);
        l_state[0] = (T)0;
    }
    T S = (T)0;                       // per-thread accumulator for butterfly-out
    T q_elem = q_rot[head * dim + elem];
    uint v_word_idx = elem / v_vpw;
    uint v_pos_in_word = elem % v_vpw;
    uint v_shift = v_pos_in_word * bits;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint bstart = 0; bstart < seq_len; bstart += B_c) {
        uint bsize = min(B_c, seq_len - bstart);

        // Phase 1: compute bsize Q@K scores using prerotated Q. Each
        // score is a dim-wide reduction; use simd_sum + one cross-simd
        // barrier instead of a full log(dim) tree reduction.
        threadgroup T simd_sums[8];   // supports dim up to 256
        uint simd_id = elem / 32;
        uint lane_id = elem % 32;
        uint n_simds = dim / 32;
        for (uint b = 0; b < bsize; b++) {
            uint pos = bstart + b;
            uint k_word_idx = elem / k_vpw;
            uint k_pos_in_word = elem % k_vpw;
            uint word = k_packed[kv_head * seq_len * k_pdim + pos * k_pdim + k_word_idx];
            uint idx = (word >> (k_pos_in_word * bits)) & bit_mask;
            T part = k_centroids[idx] * q_elem;
            T simd_part = simd_sum(part);
            if (lane_id == 0) simd_sums[simd_id] = simd_part;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (elem == 0) {
                T total = (T)0;
                for (uint i = 0; i < n_simds; i++) total += simd_sums[i];
                scores[b] = total * k_norms[kv_head * seq_len + pos]
                            * inv_dim * attn_scale;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Phase 2: online softmax update (serialized on thread 0).
        if (elem == 0) {
            T m_old = m_state[0];
            T m_new = m_old;
            for (uint b = 0; b < bsize; b++) {
                if (scores[b] > m_new) m_new = scores[b];
            }
            m_new_s[0] = m_new;
            T alpha = exp(m_old - m_new);
            alpha_s[0] = alpha;
            T l_new = l_state[0] * alpha;
            for (uint b = 0; b < bsize; b++) {
                T exp_s = exp(scores[b] - m_new);
                scores[b] = exp_s;          // reuse slot for phase 3
                l_new += exp_s;
            }
            l_state[0] = l_new;
            m_state[0] = m_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Rescale the butterfly-out accumulator by alpha (same as we'd do
        // for an output accumulator — S is a linear combination so the
        // softmax rescaling commutes).
        S *= alpha_s[0];

        // Phase 3: accumulate exp_score * v_norm * centroid[v_idx] into S
        // *without* running the butterfly per position.
        for (uint b = 0; b < bsize; b++) {
            uint pos = bstart + b;
            uint word = v_packed[kv_head * seq_len * v_pdim + pos * v_pdim + v_word_idx];
            uint idx = (word >> v_shift) & bit_mask;
            S += scores[b] * v_norms[kv_head * seq_len + pos] * v_centroids[idx];
        }
    }

    // Final butterfly: one threadgroup-wide WHT on S.
    bfly[elem] = S;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint h = 1;
    while (h < dim) {
        uint block = elem / (2 * h);
        uint offset = elem % (2 * h);
        if (offset < h) {
            uint j = block * 2 * h + offset;
            T a = bfly[j];
            T bb = bfly[j + h];
            bfly[j]     = a + bb;
            bfly[j + h] = a - bb;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h *= 2;
    }

    out[head * dim + elem] = bfly[elem] * v_signs[elem] * inv_dim / l_state[0];
"""

_flash_attention = None


def flash_attention_turboquant(
    q_rot: mx.array,
    k_packed: mx.array,
    k_norms: mx.array,
    v_packed: mx.array,
    v_norms: mx.array,
    k_centroids: mx.array,
    v_centroids: mx.array,
    v_signs: mx.array,
    dim: int,
    bits: int,
    attn_scale: float,
    block_size: int = 16,
    n_rep: int = 1,
) -> mx.array:
    """Single-kernel fused SDPA over packed TurboQuant K/V.

    Args:
        q_rot: (n_q_heads, dim) pre-rotated query, matching convention of
            prerotate_query (raw WHT(signs_k * q), un-normalized).
        k_packed: (n_kv_heads, seq_len, k_pdim) packed uint32 K indices.
        k_norms: (n_kv_heads, seq_len) K vector norms (fp32).
        v_packed: (n_kv_heads, seq_len, v_pdim) packed uint32 V indices.
        v_norms: (n_kv_heads, seq_len) V vector norms (fp32).
        k_centroids: (n_levels,) K Lloyd-Max centroids.
        v_centroids: (n_levels,) V Lloyd-Max centroids.
        v_signs: (dim,) ±1 V rotation signs.
        dim: head dimension (power of 2, <= 128).
        bits: quantization bit width (1-4).
        attn_scale: 1/sqrt(dim) attention scaling.
        block_size: tile size B_c over K positions (1..32).
        n_rep: Q-heads per KV-head for GQA. n_rep=1 is MHA.

    Returns:
        (n_q_heads, dim) attention output (softmax(QK * scale) @ V).
    """
    global _flash_attention
    if _flash_attention is None:
        _flash_attention = mx.fast.metal_kernel(
            name="tq_flash_attention",
            input_names=[
                "q_rot",
                "k_packed",
                "k_norms",
                "v_packed",
                "v_norms",
                "k_centroids",
                "v_centroids",
                "v_signs",
                "scales",
                "dims",
            ],
            output_names=["out"],
            source=FLASH_ATTN_KERNEL,
        )

    if dim > 128 or dim & (dim - 1):
        raise ValueError(f"dim must be a power of 2 and <= 128, got {dim}")
    if block_size < 1 or block_size > 32:
        raise ValueError(f"block_size must be in 1..32, got {block_size}")

    n_q_heads = q_rot.shape[0]
    n_kv_heads, seq_len = k_norms.shape
    if n_q_heads != n_kv_heads * n_rep:
        raise ValueError(
            f"n_q_heads ({n_q_heads}) must equal n_kv_heads ({n_kv_heads}) * n_rep ({n_rep})"
        )

    k_pdim = k_packed.shape[-1]
    v_pdim = v_packed.shape[-1]
    vpw = {1: 32, 2: 16, 3: 10, 4: 8}[bits]
    scales = mx.array([attn_scale, 1.0 / dim], dtype=mx.float32)
    dims_arr = mx.array(
        [
            dim,
            seq_len,
            bits,
            vpw,     # k_vpw
            k_pdim,
            vpw,     # v_vpw (same bits)
            v_pdim,
            block_size,
            n_rep,
        ],
        dtype=mx.uint32,
    )

    outputs = _flash_attention(
        inputs=[
            q_rot.astype(mx.float32).reshape(n_q_heads * dim),
            k_packed.astype(mx.uint32).reshape(n_kv_heads * seq_len * k_pdim),
            k_norms.astype(mx.float32).reshape(n_kv_heads * seq_len),
            v_packed.astype(mx.uint32).reshape(n_kv_heads * seq_len * v_pdim),
            v_norms.astype(mx.float32).reshape(n_kv_heads * seq_len),
            k_centroids,
            v_centroids,
            v_signs,
            scales,
            dims_arr,
        ],
        template=[("T", mx.float32)],
        grid=(n_q_heads * dim, 1, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[(n_q_heads * dim,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0].reshape(n_q_heads, dim)

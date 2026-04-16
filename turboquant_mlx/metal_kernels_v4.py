"""Metal kernels v4: pre-rotated query path.

The core insight (see fused_attention.py docstring):

    <Q, dequant(K_i)> = norm_i / sqrt(d) * <WHT(signs * Q), centroids[K_i]>

So if we pre-rotate the query ONCE per decode step (O(d log d)), we can
compute Q@K scores for the whole sequence without running the WHT butterfly
on every cached K (saves O(seq_len * d log d) work in the hot path).

Convention matches turboquant_mlx.metal.FUSED_QUANTIZE_KERNEL: the WHT
butterfly inside our Metal kernels is the "raw" (un-normalized) butterfly.
The 1/sqrt(d) factor is applied explicitly as `scale[0]` where needed so
the code stays self-consistent across encode/decode/attention.

Three public functions:
  prerotate_query            — one-shot: Q -> WHT_raw(signs * Q)
  prerot_fused_qk_scores     — Q_rot @ K_packed without per-K butterfly
  prerot_packed_dequantize   — full V dequant (just re-exports the existing
                                packed_dequantize, since V always needs the
                                inverse WHT butterfly).
"""

import mlx.core as mx
import math

from turboquant_mlx.kernels import packed_dequantize as prerot_packed_dequantize

__all__ = [
    "prerotate_query",
    "prerot_fused_qk_scores",
    "prerot_packed_dequantize",
]

# --- Pre-rotate query: signs * Q, then raw WHT butterfly (no 1/sqrt(d)).
# One threadgroup per head, `dim` threads cooperating on the butterfly.
PREROTATE_QUERY_KERNEL = """
    uint head = threadgroup_position_in_grid.x;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];

    threadgroup T shared[256];
    shared[elem] = q_in[head * dim + elem] * signs[elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint h = 1;
    while (h < dim) {
        uint block = elem / (2 * h);
        uint offset = elem % (2 * h);
        if (offset < h) {
            uint j = block * 2 * h + offset;
            T a = shared[j];
            T b = shared[j + h];
            shared[j]     = a + b;
            shared[j + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h *= 2;
    }

    q_out[head * dim + elem] = shared[elem];
"""

# --- Pre-rotated Q @ K, GQA-aware: no per-K butterfly, no per-K signs.
# Grid.y addresses query heads (n_q_heads). K is indexed by kv_head =
# head / n_rep, so grouped-query attention (n_kv_heads < n_q_heads)
# reads the same packed KV slot from multiple Q heads without any
# mx.repeat expansion.
# scores[q_head, pos] = (norms[kv_head, pos] / sqrt(d)) *
#                       <Q_rot[q_head], centroids[K_idx[kv_head, pos]]>
PREROT_FUSED_QK_KERNEL = """
    uint pos = threadgroup_position_in_grid.x;
    uint head = threadgroup_position_in_grid.y;  // q_head
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint seq_len = dims[1];
    uint bits = dims[2];
    uint vals_per_word = dims[3];
    uint packed_dim = dims[4];
    uint n_rep = dims[5];
    uint bit_mask = (1u << bits) - 1u;
    uint kv_head = head / n_rep;

    // Unpack one codebook index for (kv_head, pos, elem).
    uint kv_base = kv_head * seq_len * packed_dim + pos * packed_dim;
    uint word_idx = elem / vals_per_word;
    uint pos_in_word = elem % vals_per_word;
    uint word = packed[kv_base + word_idx];
    uint idx = (word >> (pos_in_word * bits)) & bit_mask;

    // Partial product with the pre-rotated query — no butterfly here.
    T partial = centroids[idx] * q_rot[head * dim + elem];

    // Reduce across the `dim` threads using simd_sum instead of a full
    // log(dim) tree reduction through shared memory. simd_sum sums all
    // 32 lanes in a SIMD group in one instruction with no barrier; we
    // then stitch together the (dim / 32) SIMD-group partials through
    // a single threadgroup barrier. On dim=128 this goes from 7 barriers
    // to 1.
    T simd_part = simd_sum(partial);
    threadgroup T simd_sums[8];   // supports dim up to 256
    uint simd_id = elem / 32;
    uint lane_id = elem % 32;
    if (lane_id == 0) simd_sums[simd_id] = simd_part;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (elem == 0) {
        T total = (T)0;
        uint n_simds = dim / 32;
        for (uint i = 0; i < n_simds; i++) total += simd_sums[i];
        out[head * seq_len + pos] = total * norms[kv_head * seq_len + pos] * scale[0];
    }
"""

_prerotate_query = None
_prerot_fused_qk = None


def prerotate_query(q: mx.array, signs: mx.array) -> mx.array:
    """Pre-rotate a decode-step query: signs * q → raw WHT butterfly.

    Args:
        q: (n_heads, dim)
        signs: (dim,) ±1 rotation signs (same convention as the encoder).

    Returns:
        (n_heads, dim) rotated query in the same space as the raw codebook.
    """
    global _prerotate_query
    if _prerotate_query is None:
        _prerotate_query = mx.fast.metal_kernel(
            name="tq_prerotate_query",
            input_names=["q_in", "signs", "dims"],
            output_names=["q_out"],
            source=PREROTATE_QUERY_KERNEL,
        )

    if q.ndim != 2:
        raise ValueError(f"prerotate_query expects (n_heads, dim), got {q.shape}")
    n_heads, dim = q.shape
    dims_arr = mx.array([dim], dtype=mx.uint32)
    outputs = _prerotate_query(
        inputs=[q.astype(mx.float32).reshape(n_heads * dim), signs, dims_arr],
        template=[("T", mx.float32)],
        grid=(n_heads * dim, 1, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[(n_heads * dim,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0].reshape(n_heads, dim)


def prerot_fused_qk_scores(
    q_rot: mx.array,
    k_packed: mx.array,
    k_norms: mx.array,
    centroids: mx.array,
    dim: int,
    bits: int,
    n_rep: int = 1,
) -> mx.array:
    """Compute Q@K scores using a pre-rotated query.

    Args:
        q_rot: (n_q_heads, dim) output of prerotate_query.
        k_packed: (n_kv_heads, seq_len, packed_dim) packed uint32 indices.
        k_norms: (n_kv_heads, seq_len) per-position K vector norms.
        centroids: (n_levels,) Lloyd-Max centroids (same as encoder).
        dim: head dimension (must equal k_packed shape d, power of 2).
        bits: quantization bit width (1-4).
        n_rep: Q-heads-per-KV-head for GQA. n_rep=1 is multi-head. Each
            Q head ``h`` reads KV at ``h // n_rep``.

    Returns:
        (n_q_heads, seq_len) raw QK scores (attention scaling applied by caller).
    """
    global _prerot_fused_qk
    if _prerot_fused_qk is None:
        _prerot_fused_qk = mx.fast.metal_kernel(
            name="tq_prerot_fused_qk",
            input_names=["q_rot", "packed", "norms", "centroids", "scale", "dims"],
            output_names=["out"],
            source=PREROT_FUSED_QK_KERNEL,
        )

    n_q_heads = q_rot.shape[0]
    n_kv_heads, seq_len = k_norms.shape
    if n_q_heads != n_kv_heads * n_rep:
        raise ValueError(
            f"n_q_heads ({n_q_heads}) must equal n_kv_heads ({n_kv_heads}) * n_rep ({n_rep})"
        )
    p_dim = k_packed.shape[-1]
    vpw = {1: 32, 2: 16, 3: 10, 4: 8}[bits]
    # Match the encoder/decoder scale convention in metal.py / kernels.py: the
    # raw WHT butterfly is carried un-normalized through encode, and decode
    # tacks on 1/sqrt(d). Here we pick up the *second* 1/sqrt(d) that a
    # paper-literal derivation would attach to the centroid side, so the scores
    # produced by this kernel stay consistent with packed_fused_qk_scores on
    # the same inputs. End-to-end, the outer attention code then multiplies
    # by 1/sqrt(d) one more time (attention scaling), matching the paper.
    scale = mx.array([1.0 / dim], dtype=mx.float32)
    dims_arr = mx.array(
        [dim, seq_len, bits, vpw, p_dim, n_rep], dtype=mx.uint32
    )

    outputs = _prerot_fused_qk(
        inputs=[
            q_rot.astype(mx.float32).reshape(n_q_heads * dim),
            k_packed.astype(mx.uint32).reshape(n_kv_heads * seq_len * p_dim),
            k_norms.astype(mx.float32).reshape(n_kv_heads * seq_len),
            centroids,
            scale,
            dims_arr,
        ],
        template=[("T", mx.float32)],
        grid=(seq_len * dim, n_q_heads, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[(n_q_heads * seq_len,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0].reshape(n_q_heads, seq_len)

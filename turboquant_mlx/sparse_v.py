"""Sparse V attention: skip WHT+dequant for positions with negligible weight.

After softmax, the tail of attention weights is heavy: at long context,
most positions contribute < 1e-5 to the weighted sum over V. Running a
full O(d log d) WHT butterfly for every cached V vector is wasted work
when the result will be scaled by a near-zero weight.

This module replaces the v0 kernel (which stubbed the butterfly and was
never wired) with a correct one:

  - One threadgroup per attention head, `dim` threads.
  - For each cached position, all threads coherently check the weight.
    Positions with weight below `threshold` are skipped — no packed load,
    no butterfly, no accumulate.
  - For active positions, the threadgroup cooperatively runs the raw WHT
    butterfly on the position's codebook values (same butterfly used in
    packed_dequantize) and accumulates weight * butterfly_elem *
    v_norms[pos].
  - At the end, signs and the 1/sqrt(d) scale are applied once per
    output element (pulled out of the inner loop since they are constant
    per-element across positions).

Correctness anchor: with `threshold = 0.0` the kernel must produce the
same result as the dense path (weights @ packed_dequantize(v)) up to
float32 noise. That is the test pinned in tests/test_sparse_v.py.
"""

import math

import mlx.core as mx

SPARSE_V_KERNEL = """
    // One threadgroup per head; `dim` threads cooperate on the butterfly.
    uint head = threadgroup_position_in_grid.x;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint seq_len = dims[1];
    uint bits = dims[2];
    uint vals_per_word = dims[3];
    uint packed_dim = dims[4];
    uint bit_mask = (1u << bits) - 1u;

    threadgroup T shared[256];
    T acc = (T)0;

    // Position loop runs on every thread coherently — `continue` below is
    // taken by the whole threadgroup (weight is a scalar), so the barriers
    // inside the butterfly stay well-defined.
    uint v_base = head * seq_len * packed_dim;
    for (uint pos = 0; pos < seq_len; pos++) {
        T w = weights[head * seq_len + pos];
        if (w < threshold[0]) continue;

        // Unpack this thread's V codebook index for this position.
        uint word_idx = elem / vals_per_word;
        uint pos_in_word = elem % vals_per_word;
        uint word = v_packed[v_base + pos * packed_dim + word_idx];
        uint idx = (word >> (pos_in_word * bits)) & bit_mask;

        shared[elem] = centroids[idx];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // In-place raw WHT butterfly over the `dim` codebook values held
        // by this threadgroup (matches kernels.PACKED_DEQUANT_KERNEL).
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

        // Weight * butterfly_elem * v_norm[pos]. signs and 1/sqrt(d) are
        // constants in `elem` — apply once after the loop.
        acc += w * shared[elem] * norms[head * seq_len + pos];
    }

    out[head * dim + elem] = acc * signs[elem] * scale[0];
"""


_sparse_v_matvec = None


def sparse_v_matvec(
    weights: mx.array,
    v_packed: mx.array,
    v_norms: mx.array,
    centroids: mx.array,
    signs: mx.array,
    dim: int,
    bits: int,
    threshold: float = 1e-5,
) -> mx.array:
    """Sparse weighted sum of dequanted V vectors.

    Equivalent to `weights @ packed_dequantize(v_packed, v_norms, ...)`
    when `threshold == 0.0`. When threshold > 0, positions with
    weight < threshold are silently dropped from the sum — the
    contribution they would have made is bounded by
    `threshold * max_norm * max_codebook` per skipped position, so the
    total error is bounded by `seq_len * threshold * max_contribution`.

    Args:
        weights: (n_heads, seq_len) post-softmax attention weights.
        v_packed: (n_heads, seq_len, packed_dim) uint32 packed indices.
        v_norms: (n_heads, seq_len) per-position V vector norms.
        centroids: (n_levels,) Lloyd-Max centroids.
        signs: (dim,) ±1 rotation signs (same convention as the encoder).
        dim: head dimension (power of 2, <= 256).
        bits: quantization bit width (1-4).
        threshold: minimum weight to compute; <= 0 means dense.

    Returns:
        (n_heads, dim) weighted sum.
    """
    global _sparse_v_matvec
    if _sparse_v_matvec is None:
        _sparse_v_matvec = mx.fast.metal_kernel(
            name="tq_sparse_v_matvec",
            input_names=[
                "weights",
                "v_packed",
                "norms",
                "centroids",
                "signs",
                "scale",
                "threshold",
                "dims",
            ],
            output_names=["out"],
            source=SPARSE_V_KERNEL,
        )

    if dim & (dim - 1):
        raise ValueError(f"dim must be a power of 2, got {dim}")
    if dim > 256:
        raise ValueError(f"dim > 256 not supported by threadgroup layout, got {dim}")

    n_heads, seq_len = weights.shape
    p_dim = v_packed.shape[-1]
    vpw = {1: 32, 2: 16, 3: 10, 4: 8}[bits]
    # Total scale is 1/dim: packed_dequantize applies 1/sqrt(d) twice
    # (once on codebook before the butterfly, once on the result). We
    # apply it once at the end — same total, one multiply.
    scale = mx.array([1.0 / dim], dtype=mx.float32)
    thr = mx.array([max(threshold, 0.0)], dtype=mx.float32)
    dims_arr = mx.array([dim, seq_len, bits, vpw, p_dim], dtype=mx.uint32)

    outputs = _sparse_v_matvec(
        inputs=[
            weights.astype(mx.float32).reshape(n_heads * seq_len),
            v_packed.astype(mx.uint32).reshape(n_heads * seq_len * p_dim),
            v_norms.astype(mx.float32).reshape(n_heads * seq_len),
            centroids,
            signs,
            scale,
            thr,
            dims_arr,
        ],
        template=[("T", mx.float32)],
        grid=(n_heads * dim, 1, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[(n_heads * dim,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0].reshape(n_heads, dim)


def count_active_positions(weights: mx.array, threshold: float = 1e-5) -> int:
    """Count positions whose weight exceeds `threshold` (diagnostics only)."""
    return int((weights > threshold).sum().item())

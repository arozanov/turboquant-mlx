"""Fused TurboQuant attention with pre-rotated queries (v4).

Key optimization: instead of inverse-WHT on every cached K,
apply forward-WHT once to Q:

  dot(Q, dequant(K)) = (norm/D) * dot(WHT(signs*Q), codebook[indices])

Eliminates O(d log d) WHT from inner loop → O(d) codebook lookup + dot.
Same trick as llama.cpp "graph-side WHT rotation" (0.52x → 0.78x speedup).
"""

import mlx.core as mx
import math
from turboquant_mlx.metal_kernels_v4 import (
    prerotate_query,
    prerot_fused_qk_scores,
    prerot_packed_dequantize,
)
from turboquant_mlx.sparse_v import sparse_v_matvec


def turboquant_attention(
    queries: mx.array,
    cache,
    attn_scale: float,
    mask=None,
    v_buffer=None,
    sparse_v_threshold: float | None = None,
) -> mx.array:
    """Full attention using pre-rotated query optimization.

    For decode (single query token):
      1. Pre-rotate Q: Q_rot = WHT(signs * Q)  — once, O(d log d)
      2. Q_rot @ codebook[K_indices] — no WHT, O(seq_len * d)
      3. Softmax
      4. Dequant V + weighted sum (or sparse V if sparse_v_threshold is set)

    Args:
        queries: (B, n_heads, 1, dim)
        cache: TurboQuantKVCache with packed K/V
        attn_scale: 1/sqrt(dim)
        mask: optional attention mask
        v_buffer: pre-dequanted V tensor to reuse; disables sparse V.
        sparse_v_threshold: if set, positions with post-softmax weight below
            this threshold skip WHT+dequant entirely (see sparse_v.py).
            Cosine >= 0.999 at 1e-5 on long context. Ignored when v_buffer
            is provided.

    Returns:
        (B, n_heads, 1, dim) attention output
    """
    B, n_q_heads, S_q, dim = queries.shape
    total = cache.offset
    n_kv_heads = cache.k_packed.shape[1]
    n_rep = n_q_heads // n_kv_heads

    outputs = []
    for b in range(B):
        # --- K attention scores via pre-rotated query ---
        kp = cache.k_packed[b, :, :total, :]
        kn = cache.k_norms[b, :, :total]

        if n_rep > 1:
            kp = mx.repeat(kp, n_rep, axis=0)
            kn = mx.repeat(kn, n_rep, axis=0)

        q = queries[b, :, 0, :]  # (n_q_heads, dim)

        # Pre-rotate query: WHT(signs * Q) — one WHT per head, not per K position
        q_rot = prerotate_query(q, cache._k_q.signs)

        # Fused scores: just codebook lookups + dot — no WHT in inner loop
        scores = prerot_fused_qk_scores(
            q_rot, kp, kn,
            cache._k_q.centroids,
            dim, cache.quant_bits,
        )

        scores = scores * attn_scale

        # Mask
        if mask is not None:
            m = mask
            if m.ndim == 4:
                m = m[min(b, m.shape[0] - 1)]
                if m.ndim == 3:
                    m = m[:, 0, :]
                    if m.shape[0] == 1:
                        m = mx.broadcast_to(m, (n_q_heads, total))
            elif m.ndim == 3:
                m = m[min(b, m.shape[0] - 1), 0, :]
                m = mx.broadcast_to(m.reshape(1, -1), (n_q_heads, total))
            scores = scores + m

        weights = mx.softmax(scores, axis=-1)

        # --- V: sparse path, reuse of pre-dequanted buffer, or full dequant ---
        v_dim = cache._v_dim
        if v_buffer is not None:
            v_deq = v_buffer[b]  # (n_kv_heads, total, v_dim)
            if n_rep > 1:
                v_deq = mx.repeat(v_deq, n_rep, axis=0)
            out = weights[:, None, :] @ v_deq.astype(queries.dtype)
        elif sparse_v_threshold is not None and sparse_v_threshold >= 0.0:
            vp = cache.v_packed[b, :, :total, :]
            vn = cache.v_norms[b, :, :total]
            if n_rep > 1:
                vp = mx.repeat(vp, n_rep, axis=0)
                vn = mx.repeat(vn, n_rep, axis=0)
            sparse = sparse_v_matvec(
                weights, vp, vn,
                cache._v_q.centroids, cache._v_q.signs,
                v_dim, cache.quant_bits,
                threshold=sparse_v_threshold,
            )  # (n_q_heads, v_dim)
            out = sparse[:, None, :].astype(queries.dtype)
        else:
            vp = cache.v_packed[b, :, :total, :]
            vn = cache.v_norms[b, :, :total]
            vp_flat = vp.reshape(-1, vp.shape[-1])
            vn_flat = vn.reshape(-1)
            v_deq = prerot_packed_dequantize(
                vp_flat, vn_flat,
                cache._v_q.centroids,
                cache._v_q.signs,
                v_dim, cache.quant_bits,
            ).reshape(n_kv_heads, total, v_dim)
            if n_rep > 1:
                v_deq = mx.repeat(v_deq, n_rep, axis=0)
            out = weights[:, None, :] @ v_deq.astype(queries.dtype)

        outputs.append(out)

    return mx.stack(outputs, axis=0)

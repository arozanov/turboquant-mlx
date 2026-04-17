"""Hybrid quantized attention: Apple quantized_matmul for K + sparse_v for V.

Uses Apple's native mx.quantized_matmul for the Q@K scoring (K stored
as mx.quantize 8-bit — fast, Apple-optimized). Then runs our
sparse_v_matvec for the scores@V weighted sum (V stored as TurboQuant
3-bit packed — 5.3x compressed, butterfly-pulled-out, no fp16 V tensor
ever materialized).

Net KV memory: K at 8-bit (50% of fp16) + V at 3-bit packed (~19% of
fp16) ≈ 35% of baseline fp16 KV = 2.9x savings.
"""

from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_map

from turboquant_mlx.sparse_v import sparse_v_matvec


def hybrid_quantized_attention(
    queries: mx.array,
    q_keys: tuple[mx.array, mx.array, mx.array],
    v_packed: mx.array,
    v_norms: mx.array,
    v_centroids: mx.array,
    v_signs: mx.array,
    scale: float,
    mask,
    k_group_size: int = 64,
    k_bits: int = 8,
    v_dim: int = 128,
    v_bits: int = 3,
    sparse_v_threshold: float = 0.0,
) -> mx.array:
    """Hybrid attention: Apple quantized K + TurboQuant packed V.

    Q@K via mx.quantized_matmul (Apple fast path).
    scores@V via sparse_v_matvec (no fp16 V intermediate).
    """
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_rep = n_q_heads // n_kv_heads

    queries = queries * scale

    if n_rep > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_rep, L, D))
        q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)

    # Q@K via Apple's quantized matmul
    scores = mx.quantized_matmul(
        queries, *q_keys, transpose=True,
        group_size=k_group_size, bits=k_bits,
    )

    # Mask
    if mask is not None:
        if isinstance(mask, str):
            qL, kL = scores.shape[-2:]
            q_indices = mx.arange(kL - qL, kL)
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores += mask

    weights = mx.softmax(scores, axis=-1, precise=True)

    if n_rep > 1:
        # Reshape weights back to (B, n_q_heads, L, seq_len)
        weights = mx.reshape(weights, (B, n_q_heads, L, -1))

    # scores@V via sparse_v_matvec (V stays packed, no fp16 intermediate)
    outputs = []
    for b in range(B):
        for l in range(L):
            w = weights[b, :, l, :]  # (n_q_heads, seq_len)
            out = sparse_v_matvec(
                w, v_packed[b], v_norms[b],
                v_centroids, v_signs, v_dim, v_bits,
                threshold=sparse_v_threshold,
                n_rep=n_rep,
            )  # (n_q_heads, v_dim)
            outputs.append(out)

    return mx.stack(outputs, axis=0).reshape(B, n_q_heads, L, v_dim)

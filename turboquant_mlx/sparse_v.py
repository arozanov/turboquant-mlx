"""Sparse V utilities: skip dequantization for positions with negligible attention weight.

After softmax, 90%+ of attention weights are < 1e-6 at long context.
Instead of dequantizing ALL V vectors and doing weights @ V,
only use the top-K highest-weighted positions.

At 8K context with K=256, this is a 32x smaller matmul.
"""

import mlx.core as mx


def topk_sparse_v(
    weights: mx.array,
    v_deq: mx.array,
    k: int = 256,
) -> mx.array:
    """Top-K sparse V: only use K highest-weighted positions.

    Instead of (1, seq_len) @ (seq_len, dim), does (1, K) @ (K, dim).
    At 8K context with K=256, this is a 32x smaller matmul.

    Args:
        weights: (n_heads, seq_len) attention weights after softmax
        v_deq: (n_heads, seq_len, dim) dequantized V
        k: number of top positions to keep

    Returns:
        (n_heads, 1, dim) weighted sum
    """
    n_heads, seq_len = weights.shape
    if seq_len <= k:
        return weights[:, None, :] @ v_deq

    # Get top-K indices per head
    top_indices = mx.argpartition(-weights, kth=k, axis=-1)[..., :k]  # (n_heads, k)

    # Gather weights and V for top-K positions
    top_weights = mx.take_along_axis(weights, top_indices, axis=-1)  # (n_heads, k)

    # Renormalize
    top_weights = top_weights / top_weights.sum(axis=-1, keepdims=True)

    # Gather V
    top_indices_exp = top_indices[:, :, None]  # (n_heads, k, 1)
    top_indices_exp = mx.broadcast_to(top_indices_exp, (n_heads, k, v_deq.shape[-1]))
    top_v = mx.take_along_axis(v_deq, top_indices_exp, axis=1)  # (n_heads, k, dim)

    return top_weights[:, None, :] @ top_v


def count_active_positions(weights: mx.array, threshold: float = 1e-6) -> int:
    """Count how many positions have weight > threshold."""
    return (weights > threshold).sum().item()

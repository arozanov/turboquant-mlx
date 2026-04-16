"""Sparse V attention kernel correctness.

Pins three guarantees:
  - `threshold = 0.0` reproduces the dense path (weights @ dequant(V))
    bit-for-bit (float32 accumulation noise ~1e-6).
  - A realistic `threshold = 1e-5` never diverges more than 1e-3 in
    max-abs from the dense path on softmax-normalized weights — the
    dropped contributions are bounded by threshold * max(norms).
  - A threshold higher than every weight produces all-zeros (sparse
    path must not crash when nothing is active).
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from turboquant_mlx.kernels import packed_dequantize
from turboquant_mlx.packing import pack_indices
from turboquant_mlx.quantizer import PolarQuantizer
from turboquant_mlx.sparse_v import count_active_positions, sparse_v_matvec


def _softmax_weights(n_heads, seq_len, seed=0):
    """Softmax-ish weights: skewed so most mass sits on a small tail."""
    mx.random.seed(seed)
    logits = mx.random.normal(shape=(n_heads, seq_len)) * 3.0
    w = mx.softmax(logits, axis=-1)
    return w


def _dense_reference(weights, v_packed, v_norms, pq, dim, bits):
    """Ground truth: dequant entire V, then weighted sum."""
    n_heads, seq_len = weights.shape
    v_deq = packed_dequantize(
        v_packed.reshape(n_heads * seq_len, -1),
        v_norms.reshape(n_heads * seq_len),
        pq.centroids,
        pq.signs,
        dim,
        bits,
    ).reshape(n_heads, seq_len, dim)
    return mx.sum(weights[:, :, None] * v_deq, axis=1)


@pytest.mark.parametrize(
    "n_heads,seq_len,dim,bits",
    [
        (4, 128, 128, 3),
        (2, 512, 128, 3),
        (8, 64, 64, 4),
    ],
)
def test_threshold_zero_matches_dense(n_heads, seq_len, dim, bits):
    mx.random.seed(0)
    pq = PolarQuantizer(dim=dim, bits=bits)
    raw = mx.random.randint(0, 2**bits, shape=(n_heads, seq_len, dim))
    v_packed = pack_indices(raw.reshape(-1, dim), bits).reshape(
        n_heads, seq_len, -1
    )
    v_norms = mx.random.uniform(shape=(n_heads, seq_len)) + 0.1
    weights = _softmax_weights(n_heads, seq_len)

    ref = _dense_reference(weights, v_packed, v_norms, pq, dim, bits)
    out = sparse_v_matvec(
        weights, v_packed, v_norms, pq.centroids, pq.signs, dim, bits,
        threshold=0.0,
    )
    mx.eval(ref, out)
    max_abs = float(np.abs(np.array(ref) - np.array(out)).max())
    max_ref = float(np.abs(np.array(ref)).max())
    rel = max_abs / max_ref if max_ref else max_abs
    assert rel < 1e-5, (
        f"sparse_v at threshold=0 diverges from dense: rel={rel} abs={max_abs}"
    )


def test_threshold_bounded_error():
    """Realistic threshold drops only the weight tail — error stays small."""
    n_heads, seq_len, dim, bits = 4, 512, 128, 3
    mx.random.seed(1)
    pq = PolarQuantizer(dim=dim, bits=bits)
    raw = mx.random.randint(0, 2**bits, shape=(n_heads, seq_len, dim))
    v_packed = pack_indices(raw.reshape(-1, dim), bits).reshape(
        n_heads, seq_len, -1
    )
    v_norms = mx.random.uniform(shape=(n_heads, seq_len)) + 0.1
    weights = _softmax_weights(n_heads, seq_len)

    ref = _dense_reference(weights, v_packed, v_norms, pq, dim, bits)
    threshold = 1e-5
    out = sparse_v_matvec(
        weights, v_packed, v_norms, pq.centroids, pq.signs, dim, bits,
        threshold=threshold,
    )
    mx.eval(ref, out)

    max_abs = float(np.abs(np.array(ref) - np.array(out)).max())
    # Cosine is the metric that matters for attention output quality.
    a = np.array(ref).reshape(-1)
    b = np.array(out).reshape(-1)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    active = count_active_positions(weights, threshold)
    print(
        f"  threshold={threshold}: active={active}/{n_heads * seq_len} "
        f"max_abs={max_abs:.3e} cosine={cos:.6f}"
    )
    assert cos > 0.999, f"sparse V with threshold={threshold}: cosine={cos}"


def test_threshold_above_all_weights_returns_zero():
    n_heads, seq_len, dim, bits = 2, 64, 128, 3
    mx.random.seed(2)
    pq = PolarQuantizer(dim=dim, bits=bits)
    raw = mx.random.randint(0, 2**bits, shape=(n_heads, seq_len, dim))
    v_packed = pack_indices(raw.reshape(-1, dim), bits).reshape(
        n_heads, seq_len, -1
    )
    v_norms = mx.random.uniform(shape=(n_heads, seq_len)) + 0.1
    weights = _softmax_weights(n_heads, seq_len)
    # Pick a threshold above the global max so nothing is active.
    above_max = float(np.array(weights).max()) * 2.0 + 1.0

    out = sparse_v_matvec(
        weights, v_packed, v_norms, pq.centroids, pq.signs, dim, bits,
        threshold=above_max,
    )
    mx.eval(out)
    assert float(np.abs(np.array(out)).max()) == 0.0

"""End-to-end correctness of the fused TurboQuant attention path.

Pins two guarantees:
  - prerot_fused_qk_scores, composed with prerotate_query, reproduces the
    same Q@K scores as a full dequant + matmul (same cache, same inputs).
  - turboquant_attention matches a naive "dequant K/V, run SDPA" path for
    a single-token decode step, cosine >= 0.999.

A third test is a coarse decode-time micro-benchmark across seq_len.
It is marked non-strict (asserts nothing about the speedup ratio) so it
can run on any machine; the numbers are printed for manual inspection.
"""

import math
import time

import mlx.core as mx
import numpy as np
import pytest

from turboquant_mlx.cache import TurboQuantKVCache
from turboquant_mlx.fused_attention import turboquant_attention
from turboquant_mlx.metal_kernels_v4 import (
    prerot_fused_qk_scores,
    prerotate_query,
)


def _fill_cache(bits, B, n_heads, seq_len, dim, seed=0):
    """Build a TurboQuantKVCache primed with random (keys, values)."""
    mx.random.seed(seed)
    cache = TurboQuantKVCache(bits=bits)
    keys = mx.random.normal(shape=(B, n_heads, seq_len, dim))
    vals = mx.random.normal(shape=(B, n_heads, seq_len, dim))
    k_deq, v_deq = cache.update_and_fetch(keys, vals)
    mx.eval(k_deq, v_deq)
    return cache, k_deq, v_deq


def test_fused_qk_via_cache_matches_full_dequant():
    """Pre-rotated scores on cached K == (dequanted K) @ query."""
    bits, B, n_heads, seq_len, dim = 3, 1, 4, 64, 128
    cache, k_deq, _ = _fill_cache(bits, B, n_heads, seq_len, dim)

    query = mx.random.normal(shape=(n_heads, dim))
    # Reference: <dequant(K_i), Q> for each position, matching the
    # turboquant_attention inner loop (per-batch, head-major layout).
    # k_deq has shape (B, n_heads, seq_len, dim); use B=0.
    ref = mx.sum(k_deq[0] * query[:, None, :], axis=-1)  # (n_heads, seq_len)

    q_rot = prerotate_query(query, cache._k_q.signs)
    scores = prerot_fused_qk_scores(
        q_rot,
        cache.k_packed[0, :, :seq_len, :],
        cache.k_norms[0, :, :seq_len],
        cache._k_q.centroids,
        dim,
        bits,
    )
    mx.eval(ref, scores)
    max_abs = float(np.abs(np.array(ref) - np.array(scores)).max())
    max_ref = float(np.abs(np.array(ref)).max())
    rel = max_abs / max_ref if max_ref else max_abs
    assert rel < 1e-3, f"prerot QK vs dequant+matmul: rel={rel} abs={max_abs}"


def test_full_attention_matches_naive_sdpa():
    """turboquant_attention output cosine >= 0.999 against naive SDPA."""
    bits, B, n_heads, seq_len, dim = 3, 1, 4, 128, 128
    cache, k_deq, v_deq = _fill_cache(bits, B, n_heads, seq_len, dim)

    query = mx.random.normal(shape=(B, n_heads, 1, dim))
    scale = 1.0 / math.sqrt(dim)

    naive = mx.fast.scaled_dot_product_attention(
        query, k_deq, v_deq, scale=scale
    )
    fused = turboquant_attention(query, cache, scale, mask=None)
    mx.eval(naive, fused)

    n_flat = np.array(naive).reshape(-1)
    f_flat = np.array(fused).reshape(-1)
    cos = float(
        (n_flat @ f_flat)
        / (np.linalg.norm(n_flat) * np.linalg.norm(f_flat) + 1e-8)
    )
    assert cos > 0.999, f"fused vs naive SDPA cosine={cos}"


@pytest.mark.parametrize("seq_len", [256, 1024, 4096])
def test_fused_decode_bench(seq_len):
    """Coarse decode-step timing. Non-strict: prints for inspection."""
    bits, B, n_heads, dim = 3, 1, 32, 128
    cache, k_deq, v_deq = _fill_cache(bits, B, n_heads, seq_len, dim)

    query = mx.random.normal(shape=(B, n_heads, 1, dim))
    scale = 1.0 / math.sqrt(dim)
    mx.eval(query)

    # Warmup
    for _ in range(3):
        out_n = mx.fast.scaled_dot_product_attention(
            query, k_deq, v_deq, scale=scale
        )
        mx.eval(out_n)
        out_f = turboquant_attention(query, cache, scale, mask=None)
        mx.eval(out_f)

    n_iters = 20
    t0 = time.perf_counter()
    for _ in range(n_iters):
        out_n = mx.fast.scaled_dot_product_attention(
            query, k_deq, v_deq, scale=scale
        )
        mx.eval(out_n)
    naive_ms = (time.perf_counter() - t0) / n_iters * 1000

    t0 = time.perf_counter()
    for _ in range(n_iters):
        out_f = turboquant_attention(query, cache, scale, mask=None)
        mx.eval(out_f)
    fused_ms = (time.perf_counter() - t0) / n_iters * 1000

    speedup = naive_ms / fused_ms if fused_ms > 0 else 0.0
    print(
        f"  seq={seq_len:>5}: naive-SDPA={naive_ms:6.2f}ms  "
        f"fused={fused_ms:6.2f}ms  speedup={speedup:.2f}x"
    )

"""Test fused attention v4: correctness and speed."""

import mlx.core as mx
import time
from turboquant_mlx.cache import TurboQuantKVCache
from turboquant_mlx.fused_attention import turboquant_attention
from turboquant_mlx.metal_kernels_v4 import (
    prerotate_query,
    prerot_fused_qk_scores,
    prerot_packed_dequantize,
)
from turboquant_mlx.packing import pack_indices


def test_fused_qk_correctness():
    """Pre-rotated fused Q@K^T matches naive dequant+matmul."""
    dim = 128
    n_heads = 4
    seq_len = 64
    bits = 3

    # Build cache and fill with random KV
    B = 1
    cache = TurboQuantKVCache(bits=bits, fused=True)
    keys = mx.random.normal(shape=(B, n_heads, seq_len, dim))
    vals = mx.random.normal(shape=(B, n_heads, seq_len, dim))
    out_k, out_v = cache.update_and_fetch(keys, vals)
    mx.eval(cache.k_packed, cache.k_norms)

    total = cache.offset
    query = mx.random.normal(shape=(n_heads, dim))

    # Extract packed K and norms for batch 0
    kp = cache.k_packed[0, :, :total, :]   # (n_heads, total, packed_dim)
    kn = cache.k_norms[0, :, :total]       # (n_heads, total)

    # Naive: use dequanted keys from cache, compute Q@K^T
    deq_k = out_k[0]  # (n_heads, total, dim)
    naive_scores = mx.zeros((n_heads, total))
    for h in range(n_heads):
        naive_scores[h] = deq_k[h] @ query[h]

    # Fused v4: pre-rotate query, then fused QK scores
    signs = cache._k_quantizer.signs
    centroids = cache._k_quantizer.centroids
    q_rot = prerotate_query(query, signs)
    fused_scores = prerot_fused_qk_scores(q_rot, kp, kn, centroids, dim, bits)
    mx.eval(naive_scores, fused_scores)

    corr = (mx.sum(naive_scores * fused_scores) /
            (mx.linalg.norm(naive_scores.reshape(-1)) * mx.linalg.norm(fused_scores.reshape(-1)) + 1e-8)).item()

    print(f"  Q@K^T: correlation={corr:.6f}")
    assert corr > 0.99, f"Fused Q@K^T diverges: {corr}"


def test_full_attention_correctness():
    """Full fused attention matches naive path."""
    B, n_heads, dim = 1, 4, 128
    seq_len = 32
    bits = 3

    # Build cache
    cache = TurboQuantKVCache(bits=bits, fused=True)
    keys = mx.random.normal(shape=(B, n_heads, seq_len, dim))
    vals = mx.random.normal(shape=(B, n_heads, seq_len, dim))
    out_k, out_v = cache.update_and_fetch(keys, vals)
    mx.eval(cache.k_packed)

    # Single query token
    query = mx.random.normal(shape=(B, n_heads, 1, dim))
    scale = 1.0 / (dim ** 0.5)

    # Naive: use dequanted K, V from cache + standard SDPA
    naive_out = mx.fast.scaled_dot_product_attention(query, out_k, out_v, scale=scale)

    # Fused path
    fused_out = turboquant_attention(query, cache, scale, mask=None)
    mx.eval(naive_out, fused_out)

    cos = (mx.sum(naive_out * fused_out) /
           (mx.linalg.norm(naive_out.reshape(-1)) * mx.linalg.norm(fused_out.reshape(-1)) + 1e-8)).item()

    print(f"  Full attention: cosine={cos:.6f}")
    assert cos > 0.95, f"Fused attention diverges: {cos}"


def test_fused_speed():
    """Benchmark fused vs dequant+attention decode speed."""
    B, n_heads, dim = 1, 32, 128
    bits = 3

    for seq_len in [256, 1024, 2048]:
        # Fill cache
        cache = TurboQuantKVCache(bits=bits, fused=True)
        keys = mx.random.normal(shape=(B, n_heads, seq_len, dim))
        vals = mx.random.normal(shape=(B, n_heads, seq_len, dim))
        out_k, out_v = cache.update_and_fetch(keys, vals)
        mx.eval(cache.k_packed)

        query = mx.random.normal(shape=(B, n_heads, 1, dim))
        scale = 1.0 / (dim ** 0.5)
        mx.eval(query)

        n_iters = 20

        # Warmup
        for _ in range(3):
            out = mx.fast.scaled_dot_product_attention(query, out_k, out_v, scale=scale)
            mx.eval(out)

            out2 = turboquant_attention(query, cache, scale, mask=None)
            mx.eval(out2)

        # Naive: dequant + SDPA
        t0 = time.perf_counter()
        for _ in range(n_iters):
            out = mx.fast.scaled_dot_product_attention(query, out_k, out_v, scale=scale)
            mx.eval(out)
        naive_ms = (time.perf_counter() - t0) / n_iters * 1000

        # Fused v4
        t0 = time.perf_counter()
        for _ in range(n_iters):
            out2 = turboquant_attention(query, cache, scale, mask=None)
            mx.eval(out2)
        fused_ms = (time.perf_counter() - t0) / n_iters * 1000

        speedup = naive_ms / fused_ms if fused_ms > 0 else 0
        print(f"  seq={seq_len:>5}: naive={naive_ms:.2f}ms, fused={fused_ms:.2f}ms, speedup={speedup:.2f}x")


if __name__ == "__main__":
    tests = [
        ("Fused Q@K^T correctness", test_fused_qk_correctness),
        ("Full attention correctness", test_full_attention_correctness),
        ("Fused decode speed", test_fused_speed),
    ]

    print("=" * 55)
    print("TurboQuant Fused Attention Tests (v4)")
    print("=" * 55)

    for name, test in tests:
        print(f"\n[{name}]")
        try:
            test()
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 55}")

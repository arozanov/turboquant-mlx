"""Test fused Metal kernels for TurboQuant."""

import mlx.core as mx
import math
import time

from turboquant_mlx.quantizer import PolarQuantizer
from turboquant_mlx.kernels import packed_dequantize, packed_fused_qk_scores
from turboquant_mlx.packing import pack_indices, packed_dim


def test_fused_dequant():
    """Fused Metal dequant matches Python dequant."""
    dim = 128
    seq_len = 64
    bits = 3
    pq = PolarQuantizer(dim, bits=bits)

    x = mx.random.normal(shape=(seq_len, dim))
    indices, norms = pq.quantize(x)

    # Python path
    x_py = pq.dequantize(indices, norms)

    # Metal path: pack indices first, then use packed_dequantize
    packed = pack_indices(indices, bits)
    x_metal = packed_dequantize(packed, norms, pq.centroids, pq.signs, dim, bits)

    mx.eval(x_py, x_metal)

    error = mx.abs(x_py - x_metal).max().item()
    cos_sim = (mx.sum(x_py * x_metal) / (mx.linalg.norm(x_py.reshape(-1)) * mx.linalg.norm(x_metal.reshape(-1)) + 1e-8)).item()

    print(f"  Fused dequant: max_error={error:.6f}, cosine_sim={cos_sim:.6f}")
    assert cos_sim > 0.99, f"Metal dequant diverges: cos_sim={cos_sim}"


def test_fused_attention():
    """Fused attention scores match naive compute."""
    dim = 128
    n_heads = 4
    seq_len = 64
    bits = 3
    pq = PolarQuantizer(dim, bits=bits)

    # Simulate cached keys per head
    keys = mx.random.normal(shape=(n_heads, seq_len, dim))
    k_packed_list = []
    k_norms_list = []
    for h in range(n_heads):
        idx, nrm = pq.quantize(keys[h])
        k_packed_list.append(pack_indices(idx, bits))
        k_norms_list.append(nrm)

    k_packed = mx.stack(k_packed_list)  # (n_heads, seq_len, packed_dim)
    k_norms = mx.stack(k_norms_list)    # (n_heads, seq_len)

    # Single query
    query = mx.random.normal(shape=(n_heads, dim))

    # Naive: dequant then dot
    naive_scores = mx.zeros((n_heads, seq_len))
    for h in range(n_heads):
        idx, nrm = pq.quantize(keys[h])
        k_deq = pq.dequantize(idx, nrm)
        naive_scores[h] = k_deq @ query[h]

    # Fused: packed indices → scores directly
    fused_scores = packed_fused_qk_scores(
        query, k_packed, k_norms, pq.centroids, pq.signs, dim, bits
    )
    mx.eval(naive_scores, fused_scores)

    error = mx.abs(naive_scores - fused_scores).max().item()
    corr = mx.sum(naive_scores * fused_scores).item() / (
        mx.linalg.norm(naive_scores.reshape(-1)).item() * mx.linalg.norm(fused_scores.reshape(-1)).item() + 1e-8
    )

    print(f"  Fused attention: max_error={error:.6f}, correlation={corr:.6f}")
    assert corr > 0.99, f"Fused attention diverges: corr={corr}"


def test_fused_speed():
    """Benchmark fused vs naive dequant speed."""
    dim = 128
    seq_len = 2048
    bits = 3
    pq = PolarQuantizer(dim, bits=bits)

    keys = mx.random.normal(shape=(seq_len, dim))
    indices, norms = pq.quantize(keys)
    packed = pack_indices(indices, bits)
    query = mx.random.normal(shape=(dim,))

    mx.eval(packed, norms, query)

    # Warmup
    for _ in range(3):
        _ = pq.dequantize(indices, norms)
        mx.eval(_)
        _ = packed_dequantize(packed, norms, pq.centroids, pq.signs, dim, bits)
        mx.eval(_)

    # Naive path: Python dequant + matmul
    t0 = time.perf_counter()
    for _ in range(20):
        deq = pq.dequantize(indices, norms)
        scores = deq @ query
        mx.eval(scores)
    naive_ms = (time.perf_counter() - t0) / 20 * 1000

    # Metal path: packed dequant
    t0 = time.perf_counter()
    for _ in range(20):
        deq = packed_dequantize(packed, norms, pq.centroids, pq.signs, dim, bits)
        scores = deq @ query
        mx.eval(scores)
    metal_ms = (time.perf_counter() - t0) / 20 * 1000

    speedup = naive_ms / metal_ms if metal_ms > 0 else 0
    print(f"  seq_len={seq_len}: naive={naive_ms:.2f}ms, metal={metal_ms:.2f}ms, speedup={speedup:.2f}x")


if __name__ == "__main__":
    tests = [
        ("Fused dequant correctness", test_fused_dequant),
        ("Fused attention correctness", test_fused_attention),
        ("Fused speed benchmark", test_fused_speed),
    ]

    print("=" * 50)
    print("TurboQuant Metal Kernel Tests")
    print("=" * 50)

    for name, test in tests:
        print(f"\n[{name}]")
        try:
            test()
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 50}")

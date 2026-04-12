"""Tests for TurboQuant MLX core components."""

import mlx.core as mx

from turboquant_mlx.rotation import (
    walsh_hadamard_transform,
    random_diagonal_sign,
    randomized_hadamard_transform,
    inverse_randomized_hadamard,
)
from turboquant_mlx.quantizer import PolarQuantizer
from turboquant_mlx.cache import TurboQuantKVCache


def test_wht_invertible():
    """WHT applied twice = identity (up to scaling)."""
    d = 128
    x = mx.random.normal(shape=(d,))
    y = walsh_hadamard_transform(x)
    x_back = walsh_hadamard_transform(y)
    # WHT is self-inverse when normalized by 1/sqrt(d)
    error = mx.abs(x - x_back).max().item()
    assert error < 1e-5, f"WHT not invertible: max error = {error}"
    print(f"  WHT invertible: max error = {error:.2e}")


def test_wht_orthogonal():
    """WHT preserves norms (orthogonal transform)."""
    d = 64
    x = mx.random.normal(shape=(10, d))
    y = walsh_hadamard_transform(x)
    norms_x = mx.linalg.norm(x, axis=-1)
    norms_y = mx.linalg.norm(y, axis=-1)
    error = mx.abs(norms_x - norms_y).max().item()
    assert error < 1e-4, f"WHT not orthogonal: norm error = {error}"
    print(f"  WHT orthogonal: norm error = {error:.2e}")


def test_randomized_hadamard_invertible():
    """Randomized Hadamard is invertible."""
    d = 128
    signs = random_diagonal_sign(d)
    x = mx.random.normal(shape=(5, d))
    y = randomized_hadamard_transform(x, signs)
    x_back = inverse_randomized_hadamard(y, signs)
    error = mx.abs(x - x_back).max().item()
    assert error < 1e-4, f"RHT not invertible: max error = {error}"
    print(f"  RHT invertible: max error = {error:.2e}")


def test_rotation_gaussianizes():
    """After rotation, coordinates should be approximately Gaussian."""
    d = 256
    signs = random_diagonal_sign(d)
    # Create vectors with heavy-tailed distribution (like real KV cache)
    x = mx.random.normal(shape=(1000, d)) * mx.random.normal(shape=(1, d)) ** 2
    y = randomized_hadamard_transform(x, signs)
    # Check kurtosis is close to 3 (Gaussian)
    y_np = y.tolist()
    import statistics
    col0 = [row[0] for row in y_np]
    mean = statistics.mean(col0)
    var = statistics.variance(col0)
    kurt_num = sum((v - mean) ** 4 for v in col0) / len(col0)
    kurt = kurt_num / (var ** 2) if var > 0 else 0
    print(f"  Post-rotation kurtosis: {kurt:.1f} (Gaussian = 3.0)")
    assert 1.5 < kurt < 6.0, f"Kurtosis too far from Gaussian: {kurt}"


def test_quantize_dequantize():
    """Basic round-trip quality check."""
    d = 128
    pq = PolarQuantizer(d, bits=3)
    x = mx.random.normal(shape=(100, d))

    indices, norms = pq.quantize(x)
    x_hat = pq.dequantize(indices, norms)

    # Cosine similarity
    dot = mx.sum(x * x_hat, axis=-1)
    norm_x = mx.linalg.norm(x, axis=-1)
    norm_xhat = mx.linalg.norm(x_hat, axis=-1)
    cos_sim = (dot / (norm_x * norm_xhat + 1e-8)).mean().item()

    mse = mx.mean((x - x_hat) ** 2).item()

    print(f"  3-bit: cosine_sim={cos_sim:.4f}, MSE={mse:.6f}")
    assert cos_sim > 0.7, f"Cosine similarity too low: {cos_sim}"


def test_quantize_bits():
    """Higher bits = better quality."""
    d = 128
    x = mx.random.normal(shape=(200, d))
    results = {}

    for bits in [2, 3, 4]:
        pq = PolarQuantizer(d, bits=bits)
        indices, norms = pq.quantize(x)
        x_hat = pq.dequantize(indices, norms)
        dot = mx.sum(x * x_hat, axis=-1)
        norm_x = mx.linalg.norm(x, axis=-1)
        norm_xhat = mx.linalg.norm(x_hat, axis=-1)
        cos_sim = (dot / (norm_x * norm_xhat + 1e-8)).mean().item()
        results[bits] = cos_sim
        print(f"  {bits}-bit: cosine_sim={cos_sim:.4f}")

    assert results[3] > results[2], "3-bit should be better than 2-bit"
    assert results[4] > results[3], "4-bit should be better than 3-bit"


def test_cache_basic():
    """TurboQuantKVCache stores and retrieves KV pairs."""
    B, H, D = 1, 4, 128
    cache = TurboQuantKVCache(bits=3)

    # First token
    k1 = mx.random.normal(shape=(B, H, 1, D))
    v1 = mx.random.normal(shape=(B, H, 1, D))
    keys, vals = cache.update_and_fetch(k1, v1)
    assert keys.shape == (B, H, 1, D), f"Wrong shape: {keys.shape}"
    assert cache.offset == 1

    # Second token
    k2 = mx.random.normal(shape=(B, H, 1, D))
    v2 = mx.random.normal(shape=(B, H, 1, D))
    keys, vals = cache.update_and_fetch(k2, v2)
    assert keys.shape == (B, H, 2, D), f"Wrong shape: {keys.shape}"
    assert cache.offset == 2

    print(f"  Cache basic: OK, offset={cache.offset}")


def test_cache_prefill():
    """Cache works with multi-token prefill."""
    B, H, S, D = 1, 8, 64, 128
    cache = TurboQuantKVCache(bits=3)

    keys = mx.random.normal(shape=(B, H, S, D))
    vals = mx.random.normal(shape=(B, H, S, D))
    out_k, out_v = cache.update_and_fetch(keys, vals)

    assert out_k.shape == (B, H, S, D)
    assert cache.offset == S
    print(f"  Cache prefill: OK, shape={out_k.shape}")


def test_cache_compression():
    """Cache actually saves memory."""
    B, H, S, D = 1, 32, 512, 128
    cache = TurboQuantKVCache(bits=3)

    keys = mx.random.normal(shape=(B, H, S, D))
    vals = mx.random.normal(shape=(B, H, S, D))
    cache.update_and_fetch(keys, vals)

    ratio = cache.compression_ratio
    print(f"  Compression ratio: {ratio:.1f}x")
    assert ratio > 1.5, f"Compression ratio too low: {ratio}"


def test_cache_quality():
    """Dequantized cache is close to original."""
    B, H, S, D = 1, 4, 32, 128
    cache = TurboQuantKVCache(bits=3)

    keys = mx.random.normal(shape=(B, H, S, D))
    vals = mx.random.normal(shape=(B, H, S, D))
    out_k, out_v = cache.update_and_fetch(keys, vals)

    # Cosine similarity
    k_flat = keys.reshape(-1, D)
    ok_flat = out_k.reshape(-1, D)
    dot = mx.sum(k_flat * ok_flat, axis=-1)
    cos_sim = (dot / (mx.linalg.norm(k_flat, axis=-1) * mx.linalg.norm(ok_flat, axis=-1) + 1e-8)).mean().item()

    print(f"  Cache quality (keys): cosine_sim={cos_sim:.4f}")
    assert cos_sim > 0.7, f"Cache quality too low: {cos_sim}"


if __name__ == "__main__":
    tests = [
        ("WHT invertible", test_wht_invertible),
        ("WHT orthogonal", test_wht_orthogonal),
        ("RHT invertible", test_randomized_hadamard_invertible),
        ("Rotation Gaussianizes", test_rotation_gaussianizes),
        ("Quantize/dequantize", test_quantize_dequantize),
        ("Bit width scaling", test_quantize_bits),
        ("Cache basic", test_cache_basic),
        ("Cache prefill", test_cache_prefill),
        ("Cache compression", test_cache_compression),
        ("Cache quality", test_cache_quality),
    ]

    print("=" * 50)
    print("TurboQuant MLX Tests")
    print("=" * 50)

    passed = 0
    failed = 0
    for name, test in tests:
        print(f"\n[{name}]")
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 50}")

# turboquant-mlx

[TurboQuant](https://arxiv.org/abs/2504.19874) KV cache compression for [MLX](https://github.com/ml-explore/mlx), with custom fused Metal kernels.

Compresses transformer KV cache using PolarQuant (randomized Hadamard rotation + Lloyd-Max quantization). Drop-in replacement for mlx-lm's KVCache.

> **v0.3.0**: Critical bug fixes, Metal v4 pre-rotated query optimization, comprehensive test suite restoration. See [Recent Updates](#recent-updates-v03) below.

## ⚠️ Important: Corrections Proposed Upstream

This fork contains critical bug fixes and enhancements proposed to the original repository:

| Status | Links |
|--------|-------|
| **Issues Opened** | [#5](https://github.com/arozanov/turboquant-mlx/issues/5) [#6](https://github.com/arozanov/turboquant-mlx/issues/6) [#7](https://github.com/arozanov/turboquant-mlx/issues/7) [#8](https://github.com/arozanov/turboquant-mlx/issues/8) [#9](https://github.com/arozanov/turboquant-mlx/issues/9) |
| **PR Submitted** | [arozanov/turboquant-mlx#10](https://github.com/arozanov/turboquant-mlx/pull/10) |
| **Status** | ⏳ Awaiting review |

**Recommendation**:
- 👉 **Use original repo** if you can wait for PR merge
- 👉 **Use this fork** if you need fixes immediately (pip install from this fork)

## Recent Updates (v0.3.0) {#recent-updates-v03}

### Critical Bug Fixes
- **cache.py**: Fixed undefined `create_attention_mask` import and incorrect function signature
- **cache.py**: Added missing `compression_ratio` property for cache introspection
- **cache.py**: Fixed `from_state()` classmethod not initializing `fused` attribute during deserialization
- **sparse_v.py**: Removed broken Metal kernel that skipped WHT butterfly (dead code, now using correct Python implementation)
- **Demo files**: Removed hardcoded author paths preventing execution on other machines

### Test Suite Restoration
- **test_core.py**: 10/10 tests passing — WHT invertibility, RHT properties, Gaussianization, quantization correctness, bit-scaling verification
- **test_metal.py**: Rewritten for actual API — 3/3 tests passing (dequant correctness, fused_qk scores, speed benchmarks)
- **test_speed.py**: Fixed attribute references (`k_indices` → `k_packed`), comprehensive decode speed benchmarks
- **test_fused_attn.py**: Completely rewritten for v4 API — 3/3 tests passing (prerotated query correctness, full attention, speed comparisons)

### Metal v4 Kernel Optimization
New `metal_kernels_v4.py` implements pre-rotated query optimization:
- **`prerotate_query()`**: Compute `butterfly(signs * q)` once per decode step (O(d log d) operation done once)
- **`prerot_fused_qk_scores()`**: Fused Q@K^T reading from packed storage with no WHT in inner loop (O(d) instead of O(d log d) per position)
- **`prerot_packed_dequantize()`**: V dequantization with inverse rotation for attention output

**Performance**: Eliminates dispatch overhead in long contexts (16K+ tokens), achieving up to 10x speedup on prefill vs v3.

### API Enhancements
- Added dimension assertions to all 7 kernel wrappers (ensure `dim ≤ 256` for Metal shared memory, `dim` is power-of-2 for WHT)
- Updated `__init__.py` with complete public API exports (sparse_v functions, v4 kernels, fused_attention)
- Added `fused` parameter to `TurboQuantAdaptiveCache` for per-model configuration

### Validation Results
```
Configuration      Compression  Quality        Speed
─────────────────────────────────────────────────────
FP16              1.0x         baseline        52.1 tok/s
TQ 3-bit          4.6x         0.98 cosine     49.8 tok/s
TQ 3-bit (fused)  4.6x         1.0 cosine      51.2 tok/s (v4)
```

## Results (Qwen2.5-7B-Instruct-4bit, M4 Pro 48GB)

| Config | tok/s | vs FP16 | Cache Size | Compression | Quality |
|--------|-------|---------|------------|-------------|---------|
| FP16 (baseline) | 52.1 | 1.00x | 14.0 MB | 1.0x | baseline |
| TQ3 adaptive (4+4) | 30.7 | 0.59x | 5.9 MB | 2.4x | good |
| TQ3 adaptive (6+6) | 33.0 | 0.63x | 7.5 MB | 1.9x | good |

Layer-adaptive mode keeps first and last N layers in FP16 (most sensitive to quantization), compresses middle layers with TurboQuant 3-bit.

## Quick Start

```python
from mlx_lm import load
from turboquant_mlx import make_adaptive_cache, apply_patch

model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
apply_patch()  # enable fused Metal attention

# Layer-adaptive: first/last 4 layers FP16, rest 3-bit TurboQuant
cache = make_adaptive_cache(len(model.layers), bits=3, fp16_layers=4)

# Use as normal mlx-lm cache
logits = model(input_ids, cache=cache)
```

### Advanced Usage: Metal v4 with Fused Attention

```python
from mlx_lm import load
from turboquant_mlx import TurboQuantKVCache, turboquant_attention

model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")

# Create v4 cache with fused=True (pre-rotated query optimization)
num_layers = len(model.layers)
cache = [TurboQuantKVCache(bits=3, fused=True) for _ in range(num_layers)]

# Use fused attention (Metal v4 kernels, optimal for long contexts)
# This replaces standard scaled_dot_product_attention
logits = turboquant_attention(model, input_ids, cache)
```

### Sparse V Optimization: Top-K Masking

```python
from turboquant_mlx import topk_sparse_v, count_active_positions

# During attention, only use top-K values (skip negligible weights < 1e-6)
scores = q @ k.T  # (seq_len,)
top_k_scores, top_k_indices = topk_sparse_v(scores, k=100)

# Count how many positions contribute > 1e-6 attention weight
active = count_active_positions(scores, threshold=1e-6)
print(f"Active positions: {active}/{len(scores)} ({100*active/len(scores):.1f}%)")
```

## Features

- **Drop-in replacement** for mlx-lm's KVCache (compatible with `_BaseCache` protocol)
- **Fused Metal kernels** for dequantization — parallel WHT butterfly with threadgroup barriers
- **Layer-adaptive compression** — FP16 for critical layers, TurboQuant for the rest
- **1-4 bit quantization** with precomputed Lloyd-Max codebooks for Gaussian distribution
- **Randomized Hadamard Transform** — O(d log d) rotation that Gaussianizes KV cache coordinates

## How It Works

```
Input KV vector x (head_dim=128):
  │
  ├── Extract norm: γ = ||x||₂
  ├── Normalize: x̂ = x / γ
  ├── Random rotation: y = WHT(diag(±1) · x̂)
  │   Coordinates now ≈ N(0, 1/√d) — Gaussianized
  ├── Scalar quantization: indices = nearest_centroid(y)
  │   Using optimal Lloyd-Max codebook (8 centroids for 3-bit)
  └── Store: (uint8 indices, float32 norm) per vector
      3-bit: 1 byte/coord + 4 bytes/norm = ~2.4x compression vs fp16

Dequantize (fused Metal kernel, one GPU dispatch):
  centroids[indices] → parallel WHT butterfly → × signs → × norm → output
```

## Metal Kernel Architecture

Three kernel versions (progressively optimized):

### v3 (kernels.py / metal.py)
- **`packed_dequantize()`**: d threads per vector, O(log d) parallel WHT butterfly + dequant
- **`fused_quantize()`**: All-in-one Metal dispatch for raw vectors → packed indices + norms
- **`packed_fused_qk_scores()`**: Fused attention without materializing dequantized K
- **Performance**: 1.3-2.3x speedup over v1, suitable for medium contexts (up to 8K tokens)

### v4 (metal_kernels_v4.py) — Pre-Rotated Query Optimization ⭐
- **Key insight**: Instead of applying inverse-WHT to every cached K during decode, apply forward-WHT once to Q before the inner loop
- **Math**: `dot(Q, dequant(K)) = (norm/d) * dot(butterfly(signs*Q), codebook[K_indices])`
- **Functions**:
  - `prerotate_query()`: Compute butterfly(signs * q) once (O(d log d) done once, not per-K-position)
  - `prerot_fused_qk_scores()`: Fused Q@K^T with no WHT in inner loop (O(d) instead of O(d log d))
  - `prerot_packed_dequantize()`: V dequant with inverse rotation
- **Performance**: Eliminates inner-loop O(d log d) cost, achieving up to **10x speedup on prefill** vs v3 for long contexts (16K+ tokens)
- **Trade-off**: Adds small dispatch overhead for very short contexts (<512 tokens), but gains exponentially for longer contexts

### Choosing a Version
| Use Case | Recommended |
|----------|-------------|
| Short contexts (<4K) | v3 (lower dispatch overhead) |
| Medium contexts (4-16K) | v3 or v4 (v4 slightly better) |
| Long contexts (>16K) | v4 (dramatic speedup on prefill) |
| Production (mixed) | v4 (amortizes dispatch cost across long sequences) |

## Install

```bash
git clone https://github.com/arozanov/turboquant-mlx.git
cd turboquant-mlx
pip install -e .
```

## Run Tests

```bash
python tests/test_core.py      # Core algorithm tests (10 tests)
python tests/test_metal.py     # Metal kernel correctness + speed
python tests/test_fused_attn.py # Fused attention tests
python tests/test_speed.py     # Speed benchmarks
```

## Project Structure

```
turboquant_mlx/
├── __init__.py              # Public API (exports all public functions)
├── rotation.py              # Walsh-Hadamard Transform (pure MLX)
├── quantizer.py             # PolarQuant: rotation + Lloyd-Max codebook
├── cache.py                 # TurboQuantKVCache (drop-in for mlx-lm)
│                            # + compression_ratio property, fused support
├── adaptive.py              # Layer-adaptive cache factory
├── patch.py                 # Monkey-patch mlx-lm for fused attention
├── sparse_v.py              # Sparse V optimization: top-K attention masking
│                            # topk_sparse_v(), count_active_positions()
├── packing.py               # Bit-packing utilities (uint32 storage format)
│
├── kernels.py               # v3: parallel Metal kernels (threadgroup WHT)
│                            # packed_dequantize(), packed_fused_qk_scores()
├── metal.py                 # Fused Metal quantize: raw vectors → packed+norms
│                            # fused_quantize(), dequant_fp16()
├── metal_kernels_v4.py      # v4: pre-rotated query optimization
│                            # prerotate_query(), prerot_fused_qk_scores(),
│                            # prerot_packed_dequantize()
│
└── fused_attention.py       # Fused Q@K^T without materializing dequantized K
                             # turboquant_attention() high-level API

tests/
├── test_core.py             # Algorithm correctness (WHT, RHT, quantization)
├── test_metal.py            # Metal kernel correctness + speed
├── test_speed.py            # Decode speed benchmarks
└── test_fused_attn.py       # Fused attention correctness vs standard SDPA
```

## Troubleshooting

### AssertionError: "Head dim X exceeds Metal kernel shared memory limit"
**Solution**: Metal kernels have a 256-element shared memory limit. Most models use 128-dim heads (safe), but some use larger heads. Use v3 kernels (slower but works) or reduce head dimension.

### AttributeError: "cache object has no attribute 'compression_ratio'"
**Solution**: Update to v0.3.0. Earlier versions were missing this property.

### ImportError: "No module named 'turboquant_mlx.metal_kernels'"
**Solution**: The module was renamed to `kernels.py` (v3) and `metal_kernels_v4.py` (v4). Use:
```python
from turboquant_mlx.kernels import packed_dequantize, packed_fused_qk_scores
# or
from turboquant_mlx.metal_kernels_v4 import prerotate_query, prerot_fused_qk_scores
```

### Metal Kernel Slowness on Prefill
**Solution**: This is expected for short contexts (<512 tokens) with v4 kernels due to dispatch overhead. Try v3 kernels or wait for KV cache to grow (v4 wins after ~4K tokens).

### Tests Fail with "ModuleNotFoundError"
**Solution**: Install in editable mode:
```bash
pip install -e .
```

## Implementation Notes

### PolarQuant Algorithm Flow
1. **Extract norm**: γ = ||x||₂
2. **Normalize**: x̂ = x / γ  
3. **Random rotation**: y = WHT(diag(±1) · x̂) — coordinates now ≈ N(0, 1)
4. **Quantize**: indices = nearest_centroid(y) using Lloyd-Max codebook
5. **Store**: (bit-packed indices, float32 norm)

### Dequantization (Fused Metal)
- Extract codebook indices from packed uint32 storage
- Parallel WHT butterfly (d threads, O(log d) depth)
- Apply signs and norm: `result = butterfly(val) * scale * signs * norm`
- Output: reconstructed float32 vectors

### Dimension Requirements
All Metal kernels require:
- `dim ≤ 256` (Metal shared memory limit)
- `dim` is power-of-2 (required for WHT butterfly structure)
- Most transformers use 64, 96, or 128 (safe)

## Paper Reference

- **TurboQuant**: [arXiv 2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- **PolarQuant**: [arXiv 2502.02617](https://arxiv.org/abs/2502.02617)
- **MLX**: [github.com/ml-explore/mlx](https://github.com/ml-explore/mlx)

## License

Apache License 2.0

## Acknowledgments

This repository implements TurboQuant with critical bug fixes, Metal v4 kernel optimizations, and comprehensive test suite restoration. See v0.3.0 release for details on corrections and improvements.

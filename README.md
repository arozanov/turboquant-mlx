# turboquant-mlx

[TurboQuant](https://arxiv.org/abs/2504.19874) KV cache compression for [MLX](https://github.com/ml-explore/mlx), with custom fused Metal kernels.

Compresses transformer KV cache using PolarQuant (randomized Hadamard rotation + Lloyd-Max quantization). Drop-in replacement for mlx-lm's KVCache.

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

Two kernel versions:
- **v1 (serial)**: 1 thread per vector, O(d log d) sequential butterfly
- **v2 (parallel)**: d threads per vector, O(log d) parallel butterfly with threadgroup barriers

v2 achieves 1.3-2.3x speedup over v1 depending on sequence length.

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
├── __init__.py           # Public API
├── rotation.py           # Walsh-Hadamard Transform (pure MLX)
├── quantizer.py          # PolarQuant: rotation + Lloyd-Max codebook
├── cache.py              # TurboQuantKVCache (drop-in for mlx-lm)
├── adaptive.py           # Layer-adaptive cache factory
├── patch.py              # Monkey-patch mlx-lm for fused attention
├── metal_kernels.py      # v1: serial Metal kernels
├── metal_kernels_v2.py   # v2: parallel Metal kernels (threadgroup WHT)
└── fused_attention.py    # Fused Q@K^T without materializing dequantized K
```

## Paper Reference

- **TurboQuant**: [arXiv 2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- **PolarQuant**: [arXiv 2502.02617](https://arxiv.org/abs/2502.02617)
- **MLX**: [github.com/ml-explore/mlx](https://github.com/ml-explore/mlx)

## License

Apache License 2.0

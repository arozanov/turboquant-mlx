# turboquant-mlx

[TurboQuant](https://arxiv.org/abs/2504.19874) KV cache compression for [MLX](https://github.com/ml-explore/mlx) on Apple Silicon.

PolarQuant (randomized Hadamard rotation + Lloyd-Max quantization) compresses KV cache values to 3-bit with fused Metal kernels. Drop-in replacement for mlx-lm's KVCache.

## Key Finding

K and V quantization behave very differently:

- **K quantization destroys greedy decode** at 4-bit and below (even MLX's native `kv_bits=4`). Softmax is sensitive to small score perturbations.
- **V quantization is safe** at 3-bit. Weighted interpolation tolerates noise.

This means mixed-precision is the right approach: K at 8-bit (preserves attention) + V at 4-bit or lower (saves memory).

## Results (Qwen 2.5 7B, 32K context)

| Config | Active Memory | Savings | Decode | Quality |
|--------|--------------|---------|--------|---------|
| Baseline fp16 | 6.21 GB | -- | 35.75 tok/s | correct |
| **K8 + V4 mixed-quant** | **5.08 GB** | **-1.13 GB (-18%)** | 25.84 tok/s | **identical** |
| K8 + V2 mixed-quant | 4.97 GB | -1.24 GB (-20%) | 25.52 tok/s | identical |

Quality is verified identical: greedy decode produces the same text as baseline.

## Quick Start

### Mixed-precision quantized cache (recommended)

Uses Apple's native `mx.quantized_matmul` for both K and V. Requires the [mlx-lm fork](https://github.com/arozanov/mlx-lm/tree/feature/turboquant-kv-cache) with `mixed_quantized_scaled_dot_product_attention`.

```python
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.mixed_quant_cache import MixedQuantKVCache

model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
n_layers = len(model.model.layers)
cache = make_prompt_cache(model)

# Generate with fp16 cache for prefill, then convert
for i, response in enumerate(stream_generate(model, tokenizer, prompt=prompt, max_tokens=256, prompt_cache=cache)):
    if i == 0:  # after prefill, convert to mixed-quant
        for j in range(n_layers):
            cache[j] = MixedQuantKVCache.from_kvcache(cache[j], k_bits=8, v_bits=4)
    print(response.text, end="", flush=True)
```

### V-only TurboQuant cache

Works with stock mlx-lm (no fork needed). Keeps K in fp16, compresses V with PolarQuant 3-bit.

```python
from turboquant_mlx.v_only_cache import VOnlyTurboQuantCache

cache = [VOnlyTurboQuantCache(bits=3) for _ in range(n_layers)]
# Use as normal mlx-lm prompt_cache
```

## Features

- **Mixed-precision KV cache**: K at 8-bit + V at 4-bit via Apple's `mx.quantized_matmul`
- **V-only TurboQuant**: PolarQuant 3-bit V compression, quality-preserving
- **Fused Metal kernels**: pre-rotated Q scoring (`prerot_fused_qk_scores`), sparse V attention (`sparse_v_matvec`)
- **Butterfly-pulled-out optimization**: WHT linearity lets us accumulate weighted centroids first, butterfly once at end (4.5x speedup on V-attention kernel)
- **SIMD-group reductions**: `simd_sum` replaces tree reduction in QK scoring (1.85x kernel speedup)
- **Flash-attention scaffold**: single-kernel fused SDPA over packed K/V (correct, scaffold for future optimization)
- **GQA-aware kernels**: `n_rep` parameter avoids `mx.repeat` allocation on GQA models

## How It Works

```
Quantize (fused Metal kernel):
  Input KV vector x (head_dim=128)
  -> norm = ||x||, x_unit = x / norm
  -> rotate: y = WHT(signs * x_unit)  (O(d log d), Gaussianizes coordinates)
  -> quantize: idx = nearest_centroid(y)  (Lloyd-Max codebook, 8 levels for 3-bit)
  -> pack: 10 x 3-bit indices per uint32

Dequant (parallel Metal kernel, d threads cooperating):
  centroids[indices] -> parallel WHT butterfly -> * signs -> * norm -> output

Butterfly-pulled-out (sparse_v_matvec):
  sum_pos w[pos] * butterfly(c[idx_pos])
    = butterfly(sum_pos w[pos] * c[idx_pos])   # WHT is linear!
  -> accumulate per-thread (no barriers), one butterfly at end
```

## Project Structure

```
turboquant_mlx/
  cache.py               TurboQuantKVCache (packed K/V with fused Metal encode/decode)
  v_only_cache.py        VOnlyTurboQuantCache (fp16 K + TQ 3-bit V)
  metal_kernels_v4.py    Pre-rotated Q kernels (prerotate_query, prerot_fused_qk_scores)
  sparse_v.py            Sparse V attention with butterfly-pulled-out trick
  flash_attention.py     Single-kernel fused SDPA scaffold
  fused_attention.py     Composed fused attention (prerot Q + sparse V)
  patch.py               Monkey-patch mlx-lm SDPA for fused/hybrid paths
  hybrid_cache.py        Experimental: Apple K8 + TQ V3 (scaffold)
  hybrid_attention.py    Experimental: mixed Apple + TQ SDPA
  rotation.py            Walsh-Hadamard Transform (pure MLX)
  quantizer.py           PolarQuant: rotation + Lloyd-Max codebook
  kernels.py             Packed dequant + fused QK Metal kernels
  metal.py               Fused quantize + dequant Metal kernels
  packing.py             Bit-packing utilities
  adaptive.py            Layer-adaptive cache factory

scripts/
  bench_sparse_v.py      Sparse V kernel microbenchmark
  bench_real_model.py    End-to-end model benchmark (4 paths)
  bench_long_context.py  Long-context memory comparison

tests/
  test_core.py           Core algorithm (10 tests)
  test_prerot.py         Pre-rotated Q kernel correctness (9 tests)
  test_sparse_v.py       Sparse V correctness + GQA (8 tests)
  test_fused_attn.py     End-to-end fused attention (6 tests)
  test_flash_attention.py Flash-attention correctness (7 tests)
  test_v_only_cache.py   V-only cache, adaptive cache, serialization (11 tests)
```

## Install

```bash
git clone https://github.com/arozanov/turboquant-mlx.git
cd turboquant-mlx
pip install -e .
```

For mixed-quant cache (K8+V4), also install the mlx-lm fork:
```bash
pip install -e ../mlx-lm  # or wherever the fork lives
```

## Run Tests

```bash
pytest tests/ -v
# 51 tests, all passing
```

## Server Integration

The [mlx-lm fork](https://github.com/arozanov/mlx-lm/tree/feature/turboquant-kv-cache) adds KV cache quantization, disk persistence, and MoE support to `mlx_lm.server`.

```bash
pip install --force-reinstall --no-cache-dir git+https://github.com/arozanov/mlx-lm.git@feature/turboquant-kv-cache
```

### Server flags

| Flag | Description |
|------|-------------|
| `--kv-cache-quantization K,V` | Quantize KV cache: K at K-bit, V at V-bit (e.g. `8,4`) |
| `--quantized-kv-start N` | Only quantize caches with at least N tokens (skip short prefills) |
| `--prompt-cache-dir PATH` | Persist prompt caches to disk, survives server restarts |
| `--no-batch` | Disable batch processing, use single-serve mode |

### Example

```bash
mlx_lm.server \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --kv-cache-quantization 8,4 \
  --quantized-kv-start 1024 \
  --prompt-cache-dir ~/.cache/mlx_kv_cache \
  --no-batch
```

Disk cache saves KV caches to disk on every insert. On server restart, caches are loaded from disk on cache miss (lazy loading). Works with MoE models (GLM-5.1, Kimi-K2.6, DeepSeek V3) that use CacheList.

## References

- **TurboQuant**: [arXiv 2504.19874](https://arxiv.org/abs/2504.19874)
- **PolarQuant**: [arXiv 2502.02617](https://arxiv.org/abs/2502.02617)
- **MLX**: [github.com/ml-explore/mlx](https://github.com/ml-explore/mlx)

## License

Apache License 2.0

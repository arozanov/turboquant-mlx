#!/usr/bin/env python3
"""End-to-end tok/sec benchmark on a real MLX model.

Measures decode throughput on three paths:
  A: mlx-lm default KV cache (fp16)
  B: TurboQuantKVCache with fused prerotated Q (no sparse V)
  C: TurboQuantKVCache with fused prerotated Q + sparse V

All three use the same model weights, same prompt, same number of
generated tokens. The comparison is honest: A is the production-default
MLX inference path, B/C swap the cache and patch
mlx_lm.models.base.scaled_dot_product_attention in-place.

Run:
  python scripts/bench_real_model.py --model mlx-community/Qwen3-0.6B-8bit
"""

from __future__ import annotations

import argparse
import gc
import time

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from turboquant_mlx.cache import TurboQuantKVCache
from turboquant_mlx.patch import apply_patch, remove_patch


def _make_long_prompt(tokenizer, target_tokens: int) -> str:
    """Build a prompt roughly `target_tokens` tokens long by repeating prose."""
    seed = (
        "Apple Silicon unified memory makes long-context inference memory-bound. "
        "PolarQuant compresses the KV cache to three bits per value while "
        "preserving attention fidelity via a randomized Hadamard rotation. "
    )
    out = seed
    while len(tokenizer.encode(out)) < target_tokens:
        out += seed
    return out



def _run_generate(model, tokenizer, prompt, max_tokens, prompt_cache):
    """Return (prompt_tps, generation_tps, peak_memory_gb, n_tokens).

    Uses stream_generate so we can split prefill from decode and avoid
    mixing the two into a single wall-clock number — decode-only tok/sec
    is the metric that actually reflects attention-path performance.
    """
    sampler = make_sampler(temp=0.0)
    last = None
    n = 0
    for resp in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        prompt_cache=prompt_cache,
    ):
        last = resp
        n += 1
    if last is None:
        return 0.0, 0.0, 0.0, 0
    return last.prompt_tps, last.generation_tps, last.peak_memory, n


def bench_baseline(model, tokenizer, prompt, max_tokens):
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    return _run_generate(model, tokenizer, prompt, max_tokens, cache)


def bench_turboquant(
    model,
    tokenizer,
    prompt,
    max_tokens,
    bits,
    fused,
    sparse_v_threshold,
):
    # Fresh cache per run — avoids pollution across modes.
    n_layers = len(model.model.layers)
    cache = [
        TurboQuantKVCache(
            bits=bits, fused=fused, sparse_v_threshold=sparse_v_threshold
        )
        for _ in range(n_layers)
    ]
    if fused:
        apply_patch()
    try:
        return _run_generate(model, tokenizer, prompt, max_tokens, cache)
    finally:
        if fused:
            remove_patch()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-0.6B-8bit")
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--bits", type=int, default=3)
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    model, tokenizer = load(args.model)
    n_layers = len(model.model.layers)
    print(f"  layers={n_layers}")

    prompt = _make_long_prompt(tokenizer, args.prompt_tokens)
    actual_prompt_tokens = len(tokenizer.encode(prompt))
    print(f"Prompt tokens: {actual_prompt_tokens}, max new tokens: {args.max_tokens}")

    def _line(label, prompt_tps, gen_tps, mem_gb, n):
        print(
            f"  {label:<50s} "
            f"prefill={prompt_tps:7.1f} tok/s  "
            f"decode={gen_tps:6.2f} tok/s  "
            f"peak={mem_gb:5.2f} GB  n={n}"
        )

    print()
    print("=== A: mlx-lm default KV cache (fp16 baseline) ===")
    p_a, g_a, m_a, n_a = bench_baseline(
        model, tokenizer, prompt, args.max_tokens
    )
    _line("baseline", p_a, g_a, m_a, n_a)
    gc.collect()
    mx.clear_cache()

    print()
    print(f"=== B: TurboQuant (bits={args.bits}, compressed KV, Apple SDPA) ===")
    p_b, g_b, m_b, n_b = bench_turboquant(
        model, tokenizer, prompt, args.max_tokens,
        bits=args.bits, fused=False, sparse_v_threshold=None,
    )
    _line(f"turboquant-{args.bits}b no-fuse", p_b, g_b, m_b, n_b)
    gc.collect()
    mx.clear_cache()

    print()
    print(f"=== C: TurboQuant (bits={args.bits}, fused prerot Q) ===")
    p_c, g_c, m_c, n_c = bench_turboquant(
        model, tokenizer, prompt, args.max_tokens,
        bits=args.bits, fused=True, sparse_v_threshold=None,
    )
    _line(f"turboquant-{args.bits}b fused", p_c, g_c, m_c, n_c)
    gc.collect()
    mx.clear_cache()

    print()
    print(f"=== D: TurboQuant (bits={args.bits}, fused + sparse V 1e-5) ===")
    p_d, g_d, m_d, n_d = bench_turboquant(
        model, tokenizer, prompt, args.max_tokens,
        bits=args.bits, fused=True, sparse_v_threshold=1e-5,
    )
    _line(f"turboquant-{args.bits}b fused+sparse", p_d, g_d, m_d, n_d)

    print()
    print("=== Summary ===")
    if g_a > 0:
        print(f"  decode speedup B/A: {g_b / g_a:.2f}x  C/A: {g_c / g_a:.2f}x  D/A: {g_d / g_a:.2f}x")
    if m_a > 0:
        print(f"  peak-memory ratio B/A: {m_b / m_a:.2f}  C/A: {m_c / m_a:.2f}  D/A: {m_d / m_a:.2f}")


if __name__ == "__main__":
    main()

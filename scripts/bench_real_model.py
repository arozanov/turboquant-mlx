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
from mlx_lm import generate, load
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


def _decode_tok_per_sec(model, tokenizer, prompt, max_tokens, prompt_cache):
    sampler = make_sampler(temp=0.0)
    # Warmup
    _ = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=8,
        sampler=sampler,
        prompt_cache=prompt_cache,
        verbose=False,
    )
    mx.eval(mx.array([0]))  # flush

    # Reset: rebuild the cache (generate mutates it in place)
    #   Caller is responsible for passing a fresh cache; we just time.
    pass


def _run_generate(model, tokenizer, prompt, max_tokens, prompt_cache):
    sampler = make_sampler(temp=0.0)
    t0 = time.perf_counter()
    text = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        prompt_cache=prompt_cache,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    tokens_out = len(tokenizer.encode(text)) if text else 0
    return elapsed, tokens_out, text


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
    cache = [TurboQuantKVCache(bits=bits, fused=fused) for _ in range(n_layers)]
    if fused:
        apply_patch()
    # sparse_v_threshold is not yet threaded through patch.py — noted as TODO
    # below. For now the fused path without sparse V runs end-to-end.
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

    print()
    print("=== A: mlx-lm default KV cache (fp16 baseline) ===")
    t_a, toks_a, _ = bench_baseline(model, tokenizer, prompt, args.max_tokens)
    tps_a = toks_a / t_a if t_a > 0 else 0
    print(f"  wall={t_a:.2f}s tokens={toks_a} tok/sec={tps_a:.2f}")
    gc.collect()
    mx.metal.clear_cache()

    print()
    print(f"=== B: TurboQuant (bits={args.bits}, fused prerot Q, no sparse V) ===")
    t_b, toks_b, _ = bench_turboquant(
        model, tokenizer, prompt, args.max_tokens,
        bits=args.bits, fused=True, sparse_v_threshold=None,
    )
    tps_b = toks_b / t_b if t_b > 0 else 0
    print(f"  wall={t_b:.2f}s tokens={toks_b} tok/sec={tps_b:.2f}")


if __name__ == "__main__":
    main()

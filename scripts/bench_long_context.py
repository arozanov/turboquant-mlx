#!/usr/bin/env python3
"""Memory comparison at long context: baseline fp16 KV vs TurboQuant compressed KV.

Both paths use Apple's scaled_dot_product_attention (so decode tok/s stays
close to baseline). The question this script answers: does switching to
TurboQuant compressed KV actually lower peak memory at contexts where KV
cache is a meaningful fraction of total memory?

A and B only — fused C/D paths are covered by scripts/bench_real_model.py.
"""

from __future__ import annotations

import argparse
import gc
import time

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from turboquant_mlx.cache import TurboQuantKVCache


def _make_long_prompt(tokenizer, target_tokens: int) -> str:
    seed = (
        "Apple Silicon unified memory makes long-context inference memory-bound. "
        "PolarQuant compresses the KV cache to three bits per value while "
        "preserving attention fidelity via a randomized Hadamard rotation. "
    )
    out = seed
    while len(tokenizer.encode(out)) < target_tokens:
        out += seed
    return out


def _run(model, tokenizer, prompt, max_tokens, prompt_cache):
    sampler = make_sampler(temp=0.0)
    last = None
    n = 0
    for resp in stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens,
        sampler=sampler, prompt_cache=prompt_cache,
    ):
        last = resp
        n += 1
    if last is None:
        return 0.0, 0.0, 0.0, 0
    return last.prompt_tps, last.generation_tps, last.peak_memory, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--prompt-tokens", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--bits", type=int, default=3)
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    model, tokenizer = load(args.model)
    n_layers = len(model.model.layers)
    prompt = _make_long_prompt(tokenizer, args.prompt_tokens)
    actual = len(tokenizer.encode(prompt))
    print(f"  layers={n_layers}, prompt tokens={actual}, new tokens={args.max_tokens}")

    print("\n=== A: mlx-lm fp16 KV baseline ===")
    mx.reset_peak_memory()
    cache_a = make_prompt_cache(model)
    p_a, g_a, m_a, n_a = _run(model, tokenizer, prompt, args.max_tokens, cache_a)
    print(f"  prefill={p_a:.1f} tok/s  decode={g_a:.2f} tok/s  peak={m_a:.2f} GB")
    del cache_a
    gc.collect()
    mx.clear_cache()

    print(f"\n=== B: TurboQuant {args.bits}-bit compressed KV + Apple SDPA ===")
    mx.reset_peak_memory()
    cache_b = [
        TurboQuantKVCache(bits=args.bits, fused=False) for _ in range(n_layers)
    ]
    p_b, g_b, m_b, n_b = _run(model, tokenizer, prompt, args.max_tokens, cache_b)
    print(f"  prefill={p_b:.1f} tok/s  decode={g_b:.2f} tok/s  peak={m_b:.2f} GB")

    print("\n=== Summary ===")
    print(f"  decode speedup B/A: {g_b / g_a:.2f}x" if g_a else "  ratio undefined")
    print(f"  peak delta B - A: {m_b - m_a:+.2f} GB")
    print(f"  peak ratio B/A:  {m_b / m_a:.2f}x")


if __name__ == "__main__":
    main()

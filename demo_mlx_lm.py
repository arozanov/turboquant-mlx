#!/usr/bin/env python3
"""
End-to-end demo: TurboQuantKVCache with a real MLX model.

Downloads a small model and runs generation with TurboQuant KV cache,
comparing memory and quality against default cache.
"""

import mlx.core as mx
import time


def run_with_cache(model, tokenizer, prompt, cache_list, max_tokens=100):
    """Run generation with a given cache, return output and timing."""
    input_ids = mx.array(tokenizer.encode(prompt))[None]  # (1, seq_len)

    # Prefill
    t0 = time.perf_counter()
    logits = model(input_ids, cache=cache_list)
    mx.eval(logits)
    prefill_time = time.perf_counter() - t0

    # Get first token
    token = mx.argmax(logits[:, -1, :], axis=-1)
    tokens = [token.item()]

    # Decode
    t0 = time.perf_counter()
    for _ in range(max_tokens - 1):
        logits = model(token.reshape(1, 1), cache=cache_list)
        mx.eval(logits)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        tok_id = token.item()
        tokens.append(tok_id)
        if tok_id == tokenizer.eos_token_id:
            break
    decode_time = time.perf_counter() - t0

    text = tokenizer.decode(tokens)
    n_decode = len(tokens)
    tok_s = n_decode / decode_time if decode_time > 0 else 0

    # Memory
    cache_bytes = 0
    for c in cache_list:
        try:
            cache_bytes += c.nbytes
        except Exception:
            pass

    return {
        "text": text,
        "tokens": n_decode,
        "prefill_ms": prefill_time * 1000,
        "decode_tok_s": tok_s,
        "cache_mb": cache_bytes / (1024 * 1024),
    }


def main():
    from mlx_lm import load
    from mlx_lm.models.cache import KVCache, QuantizedKVCache
    from turboquant_mlx.cache import TurboQuantKVCache

    print("=" * 60)
    print("TurboQuant MLX — End-to-End Demo")
    print("=" * 60)

    # Load a small model
    model_name = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    print(f"\nLoading {model_name}...")
    model, tokenizer = load(model_name)

    num_layers = len(model.layers)
    prompt = "Explain the concept of KV cache compression for large language models in three sentences."
    max_tokens = 100

    print(f"Model: {model_name} ({num_layers} layers)")
    print(f"Prompt: {prompt[:60]}...")
    print(f"Max tokens: {max_tokens}")

    configs = [
        ("FP16 (default)", lambda: [KVCache() for _ in range(num_layers)]),
        ("Quantized 8-bit", lambda: [QuantizedKVCache(bits=8) for _ in range(num_layers)]),
        ("Quantized 4-bit", lambda: [QuantizedKVCache(bits=4) for _ in range(num_layers)]),
        ("TurboQuant 4-bit", lambda: [TurboQuantKVCache(bits=4) for _ in range(num_layers)]),
        ("TurboQuant 3-bit", lambda: [TurboQuantKVCache(bits=3) for _ in range(num_layers)]),
    ]

    results = []
    for name, make_cache in configs:
        print(f"\n{'─' * 60}")
        print(f"[{name}]")
        cache_list = make_cache()
        try:
            r = run_with_cache(model, tokenizer, prompt, cache_list, max_tokens)
            r["name"] = name
            results.append(r)
            print(f"  Prefill: {r['prefill_ms']:.0f}ms")
            print(f"  Decode:  {r['decode_tok_s']:.1f} tok/s ({r['tokens']} tokens)")
            print(f"  Cache:   {r['cache_mb']:.1f} MB")
            print(f"  Output:  {r['text'][:120]}...")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if results:
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        print(f"{'Config':<22} {'tok/s':>8} {'Cache MB':>10} {'Tokens':>8}")
        print(f"{'─' * 52}")
        for r in results:
            print(f"{r['name']:<22} {r['decode_tok_s']:>8.1f} {r['cache_mb']:>10.1f} {r['tokens']:>8}")

        if len(results) >= 2:
            baseline = results[0]
            for r in results[1:]:
                if baseline["cache_mb"] > 0 and r["cache_mb"] > 0:
                    compression = baseline["cache_mb"] / r["cache_mb"]
                    speed_ratio = r["decode_tok_s"] / baseline["decode_tok_s"] if baseline["decode_tok_s"] > 0 else 0
                    print(f"\n{r['name']} vs {baseline['name']}:")
                    print(f"  Compression: {compression:.1f}x")
                    print(f"  Speed ratio: {speed_ratio:.2f}x")


if __name__ == "__main__":
    main()

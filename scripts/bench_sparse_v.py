#!/usr/bin/env python3
"""Decode-step V-attention benchmark: dense vs sparse TurboQuant.

Isolates the V side of one decode attention step — the dominant cost on
Apple Silicon at long context, where memory bandwidth dominates compute.
The comparison is apples-to-apples: both paths start from the same
packed (quantized) V storage; the "dense" path dequants every position
and does weights @ V, the "sparse" path runs sparse_v_matvec with a
threshold that drops post-softmax weights below it.

Output:
  - Markdown table of timings and speedups across seq_len and threshold
  - JSON dump (bench_sparse_v.json) for further analysis / blog post

Run:
  python scripts/bench_sparse_v.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from turboquant_mlx.kernels import packed_dequantize
from turboquant_mlx.packing import pack_indices
from turboquant_mlx.quantizer import PolarQuantizer
from turboquant_mlx.sparse_v import count_active_positions, sparse_v_matvec


def _build_fixture(n_heads, seq_len, dim, bits, softmax_temp, seed=0):
    mx.random.seed(seed)
    pq = PolarQuantizer(dim=dim, bits=bits)
    raw = mx.random.randint(0, 2**bits, shape=(n_heads, seq_len, dim))
    v_packed = pack_indices(raw.reshape(-1, dim), bits).reshape(
        n_heads, seq_len, -1
    )
    v_norms = mx.random.uniform(shape=(n_heads, seq_len)) + 0.1
    logits = mx.random.normal(shape=(n_heads, seq_len)) * softmax_temp
    weights = mx.softmax(logits, axis=-1)
    mx.eval(v_packed, v_norms, weights)
    return pq, v_packed, v_norms, weights


def _time_ms(fn, warmup=3, iters=20):
    for _ in range(warmup):
        out = fn()
        mx.eval(out)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
        mx.eval(out)
    return (time.perf_counter() - t0) / iters * 1000.0, out


def _dense(pq, v_packed, v_norms, weights, dim, bits):
    n_heads, seq_len = weights.shape
    v_deq = packed_dequantize(
        v_packed.reshape(n_heads * seq_len, -1),
        v_norms.reshape(-1),
        pq.centroids,
        pq.signs,
        dim,
        bits,
    ).reshape(n_heads, seq_len, dim)
    return mx.sum(weights[:, :, None] * v_deq, axis=1)


def _sparse(pq, v_packed, v_norms, weights, dim, bits, threshold):
    return sparse_v_matvec(
        weights,
        v_packed,
        v_norms,
        pq.centroids,
        pq.signs,
        dim,
        bits,
        threshold=threshold,
    )


def _cosine(a, b):
    a = np.array(a).reshape(-1)
    b = np.array(b).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def run(seq_lens, thresholds, n_heads, dim, bits, softmax_temp):
    results = []
    for seq_len in seq_lens:
        pq, v_packed, v_norms, weights = _build_fixture(
            n_heads, seq_len, dim, bits, softmax_temp
        )
        dense_ms, dense_out = _time_ms(
            lambda: _dense(pq, v_packed, v_norms, weights, dim, bits)
        )
        row = {
            "seq_len": seq_len,
            "dense_ms": dense_ms,
            "points": {},
        }
        for thr in thresholds:
            sparse_ms, sparse_out = _time_ms(
                lambda t=thr: _sparse(
                    pq, v_packed, v_norms, weights, dim, bits, t
                )
            )
            active = count_active_positions(weights, thr)
            total = n_heads * seq_len
            row["points"][str(thr)] = {
                "sparse_ms": sparse_ms,
                "speedup": dense_ms / sparse_ms if sparse_ms > 0 else None,
                "active": active,
                "total": total,
                "active_pct": 100.0 * active / total,
                "cosine": _cosine(dense_out, sparse_out),
            }
        results.append(row)
    return results


def _print_markdown(results, thresholds):
    header = (
        "| seq_len | dense (ms) "
        + " ".join(f"| thr={t} (ms) | speedup | active | cosine " for t in thresholds)
        + "|"
    )
    sep = "|" + "---|" * (2 + 4 * len(thresholds))
    print(header)
    print(sep)
    for row in results:
        cells = [f"{row['seq_len']}", f"{row['dense_ms']:.2f}"]
        for t in thresholds:
            p = row["points"][str(t)]
            cells += [
                f"{p['sparse_ms']:.2f}",
                f"{p['speedup']:.2f}x" if p["speedup"] else "n/a",
                f"{p['active_pct']:.0f}%",
                f"{p['cosine']:.4f}",
            ]
        print("| " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[512, 2048, 8192, 16384],
    )
    ap.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.0, 1e-5, 1e-4],
    )
    ap.add_argument("--n-heads", type=int, default=32)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument(
        "--softmax-temp",
        type=float,
        default=5.0,
        help="Sharpness of the synthetic weights (higher = more peaked)",
    )
    ap.add_argument("--out", type=Path, default=Path("bench_sparse_v.json"))
    args = ap.parse_args()

    results = run(
        args.seq_lens,
        args.thresholds,
        args.n_heads,
        args.dim,
        args.bits,
        args.softmax_temp,
    )
    _print_markdown(results, args.thresholds)
    args.out.write_text(
        json.dumps(
            {
                "config": vars(args) | {"out": str(args.out)},
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()

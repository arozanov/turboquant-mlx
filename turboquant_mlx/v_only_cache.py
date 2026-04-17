"""V-only TurboQuant KV cache: fp16 keys + 3-bit compressed values.

Key insight from integration testing: quantizing K destroys the attention
pattern (softmax is sensitive to small score perturbations), but V is a
smooth weighted interpolation that tolerates 3-bit compression well.

Keeping K in fp16 preserves generation quality while still compressing V
by ~5x. On a 32K-context 7B model this saves ~40% of total KV memory.

Usage:
    from turboquant_mlx.v_only_cache import VOnlyTurboQuantCache
    cache = [VOnlyTurboQuantCache(bits=3) for _ in range(n_layers)]
    # pass to mlx-lm generate() as prompt_cache
"""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from turboquant_mlx.cache import TurboQuantKVCache


class VOnlyTurboQuantCache:
    """KV cache with fp16 keys and TurboQuant-compressed values.

    Keys are stored and returned in fp16 (via a standard KVCache).
    Values are quantized with PolarQuant on insert and dequantized on
    fetch. The TQ cache's incremental decode buffer (only dequants new
    positions per step) keeps decode speed at parity with baseline.

    Memory layout:
      K:  fp16, stored in KVCache (same as baseline).
      V:  packed uint32 + fp32 norms in TurboQuantKVCache, PLUS an fp16
          incremental dequant buffer for fast per-step fetch.

    The class does NOT expose .bits or .group_size so mlx-lm routes
    through the standard (fp16) SDPA, not the quantized path.
    """

    def __init__(self, bits: int = 3, seed: int = 42):
        self._k_cache = KVCache()
        self._v_tq = TurboQuantKVCache(bits=bits, seed=seed, v_only=True)
        self._v_bits = bits

    def update_and_fetch(self, keys, values):
        """Store K in fp16, compress V, return (K_fp16, V_dequant)."""
        k_out, _ = self._k_cache.update_and_fetch(keys, values)
        _, v_out = self._v_tq.update_and_fetch(keys, values)
        return k_out, v_out

    @property
    def offset(self):
        return self._k_cache.offset

    @property
    def state(self):
        return self._k_cache.state

    def make_mask(self, *args, **kwargs):
        return self._k_cache.make_mask(*args, **kwargs)

    def is_trimmable(self):
        return True

    def trim(self, n):
        self._k_cache.trim(n)
        return self._v_tq.trim(n)

    def size(self):
        return self._k_cache.offset

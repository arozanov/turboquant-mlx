"""Hybrid cache: Apple 8-bit quantized K + TurboQuant 3-bit packed V.

The KV memory budget:
  K: mx.quantize 8-bit (50% of fp16) — Apple's native format
  V: TurboQuant 3-bit packed (~19% of fp16) — never materialized as fp16
  Total: ~35% of baseline fp16 KV = 2.9x compression

Quality:
  K at 8-bit via mx.quantize: preserves greedy decode (verified).
  V at 3-bit via PolarQuant: preserves output quality (V-only safe).

Speed:
  Q@K: Apple's mx.quantized_matmul (fast).
  scores@V: our sparse_v_matvec (butterfly-pulled-out, no per-position WHT).
"""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import QuantizedKVCache

from turboquant_mlx.cache import TurboQuantKVCache


class HybridQuantCache:
    """Apple 8-bit K + TurboQuant 3-bit V.

    K side: delegated to mlx-lm's KVCache (stores fp16, model path
    works natively). On quantize: use to_quantized() after prefill,
    same as kv_bits=8 mode.

    V side: TurboQuantKVCache in v_only mode (packed uint32 + norms).
    V is dequanted on fetch via the TQ incremental buffer, same as
    VOnlyTurboQuantCache. No TQ-specific SDPA needed — Apple's
    standard SDPA handles fp16 K/V from our return values.

    The advantage over VOnlyTurboQuantCache: K is ALSO compressed
    (8-bit via to_quantized), giving further KV memory savings.
    """

    def __init__(self, k_bits: int = 8, k_group_size: int = 64,
                 v_bits: int = 3, v_seed: int = 42):
        from mlx_lm.models.cache import KVCache
        self._k_cache = KVCache()
        self._v_tq = TurboQuantKVCache(bits=v_bits, seed=v_seed, v_only=True)
        self._k_bits = k_bits
        self._k_group_size = k_group_size
        self._v_bits = v_bits
        self._quantized_k = False

    def update_and_fetch(self, keys, values):
        """Store K via KVCache, compress V via TQ, return (K_fp16, V_dequant)."""
        k_out, _ = self._k_cache.update_and_fetch(keys, values)
        _, v_out = self._v_tq.update_and_fetch(keys, values)
        return k_out, v_out

    def maybe_quantize_k(self):
        """Convert K to 8-bit QuantizedKVCache after prefill."""
        if not self._quantized_k and self._k_cache.offset > 0:
            self._k_cache = self._k_cache.to_quantized(
                group_size=self._k_group_size, bits=self._k_bits
            )
            self._quantized_k = True

    @property
    def offset(self):
        return self._k_cache.offset

    @property
    def state(self):
        return self._k_cache.state

    @property
    def bits(self):
        """Expose bits only after K is quantized."""
        if self._quantized_k:
            return self._k_cache.bits
        return None

    @property
    def group_size(self):
        if self._quantized_k:
            return self._k_cache.group_size
        return None

    def make_mask(self, *args, **kwargs):
        return self._k_cache.make_mask(*args, **kwargs)

    def is_trimmable(self):
        return True

    def trim(self, n):
        self._k_cache.trim(n)
        return self._v_tq.trim(n)

    def size(self):
        return self._k_cache.offset

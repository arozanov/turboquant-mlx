"""Layer-adaptive TurboQuant: FP16 for critical layers, compressed for the rest.

First and last N layers use standard KVCache (FP16) — these are most sensitive
to quantization error. Middle layers use TurboQuantKVCache.

This matches the approach in turboquant_plus "layer-adaptive mode 2".
"""

from mlx_lm.models.cache import KVCache
from turboquant_mlx.cache import TurboQuantKVCache


def make_adaptive_cache(
    num_layers: int,
    bits: int = 3,
    k_bits: int = None,
    v_bits: int = None,
    fp16_layers: int = 4,
    seed: int = 42,
    fused: bool = False,
    model=None,
):
    """Create layer-adaptive cache list.

    Args:
        num_layers: total number of transformer layers
        bits: TurboQuant bits for compressed layers (1-4)
        k_bits: bits for K cache (overrides bits if set)
        v_bits: bits for V cache (overrides bits if set)
        fp16_layers: number of first AND last layers to keep in FP16
        seed: random seed for rotation
        fused: use fused attention path for compressed layers
        model: optional model to check for incompatible architectures

    Returns:
        list of cache objects (one per layer)
    """
    if model is not None and hasattr(model, "make_cache"):
        default_cache = model.make_cache()
        if default_cache and not isinstance(default_cache[0], KVCache):
            cache_type = type(default_cache[0]).__name__
            raise ValueError(
                f"[TurboQuant] Incompatible cache type: {cache_type}. "
                f"TurboQuant only works with standard multi-head attention "
                f"(KVCache). MLA, SSM, and hybrid architectures are not supported."
            )

    caches = []
    for i in range(num_layers):
        if i < fp16_layers or i >= num_layers - fp16_layers:
            caches.append(KVCache())
        else:
            caches.append(TurboQuantKVCache(
                bits=bits,
                fused=fused,
            ))
    return caches

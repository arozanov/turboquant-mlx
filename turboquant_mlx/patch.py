"""Monkey-patch mlx_lm to use fused TurboQuant attention during decode.

Call `apply_patch()` before running inference to enable the fused path.
The patch intercepts `scaled_dot_product_attention` in mlx_lm/models/base.py
and routes TurboQuantKVCache decode to our fused Metal kernel.
"""

import mlx.core as mx

_original_sdpa = None
_patched = False


def _patched_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
    """Patched SDPA that uses fused kernel for TurboQuant decode."""
    from turboquant_mlx.cache import TurboQuantKVCache
    from turboquant_mlx.fused_attention import turboquant_attention

    # Use fused path only for decode (single query token) with our cache
    is_decode = queries.shape[2] == 1
    is_tq = isinstance(cache, TurboQuantKVCache) and cache.offset > 0 and cache.fused

    if is_decode and is_tq:
        # Ignore the `keys`/`values` args: when cache.fused is on, update_and_fetch
        # returns empty placeholders (see cache.py lazy-V fast path). Route
        # directly through turboquant_attention, which reads packed K/V from
        # the cache and optionally skips V dequant via sparse_v_matvec.
        return turboquant_attention(
            queries, cache, scale, mask,
            sparse_v_threshold=cache.sparse_v_threshold,
        )
    elif hasattr(cache, "bits"):
        # Original quantized path
        from mlx_lm.models.base import quantized_scaled_dot_product_attention
        return quantized_scaled_dot_product_attention(
            queries, keys, values,
            scale=scale, mask=mask,
            group_size=cache.group_size, bits=cache.bits,
        )
    else:
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values,
            scale=scale, mask=mask,
            sinks=sinks,
        )


def apply_patch():
    """Apply monkey-patch to mlx_lm for fused TurboQuant attention.

    Must patch every model module that imports scaled_dot_product_attention
    from base, because Python's `from X import Y` creates a local binding.
    """
    global _patched, _original_sdpa
    if _patched:
        return

    import mlx_lm.models.base as base_module
    _original_sdpa = base_module.scaled_dot_product_attention

    # Patch base module
    base_module.scaled_dot_product_attention = _patched_sdpa

    # Patch loaded model modules that import SDPA from base
    import sys
    for name, mod in sys.modules.items():
        if name.startswith("mlx_lm.models.") and mod is not None:
            if hasattr(mod, "scaled_dot_product_attention"):
                mod.scaled_dot_product_attention = _patched_sdpa

    _patched = True


def remove_patch():
    """Remove the monkey-patch."""
    global _patched
    if not _patched:
        return

    import mlx_lm.models.base as base_module
    # Restore original — use mx.fast.scaled_dot_product_attention as default
    # The original function is defined in base.py, we need to reimport
    import importlib
    importlib.reload(base_module)
    _patched = False

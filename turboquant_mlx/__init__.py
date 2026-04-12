from turboquant_mlx.cache import TurboQuantKVCache
from turboquant_mlx.metal import fused_quantize, dequant_fp16
from turboquant_mlx.kernels import packed_dequantize, packed_fused_qk_scores
from turboquant_mlx.packing import pack_indices, unpack_indices, packed_dim
from turboquant_mlx.rotation import (
    walsh_hadamard_transform,
    randomized_hadamard_transform,
    inverse_randomized_hadamard,
    random_diagonal_sign,
)
from turboquant_mlx.sparse_v import topk_sparse_v, count_active_positions
from turboquant_mlx.fused_attention import turboquant_attention
from turboquant_mlx.metal_kernels_v4 import (
    prerotate_query,
    prerot_fused_qk_scores,
    prerot_packed_dequantize,
)

__version__ = "0.2.0"

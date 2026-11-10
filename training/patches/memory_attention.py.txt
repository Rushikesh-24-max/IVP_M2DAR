"""Scaled attention with the original equation and bounded-memory CUDA kernels."""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def pooled_scaled_attention(q, k, v, scale):
    # The released model applies no attention-probability dropout or causal mask.
    if not q.is_cuda:
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, scale=scale)
    if torch.is_autocast_enabled():
        dtype = torch.get_autocast_gpu_dtype()
        q, k, v = (tensor.to(dtype=dtype).contiguous() for tensor in (q, k, v))
    else:
        q, k, v = (tensor.contiguous() for tensor in (q, k, v))
    # Fail clearly when no fused kernel is available; never silently recreate
    # the large Q-by-K matrix with the math backend on a constrained GPU.
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, scale=scale)

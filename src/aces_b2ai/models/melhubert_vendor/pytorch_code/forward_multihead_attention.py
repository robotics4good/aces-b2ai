# BSD 3-Clause License — Copyright (c) PyTorch Contributors.
# Vendored unchanged from https://github.com/nervjack2/MelHuBERT

from typing import Callable, List, Optional, Tuple
import math
import warnings

import torch
from torch import _VF

Tensor = torch.Tensor


def dropout(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))
    return _VF.dropout_(input, p, training) if inplace else _VF.dropout(input, p, training)


def softmax(input: Tensor, dim: Optional[int]) -> Tensor:
    assert dim is not None
    return input.softmax(dim)


def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn += attn_mask
    attn = softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)
    output = torch.bmm(attn, v)
    return output, attn


def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    return torch._C._nn.linear(input, weight, bias)


def _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
    skip_embed_dim_check: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    if not skip_embed_dim_check:
        assert w_q.shape == (Eq, Eq)
        assert w_k.shape == (Eq, Ek)
        assert w_v.shape == (Eq, Ev)
        assert b_q is None or b_q.shape == (Eq,)
        assert b_k is None or b_k.shape == (Eq,)
        assert b_v is None or b_v.shape == (Eq,)
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_bias: Optional[Tensor],
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    skip_embed_dim_check: bool = False,
    need_intermediate: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    if not skip_embed_dim_check:
        assert embed_dim == embed_dim_to_check
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim
    else:
        embed_dim = q_proj_weight.shape[0]
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim

    assert key.shape[:2] == value.shape[:2]
    assert q_proj_weight is not None
    assert k_proj_weight is not None
    assert v_proj_weight is not None

    b_q, b_k, b_v = in_proj_bias.chunk(3)
    q, k, v = _in_projection(
        query, key, value,
        q_proj_weight, k_proj_weight, v_proj_weight,
        b_q, b_k, b_v,
        skip_embed_dim_check,
    )

    if attn_mask is not None:
        assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                )
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                )
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    assert key_padding_mask.dtype == torch.bool

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len)
        key_padding_mask = (
            key_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, num_heads, -1, -1)
            .reshape(bsz * num_heads, 1, src_len)
        )
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    if not training:
        dropout_p = 0.0

    attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
    context_layer_val = attn_output
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        out = (attn_output, attn_output_weights)
    else:
        out = (attn_output, None)
    if need_intermediate:
        out = out + (context_layer_val,)
    return out

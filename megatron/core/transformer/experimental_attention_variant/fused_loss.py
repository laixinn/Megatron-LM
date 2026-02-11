# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.distributed as dist

import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": BLOCK_D, "BLOCK_SK": BLOCK_SK, "BLOCK_SQ": BLOCK_SQ}, num_warps=num_warps, num_stages=num_stages)
        for BLOCK_D in [16]
        for BLOCK_SK in [16]
        for BLOCK_SQ in [4]
        for num_warps in [8]
        for num_stages in [3]
    ],
    key=["AH", "Sk", "ASq"],
    cache_results=True,
)
@triton.jit
def _fwd_fused_indexer_loss_kernel(
    Attn_Query_ptr,
    Attn_Key_ptr,
    Loss_ptr,
    Index_Scores_ptr,
    Index_Mask_ptr,
    # Attn query strides: [Sq, B, H, D]
    stride_asq,
    stride_aqb,
    stride_aqh,
    stride_aqd,
    # Attn key strides: [Sk, B, H, D]
    stride_ask,
    stride_akb,
    stride_akh,
    stride_akd,
    # Loss strides: [B, Sq]
    stride_lb,
    stride_ls,
    # Index scores strides: [B, Sq, Sk]
    stride_ibs,
    stride_isq,
    stride_isk,
    # Index mask strides: [B, Sq, Sk]
    stride_imb,
    stride_ims,
    stride_imk,
    # Dimensions
    AH: tl.constexpr,
    AD: tl.constexpr,
    Sk: tl.constexpr,
    ASq: tl.constexpr,
    Sq: tl.constexpr,
    Sq_offset: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_AH: tl.constexpr,
    SPARSE_LOSS: tl.constexpr,
    Softmax_Scale: tl.constexpr,
):
    b = tl.program_id(0)
    sq_block_id = tl.program_id(1)
    
    # should be within (ASq_offset, ASq_offset + ASq)
    aq = sq_block_id * BLOCK_SQ + tl.arange(0, BLOCK_SQ)
    aq_valid = (aq < ASq)

    sq = Sq_offset + aq
    sq_valid = aq_valid & (sq < Sq)
    
    aq_base = Attn_Query_ptr + b * stride_aqb
    ak_base = Attn_Key_ptr + b * stride_akb

    # 1-pass loss recursion
    m_i = tl.full([AH, BLOCK_SQ], float("-inf"), dtype=tl.float32)
    m1_i = tl.full([BLOCK_SQ], float("-inf"), dtype=tl.float32)
    d_i = tl.zeros([AH, BLOCK_SQ], dtype=tl.float32)
    d1_i = tl.zeros([BLOCK_SQ], dtype=tl.float32)
    loss_i = tl.zeros([BLOCK_SQ], dtype=tl.float32)

    h_ids = tl.arange(0, AH)

    # compute the first pass for attn softmax and index softmax
    # apply causal mask by loop trunctation
    causal_sk = tl.minimum(tl.min(sq) + 1, Sk)
    for sk_start in tl.range(0, causal_sk, BLOCK_SK):
        sk_offs = sk_start + tl.arange(0, BLOCK_SK)
        sk_valid = sk_offs < Sk

        index_scores = tl.load(Index_Scores_ptr + b * stride_ibs + sq[:, None] * stride_isq + sk_offs[None, :] * stride_isk, mask=(sq_valid[:, None] & sk_valid[None, :]), other=float("-inf"))
        
        if SPARSE_LOSS:
            index_mask_ptrs = Index_Mask_ptr + b * stride_imb + sq[:, None] * stride_ims + sk_offs[None, :] * stride_imk
            index_mask = tl.load(index_mask_ptrs, mask=(sq_valid[:, None] & sk_valid[None, :]), other=float("-inf"))
            index_scores += index_mask

        # first pass for index softmax
        m1_i_1 = m1_i
        m1_i = tl.maximum(m1_i, tl.max(index_scores, axis=1))
        m1_i = tl.where(m1_i <= float("-inf"), 0.0, m1_i)
        d1_i = d1_i * tl.exp(m1_i_1 - m1_i) + tl.exp(index_scores - m1_i[:, None]).sum(axis=1)

        # first pass for attn softmax
        attn_scores = tl.zeros([AH, BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
        # Vectorize across heads for parallel computation
        for d_start in tl.range(0, AD, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_offs < AD

            # Load all heads at once: [AH, BLOCK_SQ, BLOCK_D]
            aq_ptrs = aq_base + h_ids[:, None, None] * stride_aqh + aq[None, :, None] * stride_asq + d_offs[None, None, :] * stride_aqd
            aq_vals = tl.load(aq_ptrs, mask=(h_ids[:, None, None] < AH) & (aq_valid[None, :, None] & d_valid[None, None, :]), other=0.0)

            # Load all heads at once: [AH, BLOCK_D, BLOCK_SK] (transposed pattern matches original)
            ak_ptrs = ak_base + h_ids[:, None, None] * stride_akh + sk_offs[None, None, :] * stride_ask + d_offs[None, :, None] * stride_akd
            ak_vals = tl.load(ak_ptrs, mask=(h_ids[:, None, None] < AH) & (sk_valid[None, None, :] & d_valid[None, :, None]), other=0.0)

            # Compute batched matrix multiplication: [AH, BLOCK_SQ, BLOCK_D] @ [AH, BLOCK_D, BLOCK_SK] -> [AH, BLOCK_SQ, BLOCK_SK]
            # Using element-wise multiplication and sum over dimension D
            # attn_scores += tl.sum(aq_vals[:, :, :, None] * ak_vals[:, None, :, :], axis=2)
            attn_scores += tl.dot(aq_vals, ak_vals)

        attn_scores *= Softmax_Scale

        # apply causal mask
        casual_mask = tl.full([BLOCK_SQ, BLOCK_SK], float("-inf"), dtype=tl.float32)
        casual_mask = tl.where((sq[:, None] < sk_offs[None, :]), casual_mask, 0.0)
        attn_scores += casual_mask[None, :, :]

        if SPARSE_LOSS:
            attn_scores += index_mask[None, :, :]

        m_i_1 = m_i
        m_i = tl.maximum(m_i, tl.max(attn_scores, axis=-1))
        m_i = tl.where(m_i <= float("-inf"), 0.0, m_i)
        d_i = d_i * tl.exp(m_i_1 - m_i) + tl.exp(attn_scores - m_i[:, :, None]).sum(axis=-1)
    
    # recompute for the second pass of attn softmax
    for sk_start in tl.range(0, causal_sk, BLOCK_SK):
        sk_offs = sk_start + tl.arange(0, BLOCK_SK)
        sk_valid = sk_offs < Sk
        
        index_scores = tl.load(Index_Scores_ptr + b * stride_ibs + sq[:, None] * stride_isq + sk_offs[None, :] * stride_isk, mask=(sq_valid[:, None] & sk_valid[None, :]), other=float("-inf"))

        if SPARSE_LOSS:
            index_mask_ptrs = Index_Mask_ptr + b * stride_imb + sq[:, None] * stride_ims + sk_offs[None, :] * stride_imk
            index_mask = tl.load(index_mask_ptrs, mask=(sq_valid[:, None] & sk_valid[None, :]), other=float("-inf"))
            index_scores += index_mask

        attn_scores = tl.zeros([AH, BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
        # Vectorize across heads for parallel computation
        for d_start in tl.range(0, AD, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_offs < AD

            # Load all heads at once: [AH, BLOCK_SQ, BLOCK_D]
            aq_ptrs = aq_base + h_ids[:, None, None] * stride_aqh + aq[None, :, None] * stride_asq + d_offs[None, None, :] * stride_aqd
            aq_vals = tl.load(aq_ptrs, mask=(h_ids[:, None, None] < AH) & (aq_valid[None, :, None] & d_valid[None, None, :]), other=0.0)

            # Load all heads at once: [AH, BLOCK_D, BLOCK_SK] (transposed pattern matches original)
            ak_ptrs = ak_base + h_ids[:, None, None] * stride_akh + sk_offs[None, None, :] * stride_ask + d_offs[None, :, None] * stride_akd
            ak_vals = tl.load(ak_ptrs, mask=(h_ids[:, None, None] < AH) & (sk_valid[None, None, :] & d_valid[None, :, None]), other=0.0)

            # Compute batched matrix multiplication: [AH, BLOCK_SQ, BLOCK_D] @ [AH, BLOCK_D, BLOCK_SK] -> [AH, BLOCK_SQ, BLOCK_SK]
            # Using element-wise multiplication and sum over dimension D
            # attn_scores += tl.sum(aq_vals[:, :, :, None] * ak_vals[:, None, :, :], axis=2)
            attn_scores += tl.dot(aq_vals, ak_vals)

        attn_scores *= Softmax_Scale

        # apply causal mask
        casual_mask = tl.full([BLOCK_SQ, BLOCK_SK], float("-inf"), dtype=tl.float32)
        casual_mask = tl.where((sq[:, None] < sk_offs[None, :]), casual_mask, 0.0)
        attn_scores += casual_mask[None, :, :]

        if SPARSE_LOSS:
            attn_scores += index_mask[None, :, :]

        # softmax
        softmax_attn_i = tl.exp(attn_scores - m_i[:, :, None]) / d_i[:, :, None]
        softmax_index_i = tl.exp(index_scores - m1_i[:, None]) / d1_i[:, None]

        # reduce head dim
        softmax_attn_i = tl.sum(softmax_attn_i, axis=0) / AH

        # loss
        loss_sk = softmax_attn_i * (tl.log(softmax_attn_i + 1e-10) - tl.log(softmax_index_i + 1e-10))
        loss_i += loss_sk.sum(axis=-1)

    # Store loss
    tl.store(Loss_ptr + b * stride_lb + aq * stride_ls, loss_i, mask=aq_valid)


def fwd_fused_indexer_loss(
    index_scores: torch.Tensor,
    attn_query: torch.Tensor,
    attn_key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: Optional[bool] = False,
    index_mask: Optional[torch.Tensor] = None,
    Sq_offset: Optional[int] = 0,
    full_Sq: Optional[int] = 0,
) -> torch.Tensor:
    ASq, AB, AH, AD = attn_query.shape
    ASk = attn_key.shape[0]

    # This kernel performs badly when AH is larger than 16.
    # if AH <= 16:
    #     BLOCK_SK = 64
    # else:
    #     BLOCK_SK = 16
    BLOCK_SK = 16
    BLOCK_SQ = 8
    BLOCK_D  = 16
    BLOCK_AH = 8

    out_loss = torch.empty((AB, ASq), dtype=torch.float32, device=attn_query.device)
    attn_num_sq_blocks = (ASq + BLOCK_SQ - 1) // BLOCK_SQ
    # attn_grid = (AB, attn_num_sq_blocks,)
    attn_grid = (AB, ASq, )

    if sparse_loss:
        stride_imb = index_mask.stride(0)
        stride_ims = index_mask.stride(1)
        stride_imk = index_mask.stride(2)
    else:
        index_mask = torch.empty((0,), dtype=torch.float32, device=attn_query.device)
        stride_imb = stride_ims = stride_imk = 0
    
    _fwd_fused_indexer_loss_kernel[attn_grid](
        Attn_Query_ptr=attn_query,
        Attn_Key_ptr=attn_key,
        Loss_ptr=out_loss,
        Index_Scores_ptr=index_scores,
        Index_Mask_ptr=index_mask,
        # Attn query strides: [Sq, B, H, D]
        stride_asq=attn_query.stride(0),
        stride_aqb=attn_query.stride(1),
        stride_aqh=attn_query.stride(2),
        stride_aqd=attn_query.stride(3),
        # Attn key strides: [Sk, B, H, D]
        stride_ask=attn_key.stride(0),
        stride_akb=attn_key.stride(1),
        stride_akh=attn_key.stride(2),
        stride_akd=attn_key.stride(3),
        # Loss strides: [B, Sq]
        stride_lb=out_loss.stride(0),
        stride_ls=out_loss.stride(1),
        # Index scores strides: [B, Sq, Sk]
        stride_ibs=index_scores.stride(0),
        stride_isq=index_scores.stride(1),
        stride_isk=index_scores.stride(2),
        # Index mask strides: [B, Sq, Sk]
        stride_imb=stride_imb,
        stride_ims=stride_ims,
        stride_imk=stride_imk,
        # Dimensions
        AH=AH,
        AD=AD,
        Sk=ASk,
        ASq=ASq,
        Sq=full_Sq,
        Sq_offset=Sq_offset,
        # BLOCK_SQ=BLOCK_SQ,
        # BLOCK_SK=BLOCK_SK,
        # BLOCK_D=BLOCK_D,
        BLOCK_AH=BLOCK_AH,
        SPARSE_LOSS=sparse_loss,
        Softmax_Scale=softmax_scale,
    )

    indexer_loss = out_loss.mean() * loss_coeff
    
    return indexer_loss
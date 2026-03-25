# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from typing import Optional, Tuple

import torch

import triton
import triton.language as tl

# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in [4, 8, 16]
#         for num_stages in [1, 2, 3, 4]
#     ],
#     key=["AH"],
#     cache_results=True,
# )
@triton.jit
def _fwd_fused_indexer_loss_stage1_kernel(
    Attn_Query_ptr,
    Attn_Key_ptr,
    Loss_ptr,
    Index_Scores_ptr,
    Index_Mask_ptr,
    m_ptr,
    d_ptr,
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
    # softmax m strides: [B, H, Sq]
    stride_smmb,
    stride_smmh,
    stride_smmq,
    # softmax d strides: [B, H, Sq]
    stride_smdb,
    stride_smdh,
    stride_smdq,
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
    SPARSE_LOSS: tl.constexpr,
    Softmax_Scale: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    sq_block_id = tl.program_id(1).to(tl.int64)
    h = tl.program_id(2)
    
    # should be within (ASq_offset, ASq_offset + ASq)
    aq = sq_block_id * BLOCK_SQ + tl.arange(0, BLOCK_SQ)
    aq_valid = (aq < ASq)

    sq = Sq_offset + aq
    sq_valid = aq_valid & (sq < Sq)
    
    aq_base = Attn_Query_ptr + b * stride_aqb
    ak_base = Attn_Key_ptr + b * stride_akb

    # 1-pass loss recursion
    m1_i = tl.full([BLOCK_SQ], float("-inf"), dtype=tl.float32)
    d1_i = tl.zeros([BLOCK_SQ], dtype=tl.float32)
    loss_i = tl.zeros([BLOCK_SQ], dtype=tl.float32)

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

        casual_mask = tl.full([BLOCK_SQ, BLOCK_SK], float("-inf"), dtype=tl.float32)
        casual_mask = tl.where((sq[:, None] < sk_offs[None, :]), casual_mask, 0.0)

        h_scores = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
        for d_start in tl.range(0, AD, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_valid = d_offs < AD

            # Load all heads at once: [BLOCK_SQ, BLOCK_D]
            aq_ptrs = aq_base + h * stride_aqh + aq[:, None] * stride_asq + d_offs[None, :] * stride_aqd
            aq_vals = tl.load(aq_ptrs, mask=(aq_valid[:, None] & d_valid[None, :]), other=0.0)

            # Load all heads at once: [BLOCK_D, BLOCK_SK]
            ak_ptrs = ak_base + h * stride_akh + sk_offs[None, :] * stride_ask + d_offs[:, None] * stride_akd
            ak_vals = tl.load(ak_ptrs, mask=(sk_valid[None, :] & d_valid[:, None]), other=0.0)

            # Compute batched matrix multiplication: [BLOCK_SQ, BLOCK_D] @ [BLOCK_D, BLOCK_SK] -> [BLOCK_SQ, BLOCK_SK]
            h_scores = tl.dot(aq_vals, ak_vals, acc=h_scores, allow_tf32=False)

        h_scores *= Softmax_Scale

        # apply causal mask
        h_scores += casual_mask

        if SPARSE_LOSS:
            h_scores += index_mask

        m_i = tl.load(m_ptr + b * stride_smmb + h * stride_smmh + sq * stride_smmq, mask=sq_valid, other=float("-inf"))
        d_i = tl.load(d_ptr + b * stride_smdb + h * stride_smdh + sq * stride_smdq, mask=sq_valid, other=0.0)
        m_i_1 = m_i
        m_i = tl.maximum(m_i, tl.max(h_scores, axis=-1))
        m_i = tl.where(m_i <= float("-inf"), 0.0, m_i)
        d_i = d_i * tl.exp(m_i_1 - m_i) + tl.exp(h_scores - m_i[:, None]).sum(axis=-1)
        tl.store(m_ptr + b * stride_smmb + h * stride_smmh + sq * stride_smmq, m_i, mask=sq_valid)
        tl.store(d_ptr + b * stride_smdb + h * stride_smdh + sq * stride_smdq, d_i, mask=sq_valid)

@triton.jit
def _fwd_fused_indexer_loss_stage2_kernel(
    Attn_Query_ptr,
    Attn_Key_ptr,
    Loss_ptr,
    Index_Scores_ptr,
    Index_Mask_ptr,
    m_ptr,
    d_ptr,
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
    # softmax m strides: [B, H, Sq]
    stride_smmb,
    stride_smmh,
    stride_smmq,
    # softmax d strides: [B, H, Sq]
    stride_smdb,
    stride_smdh,
    stride_smdq,
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
    SPARSE_LOSS: tl.constexpr,
    Softmax_Scale: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    sq_block_id = tl.program_id(1).to(tl.int64)
    
    # should be within (ASq_offset, ASq_offset + ASq)
    aq = sq_block_id * BLOCK_SQ + tl.arange(0, BLOCK_SQ)
    aq_valid = (aq < ASq)

    sq = Sq_offset + aq
    sq_valid = aq_valid & (sq < Sq)
    
    aq_base = Attn_Query_ptr + b * stride_aqb
    ak_base = Attn_Key_ptr + b * stride_akb

    # 1-pass loss recursion
    m1_i = tl.full([BLOCK_SQ], float("-inf"), dtype=tl.float32)
    d1_i = tl.zeros([BLOCK_SQ], dtype=tl.float32)
    loss_i = tl.zeros([BLOCK_SQ], dtype=tl.float32)

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

    # recompute for the second pass of attn softmax
    for sk_start in tl.range(0, causal_sk, BLOCK_SK):
        sk_offs = sk_start + tl.arange(0, BLOCK_SK)
        sk_valid = sk_offs < Sk
        
        index_scores = tl.load(Index_Scores_ptr + b * stride_ibs + sq[:, None] * stride_isq + sk_offs[None, :] * stride_isk, mask=(sq_valid[:, None] & sk_valid[None, :]), other=float("-inf"))

        if SPARSE_LOSS:
            index_mask_ptrs = Index_Mask_ptr + b * stride_imb + sq[:, None] * stride_ims + sk_offs[None, :] * stride_imk
            index_mask = tl.load(index_mask_ptrs, mask=(sq_valid[:, None] & sk_valid[None, :]), other=float("-inf"))
            index_scores += index_mask

        softmax_attn_i = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
        casual_mask = tl.full([BLOCK_SQ, BLOCK_SK], float("-inf"), dtype=tl.float32)
        casual_mask = tl.where((sq[:, None] < sk_offs[None, :]), casual_mask, 0.0)

        for h in tl.range(0, AH):
            h_scores = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            for d_start in tl.range(0, AD, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < AD

                # Load all heads at once: [BLOCK_SQ, BLOCK_D]
                aq_ptrs = aq_base + h * stride_aqh + aq[:, None] * stride_asq + d_offs[None, :] * stride_aqd
                aq_vals = tl.load(aq_ptrs, mask=(aq_valid[:, None] & d_valid[None, :]), other=0.0)

                # Load all heads at once: [BLOCK_D, BLOCK_SK]
                ak_ptrs = ak_base + h * stride_akh + sk_offs[None, :] * stride_ask + d_offs[:, None] * stride_akd
                ak_vals = tl.load(ak_ptrs, mask=(sk_valid[None, :] & d_valid[:, None]), other=0.0)

                # Compute batched matrix multiplication: [BLOCK_SQ, BLOCK_D] @ [BLOCK_D, BLOCK_SK] -> [BLOCK_SQ, BLOCK_SK]
                h_scores = tl.dot(aq_vals, ak_vals, acc=h_scores, allow_tf32=False)

            h_scores *= Softmax_Scale

            # apply causal mask
            h_scores += casual_mask

            if SPARSE_LOSS:
                h_scores += index_mask

            # softmax
            m_i = tl.load(m_ptr + b * stride_smmb + h * stride_smmh + sq * stride_smmq, mask=sq_valid, other=float("-inf"))
            d_i = tl.load(d_ptr + b * stride_smdb + h * stride_smdh + sq * stride_smdq, mask=sq_valid, other=0.0)
            softmax_attn_i += tl.exp(h_scores - m_i[:, None]) / d_i[:, None]

        softmax_attn_i /= AH
        softmax_index_i = tl.exp(index_scores - m1_i[:, None]) / d1_i[:, None]

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
    assert AH <= 128, "This kernel might be broken for AH > 128."

    BLOCK_SK = 128
    BLOCK_SQ = 128
    BLOCK_D  = 64
    num_warps = 8
    num_stages = 3

    out_loss = torch.empty((AB, ASq), dtype=torch.float32, device=attn_query.device)
    attn_num_sq_blocks = (ASq + BLOCK_SQ - 1) // BLOCK_SQ
    attn_grid = (AB, attn_num_sq_blocks)    

    if sparse_loss:
        stride_imb = index_mask.stride(0)
        stride_ims = index_mask.stride(1)
        stride_imk = index_mask.stride(2)
    else:
        index_mask = torch.empty((0,), dtype=torch.float32, device=attn_query.device)
        stride_imb = stride_ims = stride_imk = 0

    softmax_m = torch.full((AB, AH, ASq), float("-inf"), dtype=torch.float32, device=attn_query.device)
    softmax_d = torch.full((AB, AH, ASq), 0.0, dtype=torch.float32, device=attn_query.device)

    stage1_grid = (AB, attn_num_sq_blocks, AH)
    _fwd_fused_indexer_loss_stage1_kernel[stage1_grid](
        Attn_Query_ptr=attn_query,
        Attn_Key_ptr=attn_key,
        Loss_ptr=out_loss,
        Index_Scores_ptr=index_scores,
        Index_Mask_ptr=index_mask,
        m_ptr=softmax_m,
        d_ptr=softmax_d,
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
        # softmax m strides: [B, H, Sq]
        stride_smmb=softmax_m.stride(0),
        stride_smmh=softmax_m.stride(1),
        stride_smmq=softmax_m.stride(2),
        # softmax d strides: [B, H, Sq]
        stride_smdb=softmax_d.stride(0),
        stride_smdh=softmax_d.stride(1),
        stride_smdq=softmax_d.stride(2),
        # Dimensions
        AH=AH,
        AD=AD,
        Sk=ASk,
        ASq=ASq,
        Sq=full_Sq,
        Sq_offset=Sq_offset,
        SPARSE_LOSS=sparse_loss,
        Softmax_Scale=softmax_scale,
        BLOCK_SQ=BLOCK_SQ,
        BLOCK_SK=BLOCK_SK,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    stage2_grid = (AB, attn_num_sq_blocks)
    _fwd_fused_indexer_loss_stage2_kernel[stage2_grid](
        Attn_Query_ptr=attn_query,
        Attn_Key_ptr=attn_key,
        Loss_ptr=out_loss,
        Index_Scores_ptr=index_scores,
        Index_Mask_ptr=index_mask,
        m_ptr=softmax_m,
        d_ptr=softmax_d,
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
        # softmax m strides: [B, H, Sq]
        stride_smmb=softmax_m.stride(0),
        stride_smmh=softmax_m.stride(1),
        stride_smmq=softmax_m.stride(2),
        # softmax d strides: [B, H, Sq]
        stride_smdb=softmax_d.stride(0),
        stride_smdh=softmax_d.stride(1),
        stride_smdq=softmax_d.stride(2),
        # Dimensions
        AH=AH,
        AD=AD,
        Sk=ASk,
        ASq=ASq,
        Sq=full_Sq,
        Sq_offset=Sq_offset,
        SPARSE_LOSS=sparse_loss,
        Softmax_Scale=softmax_scale,
        BLOCK_SQ=BLOCK_SQ,
        BLOCK_SK=BLOCK_SK,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    indexer_loss = out_loss.mean() * loss_coeff
    
    return indexer_loss, out_loss
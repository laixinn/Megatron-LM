# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.distributed as dist

import triton
import triton.language as tl

from megatron.core.process_groups_config import ProcessGroupCollection

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
    SPARSE_LOSS: tl.constexpr,
    Softmax_Scale: tl.constexpr,
):
    b = tl.program_id(0)
    sq_block_id = tl.program_id(1)
    
    # should be within (ASq_offset, ASq_offset + ASq)
    aq = sq_block_id * BLOCK_SQ + tl.arange(0, BLOCK_SQ)
    aq_valid = (aq < ASq)

    is_base = Sq_offset + aq
    aq_valid = aq_valid & (is_base < Sq)
    
    aq_base = Attn_Query_ptr + b * stride_aqb
    ak_base = Attn_Key_ptr + b * stride_akb

    # 1-pass loss recursion
    m_i = tl.full([AH, BLOCK_SQ], float("-inf"), dtype=tl.float32)
    m1_i = tl.full([BLOCK_SQ], float("-inf"), dtype=tl.float32)
    d_i = tl.zeros([AH, BLOCK_SQ], dtype=tl.float32)
    d1_i = tl.zeros([BLOCK_SQ], dtype=tl.float32)
    loss_i = tl.zeros([BLOCK_SQ], dtype=tl.float32)

    # compute the first pass for attn softmax and index softmax
    # apply causal mask by loop trunctation
    causal_sk = tl.minimum(tl.min(is_base) + 1, Sk)
    for sk_start in tl.range(0, causal_sk, BLOCK_SK):
        sk_offs = sk_start + tl.arange(0, BLOCK_SK)
        sk_valid = sk_offs < Sk

        index_scores = tl.load(Index_Scores_ptr + b * stride_ibs + is_base[:, None] * stride_isq + sk_offs[None, :] * stride_isk, mask=(sq_valid[:, None] & sk_valid[None, :]), other=float("-inf"))
        
        if SPARSE_LOSS:
            index_mask_ptrs = Index_Mask_ptr + b * stride_imb + is_base[:, None] * stride_ims + sk_offs[None, :] * stride_imk
            index_mask = tl.load(index_mask_ptrs, mask=(sq_valid[:, None] & sk_valid[None, :]), other=float("-inf"))
            index_scores += index_mask

        # first pass for index softmax
        m1_i_1 = m1_i
        m1_i = tl.maximum(m1_i, tl.max(index_scores, axis=1))
        m1_i = tl.where(m1_i <= float("-inf"), 0.0, m1_i)
        d1_i = d1_i * tl.exp(m1_i_1 - m1_i) + tl.exp(index_scores - m1_i[:, None]).sum(axis=1)

        # first pass for attn softmax
        h_ids = tl.arange(0, AH)

        attn_scores = tl.zeros([AH, BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
        for h in tl.range(AH):
            aq_head_base = aq_base + h * stride_aqh
            ak_head_base = ak_base + h * stride_akh

            dot = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            for d_start in tl.range(0, AD, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < AD

                aq_ptrs = aq_head_base + aq[:, None] * stride_asq + d_offs[None, :] * stride_aqd
                ak_ptrs = ak_head_base + sk_offs[None, :] * stride_ask + d_offs[:, None] * stride_akd

                aq_vals = tl.load(aq_ptrs, mask=(aq_valid[:, None] & d_valid[None, :]), other=0.0)
                ak_vals = tl.load(ak_ptrs, mask=(sk_valid[None, :] & d_valid[:, None]), other=0.0)

                dot += tl.dot(aq_vals, ak_vals)

            dot *= Softmax_Scale

            attn_scores = tl.where(h_ids[:, None, None] == h, dot[None, :, :], attn_scores)

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
        for h in tl.range(AH):
            aq_head_base = aq_base + h * stride_aqh
            ak_head_base = ak_base + h * stride_akh

            dot = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            for d_start in tl.range(0, AD, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < AD

                aq_ptrs = aq_head_base + aq[:, None] * stride_asq + d_offs[None, :] * stride_aqd
                ak_ptrs = ak_head_base + sk_offs[None, :] * stride_ask + d_offs[:, None] * stride_akd

                aq_vals = tl.load(aq_ptrs, mask=(aq_valid[:, None] & d_valid[None, :]), other=0.0)
                ak_vals = tl.load(ak_ptrs, mask=(sk_valid[None, :] & d_valid[:, None]), other=0.0)

                dot += tl.dot(aq_vals, ak_vals)

            dot *= Softmax_Scale

            attn_scores = tl.where(h_ids[:, None, None] == h, dot[None, :, :], attn_scores)

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
    index_mask: torch.Tensor,
    attn_query: torch.Tensor,
    attn_key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: Optional[bool] = False,
    Sq_offset: Optional[int] = 0,
    full_Sq: Optional[int] = 0,
) -> torch.Tensor:
    ASq, AB, AH, AD = attn_query.shape
    ASk = attn_key.shape[0]

    BLOCK_SK = 64
    BLOCK_SQ = 16
    BLOCK_D  = 64

    out_loss = torch.empty((AB, ASq), dtype=torch.float32, device=attn_query.device)
    attn_num_sq_blocks = (ASq + BLOCK_SQ - 1) // BLOCK_SQ
    attn_grid = (AB, attn_num_sq_blocks,)
    
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
        stride_imb=index_mask.stride(0),
        stride_ims=index_mask.stride(1),
        stride_imk=index_mask.stride(2),
        # Dimensions
        AH=AH,
        AD=AD,
        Sk=ASk,
        ASq=ASq,
        Sq=full_Sq,
        Sq_offset=Sq_offset,
        BLOCK_SQ=BLOCK_SQ,
        BLOCK_SK=BLOCK_SK,
        BLOCK_D=BLOCK_D,
        SPARSE_LOSS=sparse_loss,
        Softmax_Scale=softmax_scale,
    )

    indexer_loss = out_loss.mean() * loss_coeff
    
    return indexer_loss


@triton.jit
def _bwd_fused_indexer_loss_kernel(
    Q_ptr,
    K_ptr,
    W_ptr,
    Attn_Query_ptr,
    Attn_Key_ptr,
    Topk_Idx_ptr,
    Grad_Q_ptr,
    Grad_W_ptr,
    Grad_K_ptr,
    # Q strides: [Sq, B, H, D]
    stride_qs,
    stride_qb,
    stride_qh,
    stride_qd,
    # K strides: [Sk, B, D]
    stride_ks,
    stride_kb,
    stride_kd,
    # W strides: [Sq, B, H]
    stride_ws,
    stride_wb,
    stride_wh,
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
    # Topk indices strides: [B, Sq, TopK]
    stride_tb,
    stride_ts,
    stride_tk,
    # Grad Q strides: [Sq, B, H, D]
    stride_gqs,
    stride_gqb,
    stride_gqh,
    stride_gqd,
    # Grad W strides: [Sq, B, H]
    stride_gws,
    stride_gwb,
    stride_gwh,
    # Grad K strides: [B, Sk, D]
    stride_pgb,
    stride_pgk,
    stride_pgd,
    # Dimensions
    H: tl.constexpr,
    D: tl.constexpr,
    AH: tl.constexpr,
    AD: tl.constexpr,
    Sq: tl.constexpr,
    Sk: tl.constexpr,
    TopK: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    Softmax_Scale: tl.constexpr,
    Grad_Loss_Scale: tl.constexpr,
    SPARSE_LOSS: tl.constexpr,
    ASq: tl.constexpr,
    Sq_offset: tl.constexpr,
):
    """
    Compute gradient of KL loss w.r.t. index_scores logits (before softmax).
    This is the first step of the backward pass - compute grad_index_logits.
    """
    b = tl.program_id(0)
    sq_block_id = tl.program_id(1)
    
    aq = sq_block_id * BLOCK_SQ + tl.arange(0, BLOCK_SQ)
    aq_valid = (aq < ASq)
    sq = Sq_offset + sq_block_id * BLOCK_SQ + tl.arange(0, BLOCK_SQ)
    sq_valid = (sq < Sq) & (sq < Sq_offset + ASq)
    
    # Base pointers
    q_base = Q_ptr + b * stride_qb
    k_base = K_ptr + b * stride_kb
    w_base = W_ptr + b * stride_wb
    aq_base = Attn_Query_ptr + b * stride_aqb
    ak_base = Attn_Key_ptr + b * stride_akb
    
    # First pass: compute softmax denominators  
    m_i = tl.full([AH, BLOCK_SQ], float("-inf"), dtype=tl.float32)
    m1_i = tl.full([BLOCK_SQ], float("-inf"), dtype=tl.float32)
    d_i = tl.zeros([AH, BLOCK_SQ], dtype=tl.float32)
    d1_i = tl.zeros([BLOCK_SQ], dtype=tl.float32)

    sum_grad = tl.zeros([BLOCK_SQ, 1], dtype=tl.float32)
    
    causal_sk = tl.minimum(tl.max(sq) + 1, Sk)

    # First pass for softmax statistics
    for sk_start in tl.range(0, causal_sk, BLOCK_SK):
        sk_offs = sk_start + tl.arange(0, BLOCK_SK)
        sk_valid = sk_offs < Sk

        # Compute index_scores
        index_scores = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
        for h in tl.range(H):
            w_val = tl.load(w_base + sq * stride_ws + h * stride_wh, mask=sq_valid, other=0.0)
            q_head_base = q_base + h * stride_qh
            
            dot = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            for d_start in tl.range(0, D, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < D
                
                q_ptrs = q_head_base + sq[:, None] * stride_qs + d_offs[None, :] * stride_qd
                q_vals = tl.load(q_ptrs, mask=(sq_valid[:, None] & d_valid[None, :]), other=0.0)
                
                k_ptrs = k_base + sk_offs[None, :] * stride_ks + d_offs[:, None] * stride_kd
                k_vals = tl.load(k_ptrs, mask=(sk_valid[None, :] & d_valid[:, None]), other=0.0)
                
                dot += tl.dot(q_vals, k_vals)
            
            dot = tl.maximum(dot, 0.0)
            index_scores += dot * w_val[:, None]
        
        causal_mask = tl.where((sq[:, None] >= sk_offs[None, :]), 0.0, float("-inf"))
        index_scores = index_scores + causal_mask
        
        # Apply sparse loss mask if enabled
        if SPARSE_LOSS:
            sparse_mask = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            sq_strides = tl.arange(0, BLOCK_SQ)
            sk_strides = tl.arange(0, BLOCK_SK)
            for sq_i in tl.range(BLOCK_SQ):
                sq_i_offs = Sq_offset + sq_block_id * BLOCK_SQ + sq_i
                sq_i_valid = (sq_i_offs < Sq) & (sq_i_offs < Sq_offset + ASq)

                for topk_i in tl.range(tl.cdiv(TopK, BLOCK_TOPK)):
                    topk_off = topk_i * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
                    topk_valid = topk_off < TopK
                    
                    topk_indices = tl.load(
                        Topk_Idx_ptr + b * stride_tb + sq_i_offs * stride_ts + topk_off * stride_tk,
                        mask=sq_i_valid & topk_valid,
                        other=-1
                    )
                    
                    for sk_j in tl.range(BLOCK_SK):
                        if sk_start + sk_j < Sk:
                            sk_j_offs = sk_start + sk_j

                            ij_mask = tl.sum(topk_indices == sk_j_offs) > 0
                            
                            sparse_mask = tl.where((sq_strides[:, None] == sq_i) & (sk_strides[None, :] == sk_j), ij_mask + sparse_mask, sparse_mask)
            
            sparse_mask = tl.where(sparse_mask > 0, 0.0, float("-inf"))
            index_scores = index_scores + sparse_mask
        
        m1_i_1 = m1_i
        m1_i = tl.maximum(m1_i, tl.max(index_scores, axis=1))
        d1_i = d1_i * tl.exp(m1_i_1 - m1_i) + tl.sum(tl.exp(index_scores - m1_i[:, None]), axis=1)
        
        # Compute attention scores
        attn_scores = tl.zeros([AH, BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
        for h in tl.range(AH):
            aq_head_base = aq_base + h * stride_aqh
            ak_head_base = ak_base + h * stride_akh
            
            dot = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            for d_start in tl.range(0, AD, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < AD
                
                aq_ptrs = aq_head_base + aq[:, None] * stride_asq + d_offs[None, :] * stride_aqd
                aq_vals = tl.load(aq_ptrs, mask=(aq_valid[:, None] & d_valid[None, :]), other=0.0)
                
                ak_ptrs = ak_head_base + sk_offs[None, :] * stride_ask + d_offs[:, None] * stride_akd
                ak_vals = tl.load(ak_ptrs, mask=(sk_valid[None, :] & d_valid[:, None]), other=0.0)
                
                dot += tl.dot(aq_vals, ak_vals)
            
            dot = dot * Softmax_Scale + causal_mask
            if SPARSE_LOSS:
                dot = dot + sparse_mask
            h_idx = tl.arange(0, AH)
            attn_scores = tl.where(h_idx[:, None, None] == h, dot[None, :, :], attn_scores)
        
        m_i_1 = m_i
        m_i = tl.maximum(m_i, tl.max(attn_scores, axis=-1))
        d_i = d_i * tl.exp(m_i_1 - m_i) + tl.sum(tl.exp(attn_scores - m_i[:, :, None]), axis=-1)
    
    # Second pass: compute gradient w.r.t. index_logits
    for sk_start in tl.range(0, causal_sk, BLOCK_SK):
        sk_offs = sk_start + tl.arange(0, BLOCK_SK)
        sk_valid = sk_offs < Sk
        
        # Recompute index_scores
        index_scores = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
        for h in tl.range(H):
            w_val = tl.load(w_base + sq * stride_ws + h * stride_wh, mask=sq_valid, other=0.0)
            q_head_base = q_base + h * stride_qh
            
            dot = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            for d_start in tl.range(0, D, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < D
                
                q_ptrs = q_head_base + sq[:, None] * stride_qs + d_offs[None, :] * stride_qd
                q_vals = tl.load(q_ptrs, mask=(sq_valid[:, None] & d_valid[None, :]), other=0.0)
                
                k_ptrs = k_base + sk_offs[None, :] * stride_ks + d_offs[:, None] * stride_kd
                k_vals = tl.load(k_ptrs, mask=(sk_valid[None, :] & d_valid[:, None]), other=0.0)
                
                dot += tl.dot(q_vals, k_vals)
            
            dot = tl.maximum(dot, 0.0)
            index_scores += dot * w_val[:, None]
        
        causal_mask = tl.where((sq[:, None] >= sk_offs[None, :]), 0.0, float("-inf"))
        index_scores = index_scores + causal_mask
        
        # Apply sparse loss mask if enabled
        if SPARSE_LOSS:
            sparse_mask = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            sq_strides = tl.arange(0, BLOCK_SQ)
            sk_strides = tl.arange(0, BLOCK_SK)
            for sq_i in tl.range(BLOCK_SQ):
                sq_i_offs = Sq_offset + sq_block_id * BLOCK_SQ + sq_i
                sq_i_valid = (sq_i_offs < Sq) & (sq_i_offs < Sq_offset + ASq)

                for topk_i in tl.range(tl.cdiv(TopK, BLOCK_TOPK)):
                    topk_off = topk_i * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
                    topk_valid = topk_off < TopK
                    
                    topk_indices = tl.load(
                        Topk_Idx_ptr + b * stride_tb + sq_i_offs * stride_ts + topk_off * stride_tk,
                        mask=sq_i_valid & topk_valid,
                        other=-1
                    )
                    
                    for sk_j in tl.range(BLOCK_SK):
                        if sk_start + sk_j < Sk:
                            sk_j_offs = sk_start + sk_j

                            ij_mask = tl.sum(topk_indices == sk_j_offs) > 0
                            
                            sparse_mask = tl.where((sq_strides[:, None] == sq_i) & (sk_strides[None, :] == sk_j), ij_mask + sparse_mask, sparse_mask)
            
            sparse_mask = tl.where(sparse_mask > 0, 0.0, float("-inf"))
            index_scores = index_scores + sparse_mask
        
        # Recompute attention scores
        attn_scores = tl.zeros([AH, BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
        for h in tl.range(AH):
            aq_head_base = aq_base + h * stride_aqh
            ak_head_base = ak_base + h * stride_akh
            
            dot = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            for d_start in tl.range(0, AD, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < AD
                
                aq_ptrs = aq_head_base + aq[:, None] * stride_asq + d_offs[None, :] * stride_aqd
                aq_vals = tl.load(aq_ptrs, mask=(aq_valid[:, None] & d_valid[None, :]), other=0.0)
                
                ak_ptrs = ak_head_base + sk_offs[None, :] * stride_ask + d_offs[:, None] * stride_akd
                ak_vals = tl.load(ak_ptrs, mask=(sk_valid[None, :] & d_valid[:, None]), other=0.0)
                
                dot += tl.dot(aq_vals, ak_vals)
            
            dot = dot * Softmax_Scale + causal_mask
            if SPARSE_LOSS:
                dot = dot + sparse_mask
            h_idx = tl.arange(0, AH)
            attn_scores = tl.where(h_idx[:, None, None] == h, dot[None, :, :], attn_scores)
        
        # Compute softmax values
        index_scores_softmax = tl.exp(index_scores - m1_i[:, None]) / d1_i[:, None]
        attn_scores_softmax = tl.exp(attn_scores - m_i[:, :, None]) / d_i[:, :, None]
        
        # Sum and normalize attention scores
        attn_scores_sum = tl.sum(attn_scores_softmax, axis=0) / AH
        
        # Gradient of KL divergence w.r.t. index_scores_softmax
        grad_index_softmax = -attn_scores_sum / (index_scores_softmax + 1e-10) * Grad_Loss_Scale
        
        # Backward through softmax
        sum_grad += tl.sum(grad_index_softmax * index_scores_softmax, axis=-1, keep_dims=True)

    # Third pass
    for sk_start in tl.range(0, causal_sk, BLOCK_SK):
        sk_offs = sk_start + tl.arange(0, BLOCK_SK)
        sk_valid = sk_offs < Sk

        # Recompute index_scores
        index_scores = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
        for h in tl.range(H):
            w_val = tl.load(w_base + sq * stride_ws + h * stride_wh, mask=sq_valid, other=0.0)
            q_head_base = q_base + h * stride_qh
            
            dot = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            for d_start in tl.range(0, D, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < D
                
                q_ptrs = q_head_base + sq[:, None] * stride_qs + d_offs[None, :] * stride_qd
                q_vals = tl.load(q_ptrs, mask=(sq_valid[:, None] & d_valid[None, :]), other=0.0)
                
                k_ptrs = k_base + sk_offs[None, :] * stride_ks + d_offs[:, None] * stride_kd
                k_vals = tl.load(k_ptrs, mask=(sk_valid[None, :] & d_valid[:, None]), other=0.0)
                
                dot += tl.dot(q_vals, k_vals)
            
            dot = tl.maximum(dot, 0.0)
            index_scores += dot * w_val[:, None]
        
        causal_mask = tl.where((sq[:, None] >= sk_offs[None, :]), 0.0, float("-inf"))
        index_scores = index_scores + causal_mask
        
        # Apply sparse loss mask if enabled
        if SPARSE_LOSS:
            sparse_mask = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            sq_strides = tl.arange(0, BLOCK_SQ)
            sk_strides = tl.arange(0, BLOCK_SK)
            # leave all threads to topk parallelism
            for sq_i in tl.range(BLOCK_SQ):
                sq_i_offs = Sq_offset + sq_block_id * BLOCK_SQ + sq_i
                sq_i_valid = (sq_i_offs < Sq) & (sq_i_offs < Sq_offset + ASq)

                for topk_i in tl.range(tl.cdiv(TopK, BLOCK_TOPK)):
                    topk_off = topk_i * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
                    topk_valid = topk_off < TopK
                    
                    topk_indices = tl.load(
                        Topk_Idx_ptr + b * stride_tb + sq_i_offs * stride_ts + topk_off * stride_tk,
                        mask=sq_i_valid & topk_valid,
                        other=-1
                    )
                    
                    for sk_j in tl.range(BLOCK_SK):
                        if sk_start + sk_j < Sk:
                            sk_j_offs = sk_start + sk_j

                            ij_mask = tl.sum(topk_indices == sk_j_offs) > 0
                            
                            sparse_mask = tl.where(sq_i_valid & (sq_strides[:, None] == sq_i) & (sk_strides[None, :] == sk_j), ij_mask + sparse_mask, sparse_mask)
            
            sparse_mask = tl.where(sparse_mask > 0, 0.0, float("-inf"))
            index_scores = index_scores + sparse_mask
        
        # Recompute attention scores
        attn_scores = tl.zeros([AH, BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
        for h in tl.range(AH):
            aq_head_base = aq_base + h * stride_aqh
            ak_head_base = ak_base + h * stride_akh
            
            dot = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            for d_start in tl.range(0, AD, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < AD
                
                aq_ptrs = aq_head_base + aq[:, None] * stride_asq + d_offs[None, :] * stride_aqd
                aq_vals = tl.load(aq_ptrs, mask=(aq_valid[:, None] & d_valid[None, :]), other=0.0)
                
                ak_ptrs = ak_head_base + sk_offs[None, :] * stride_ask + d_offs[:, None] * stride_akd
                ak_vals = tl.load(ak_ptrs, mask=(sk_valid[None, :] & d_valid[:, None]), other=0.0)
                
                dot += tl.dot(aq_vals, ak_vals)
            
            dot = dot * Softmax_Scale + causal_mask
            if SPARSE_LOSS:
                dot = dot + sparse_mask
            h_idx = tl.arange(0, AH)
            attn_scores = tl.where(h_idx[:, None, None] == h, dot[None, :, :], attn_scores)
        
        # Compute softmax values
        index_scores_softmax = tl.exp(index_scores - m1_i[:, None]) / d1_i[:, None]
        attn_scores_softmax = tl.exp(attn_scores - m_i[:, :, None]) / d_i[:, :, None]
        
        # Sum and normalize attention scores
        attn_scores_sum = tl.sum(attn_scores_softmax, axis=0) / AH

        # Gradient of KL divergence w.r.t. index_scores_softmax
        grad_index_softmax = -attn_scores_sum / (index_scores_softmax + 1e-10) * Grad_Loss_Scale        

        grad_index_logits = index_scores_softmax * (grad_index_softmax - sum_grad)
        
        # Apply valid mask
        valid_mask = (sq[:, None] >= sk_offs[None, :])
        if SPARSE_LOSS:
            valid_mask = valid_mask & (sparse_mask == 0.0)
        grad_index_logits = tl.where(valid_mask, grad_index_logits, 0.0)

        for h in tl.range(H):
            w_val = tl.load(w_base + sq * stride_ws + h * stride_wh, mask=sq_valid, other=0.0)
            q_head_base = q_base + h * stride_qh
            
            # Compute scores = q @ k.T [BLOCK_SQ, BLOCK_SK]
            dot = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            for d_start in tl.range(0, D, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < D
                
                q_ptrs = q_head_base + sq[:, None] * stride_qs + d_offs[None, :] * stride_qd
                q_vals = tl.load(q_ptrs, mask=(sq_valid[:, None] & d_valid[None, :]), other=0.0)
                
                k_ptrs = k_base + sk_offs[None, :] * stride_ks + d_offs[:, None] * stride_kd
                k_vals = tl.load(k_ptrs, mask=(sk_valid[None, :] & d_valid[:, None]), other=0.0)
                
                dot += tl.dot(q_vals, k_vals)
            
            # ReLU activation and mask
            scores_relu = tl.maximum(dot, 0.0)
            relu_mask = (dot > 0.0).to(tl.float32)

            # grad_weights: sum(grad_logits * scores_relu, dim=sk)
            # [BLOCK_SQ, BLOCK_SK] --sum over sk--> [BLOCK_SQ]
            # [sq, b, 1, sk] * [sq, b, h, sk] -> [sq, b, h]
            grad_w_val = tl.sum(grad_index_logits * scores_relu, axis=-1)
            grad_w_ptrs = Grad_W_ptr + sq * stride_gws + b * stride_gwb + h * stride_gwh
            tl.atomic_add(grad_w_ptrs, grad_w_val, mask=sq_valid)

            # grad_scores = grad_logits * weights * relu_mask
            grad_scores = grad_index_logits * w_val[:, None] * relu_mask

            # Compute grad_q for this head and write with atomic add
            for d_start in tl.range(0, D, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < D

                k_ptrs = k_base + sk_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd
                k_vals = tl.load(k_ptrs, mask=(sk_valid[:, None] & d_valid[None, :]), other=0.0)
                
                # grad_q: grad_scores @ k [BLOCK_SQ, BLOCK_SK] @ [BLOCK_SK, BLOCK_D]
                grad_q_part = tl.dot(grad_scores, k_vals.to(tl.float32))
                grad_q_base = Grad_Q_ptr + b * stride_gqb + h * stride_gqh
                grad_q_ptrs = grad_q_base + sq[:, None] * stride_gqs + d_offs[None, :] * stride_gqd
                tl.atomic_add(grad_q_ptrs, grad_q_part, mask=(sq_valid[:, None] & d_valid[None, :]))

                q_ptrs = q_head_base + sq[:, None] * stride_qs + d_offs[None, :] * stride_qd
                q_vals = tl.load(q_ptrs, mask=(sq_valid[:, None] & d_valid[None, :]), other=0.0)                

                # Compute partial grad_k: grad_scores.T @ q [BLOCK_SK, BLOCK_SQ] @ [BLOCK_SQ, BLOCK_D]
                partial_grad_k = tl.dot(tl.trans(grad_scores), q_vals.to(tl.float32))
                partial_base = Grad_K_ptr + b * stride_pgb
                partial_ptrs = partial_base + sk_offs[:, None] * stride_pgk + d_offs[None, :] * stride_pgd
                tl.atomic_add(partial_ptrs, partial_grad_k, mask=(sk_valid[:, None] & d_valid[None, :]))


def bwd_fused_indexer_loss(
    q: torch.Tensor,
    weights: torch.Tensor, 
    k: torch.Tensor, 
    query: torch.Tensor, 
    key: torch.Tensor, 
    topk_indices: torch.Tensor,
    softmax_scale: float, 
    loss_coeff: float, 
    sparse_loss: bool,
    grad_loss: torch.Tensor,
    Sq_offset: int = 0,
):
    """
    Fully-fused Triton implementation of backward pass.
    
    Uses three Triton kernels:
    1. Compute grad_index_logits
    2. Compute grad_q and grad_weights (no atomics needed)
    3. Compute grad_k using two-phase reduction (avoid atomics)
    
    This is more complex than the hybrid approach but potentially faster
    for large problem sizes.
    """
    Sq, B, H, D = q.shape
    Sk = k.shape[0]
    ASq, AB, AH, AD = query.shape
    ASk = key.shape[0]
    
    BLOCK_SQ = 16  # Must be >= 16 for Triton tl.dot
    BLOCK_SK = 64
    BLOCK_D = 64
    
    grad_q = torch.zeros_like(q, dtype=torch.float32)
    grad_weights = torch.zeros_like(weights, dtype=torch.float32)
    num_sq_blocks = triton.cdiv(Sq, BLOCK_SQ)
    grad_k = torch.zeros(B, Sk, D, device=q.device, dtype=torch.float32)

    grid1 = (B, num_sq_blocks)
    grad_loss_scale = grad_loss.item() * loss_coeff / (B * Sq)
    
    # Get topk
    topk = topk_indices.size(-1)
    BLOCK_TOPK = min(topk, 2048)
    
    _bwd_fused_indexer_loss_kernel[grid1](
        Q_ptr=q,
        K_ptr=k,
        W_ptr=weights,
        Attn_Query_ptr=query,
        Attn_Key_ptr=key,
        Topk_Idx_ptr=topk_indices,
        Grad_Q_ptr=grad_q,
        Grad_W_ptr=grad_weights,
        Grad_K_ptr=grad_k,
        stride_qs=q.stride(0),
        stride_qb=q.stride(1),
        stride_qh=q.stride(2),
        stride_qd=q.stride(3),
        stride_ks=k.stride(0),
        stride_kb=k.stride(1),
        stride_kd=k.stride(2),
        stride_ws=weights.stride(0),
        stride_wb=weights.stride(1),
        stride_wh=weights.stride(2),
        stride_asq=query.stride(0),
        stride_aqb=query.stride(1),
        stride_aqh=query.stride(2),
        stride_aqd=query.stride(3),
        stride_ask=key.stride(0),
        stride_akb=key.stride(1),
        stride_akh=key.stride(2),
        stride_akd=key.stride(3),
        stride_tb=topk_indices.stride(0),
        stride_ts=topk_indices.stride(1),
        stride_tk=topk_indices.stride(2),
        stride_gqs=grad_q.stride(0),
        stride_gqb=grad_q.stride(1),
        stride_gqh=grad_q.stride(2),
        stride_gqd=grad_q.stride(3),
        stride_gws=grad_weights.stride(0),
        stride_gwb=grad_weights.stride(1),
        stride_gwh=grad_weights.stride(2),
        stride_pgb=grad_k.stride(0),
        stride_pgk=grad_k.stride(1),
        stride_pgd=grad_k.stride(2),
        H=H,
        D=D,
        AH=AH,
        AD=AD,
        Sq=Sq,
        Sk=Sk,
        TopK=topk,
        BLOCK_SQ=BLOCK_SQ,
        BLOCK_SK=BLOCK_SK,
        BLOCK_D=BLOCK_D,
        BLOCK_TOPK=BLOCK_TOPK,
        Softmax_Scale=softmax_scale,
        Grad_Loss_Scale=grad_loss_scale,
        SPARSE_LOSS=sparse_loss,
        Sq_offset=Sq_offset,
        ASq=ASq,
    )

    grad_k = grad_k.permute(1, 0, 2)
    
    return grad_q.to(q.dtype), grad_weights.to(weights.dtype), grad_k.to(k.dtype)


class FusedDSAIndexerLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        q: torch.Tensor, 
        weights: torch.Tensor, 
        k: torch.Tensor, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        softmax_scale: float, 
        topk: int, 
        loss_coeff: float, 
        mask: torch.Tensor, 
        sparse_loss: bool, 
        pg_collection: Optional[ProcessGroupCollection],
        accuracy_check: bool,
    ):
        """
        Fused forward: index_scores never materialized in full.
        """

        # tensor parallel
        Sq_offset = 0
        if pg_collection is not None and pg_collection.tp.size() > 1:
            Sq, B, H, D = q.shape
            Sk = k.shape[0]

            ASq, AB, AH, AD = query.shape
            ASk = key.shape[0]

            tp_size = pg_collection.tp.size()

            # all-to-all for attn query
            assert ASq % tp_size == 0, f"ASq {ASq} % tp_size {tp_size} != 0"
            # [ASq, B, H, D] -> [tp_size, H, ASq // tp_size, B, D]
            view_attn_q = (
                query.permute(2, 0, 1, 3)
                .reshape(AH, tp_size, ASq // tp_size, AB, AD)
                .transpose(0, 1)
                .contiguous()
            )
            output_attn_q = torch.empty_like(view_attn_q)
            dist.all_to_all_single(
                output_attn_q, 
                view_attn_q,
                group=pg_collection.tp
            )
            # [tp_size, H, ASq // tp_size, B, D] -> [ASq // tp_size, B, H * tp_size, D]
            query = (
                output_attn_q.reshape(AH * tp_size, ASq // tp_size, AB, AD)
                .permute(1, 2, 0, 3)
                .contiguous()
            )

            # all-gather for attn key
            gathered_attn_k = torch.empty(
                (tp_size, *key.shape), 
                device=key.device, 
                dtype=key.dtype
            )
            dist.all_gather_into_tensor(gathered_attn_k, key.contiguous(), group=pg_collection.tp)
            # [tp_size, Sk, B, H, D] -> [Sk, B, H * tp_size, D]
            key = (
                gathered_attn_k.permute(1, 2, 0, 3, 4)
                .reshape(ASk, AB, AH * tp_size, AD)
                .contiguous()
            )

            ASq, AB, AH, AD = query.shape
            AHk = key.shape[2]

            assert Sq == ASq * tp_size and \
                Sk == ASk and AH == H and AHk == H and \
                AB == B and AD == D
            
            # Do not split index scores, it introduces extra problem in communication and casual mask
            # Sq should be within (Sq_offset, Sq_offset + Sq)
            tp_rank = pg_collection.tp.rank()
            Sq_offset = Sq // tp_size * tp_rank
            assert Sq_offset + ASq <= Sq

        # Run fused Triton kernel
        topk_indices, loss, _ = fwd_fused_indexer_loss(
            q, weights, k, query, key, topk, softmax_scale, loss_coeff, mask, sparse_loss=sparse_loss, Sq_offset=Sq_offset,
            # TODO: remove accuracy_check
            accuracy_check=accuracy_check,
        )

        if pg_collection is not None and pg_collection.tp.size() > 1:
            # reduce loss
            dist.all_reduce(loss, group=pg_collection.tp)
            loss /= pg_collection.tp.size()
        
        # Save for backward (recomputation strategy)
        ctx.save_for_backward(q, weights, k, query, key, topk_indices)
        ctx.softmax_scale = softmax_scale
        ctx.loss_coeff = loss_coeff
        ctx.sparse_loss = sparse_loss
        ctx.Sq_offset = Sq_offset
        
        return topk_indices, loss
    
    @staticmethod
    def backward(
        ctx, 
        grad_topk_indices: torch.Tensor, 
        grad_loss: torch.Tensor
    ):
        """
        Backward: Recompute what we need.
        """
        q, weights, k, query, key, topk_indices = ctx.saved_tensors

        grad_q , grad_weights, grad_k = bwd_fused_indexer_loss(
            q, weights, k, query, key, topk_indices, 
            ctx.softmax_scale, ctx.loss_coeff, ctx.sparse_loss, grad_loss, ctx.Sq_offset
        )
        
        # query and key are detached in forward, so return None for their gradients
        return grad_q, grad_weights, grad_k, None, None, None, None, None, None, None, None, None
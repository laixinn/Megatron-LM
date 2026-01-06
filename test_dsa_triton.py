"""
Test suite for DSA (Dynamic Sparse Attention) indexer loss backward pass.

This file contains:
1. backward_native: Pure PyTorch implementation of the manual backward pass
2. Test functions to validate both implementations against PyTorch autograd

The DSA indexer loss computes KL divergence between:
- Index scores: Predicted importance scores from the indexer network
- Attention scores: True attention scores from the full attention mechanism

Backward pass computes gradients w.r.t.:
- q: Indexer query embeddings [Sq, B, H, D]
- weights: Indexer attention weights [Sq, B, H]  
- k: Indexer key embeddings [Sk, B, D]
"""

import torch
import torch.distributed as dist

import triton
import triton.language as tl

import numpy as np

from megatron.core.transformer.experimental_attention_variant.dsa_triton import compute_dsa_indexer_loss_triton
from megatron.core.process_groups_config import ProcessGroupCollection
import megatron.core.parallel_state as parallel_state

@triton.jit
def _compute_grad_index_logits_kernel(
    Q_ptr,
    K_ptr,
    W_ptr,
    Attn_Query_ptr,
    Attn_Key_ptr,
    Topk_Indices_ptr,
    Index_Mask_ptr, # [BLOCK_SQ, BLOCK_SK]
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
    # Index mask strides: [BLOCK_SQ, Sk]
    stride_imsq,
    stride_imsk,
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
    Loss_Coeff: tl.constexpr,
    Grad_Loss_Scale: tl.constexpr,
    SPARSE_LOSS: tl.constexpr,
):
    """
    Compute gradient of KL loss w.r.t. index_scores logits (before softmax).
    This is the first step of the backward pass - compute grad_index_logits.
    """
    b = tl.program_id(0)
    sq_block_id = tl.program_id(1)
    
    sq = sq_block_id * BLOCK_SQ + tl.arange(0, BLOCK_SQ)
    sq_valid = sq < Sq
    
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
            # sq_strides = tl.arange(0, BLOCK_SQ)
            # sk_strides = tl.arange(0, BLOCK_SK)

            # tl.store(Index_Mask_ptr + sq_strides[:, None] * stride_imsq + sk_strides[None, :] * stride_imsk, float("-inf"))

            # for topk_i in tl.range(tl.cdiv(TopK, BLOCK_TOPK)):
            #     topk_off = topk_i * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
            #     topk_valid = topk_off < TopK

            #     topk_indices = tl.load(Topk_Indices_ptr + b * stride_tb + sq[:, None] * stride_ts + topk_off[None, :] * stride_tk, mask=sq_valid[:, None] & topk_valid[None, :], other=0)  # [BLOCK_SQ, BLOCK_TOPK]

            #     topk_indices_norm = topk_indices - sk_start
            #     topk_idx_valid = (topk_indices_norm >= 0) & (topk_indices_norm < BLOCK_SK)
            #     # addr = topk_indices + sq_strides
            #     topk_indices_ptrs = sq_strides[:, None] * stride_imsq + topk_indices_norm * stride_imsk
            #     # index_mask: [BLOCK_SQ, BLOCK_SK]
            #     # tl.store(Index_Mask_ptr + topk_indices_ptrs, 0.0, mask=sq_valid[:, None] & topk_valid[None, :] & topk_idx_valid)
            #     tl.atomic_max(Index_Mask_ptr + topk_indices_ptrs, 0.0, mask=topk_idx_valid & sq_valid[:, None] & topk_valid[None, :])

            # tl.debug_barrier()

            # sparse_mask = tl.load(Index_Mask_ptr + sq_strides[:, None] * stride_imsq + sk_strides[None, :] * stride_imsk, mask=sq_valid[:, None] & sk_valid[None, :], other=0.0)

            sparse_mask = tl.full([BLOCK_SQ, BLOCK_SK], float("-inf"), dtype=tl.float32)
            for i in range(BLOCK_SQ):
                sq_i = sq_block_id * BLOCK_SQ + i
                for topk_i in tl.range(tl.cdiv(TopK, BLOCK_TOPK)):
                    topk_off = topk_i * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
                    topk_valid = topk_off < TopK

                    topk_indices = tl.load(Topk_Indices_ptr + b * stride_tb + sq_i * stride_ts + topk_off * stride_tk, mask=(sq_i < Sq) & topk_valid, other=0)

                    topk_indices_norm = topk_indices - sk_start

                    topk_mask = tl.sum(topk_indices_norm[:, None] == sk_offs[None, :], axis=0) > 0

                    sparse_mask = tl.where(topk_mask, 0.0, sparse_mask)


            # sparse_mask = tl.full([BLOCK_SQ, BLOCK_SK], 0.0, dtype=tl.float32)
            
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
                
                aq_ptrs = aq_head_base + sq[:, None] * stride_asq + d_offs[None, :] * stride_aqd
                aq_vals = tl.load(aq_ptrs, mask=(sq_valid[:, None] & d_valid[None, :]), other=0.0)
                
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
            # sq_strides = tl.arange(0, BLOCK_SQ)
            # sk_strides = tl.arange(0, BLOCK_SK)

            # tl.store(Index_Mask_ptr + sq_strides[:, None] * stride_imsq + sk_strides[None, :] * stride_imsk, float("-inf"))

            # for topk_i in tl.range(tl.cdiv(TopK, BLOCK_TOPK)):
            #     topk_off = topk_i * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
            #     topk_valid = topk_off < TopK

            #     topk_indices = tl.load(Topk_Indices_ptr + b * stride_tb + sq[:, None] * stride_ts + topk_off[None, :] * stride_tk, mask=sq_valid[:, None] & topk_valid[None, :], other=0)  # [BLOCK_SQ, BLOCK_TOPK]

            #     topk_indices_norm = topk_indices - sk_start
            #     topk_idx_valid = (topk_indices_norm >= 0) & (topk_indices_norm < BLOCK_SK)
            #     # addr = topk_indices + sq_strides
            #     topk_indices_ptrs = topk_indices_norm * stride_imsk + sq_strides[:, None] * stride_imsq
            #     # index_mask: [BLOCK_SQ, BLOCK_SK]
            #     # tl.store(Index_Mask_ptr + topk_indices_ptrs, 0.0, mask=sq_valid[:, None] & topk_valid[None, :] & topk_idx_valid)
            #     tl.atomic_max(Index_Mask_ptr + topk_indices_ptrs, 0.0, mask=topk_idx_valid & sq_valid[:, None] & topk_valid[None, :])

            # tl.debug_barrier()

            # sparse_mask = tl.load(Index_Mask_ptr + sq_strides[:, None] * stride_imsq + sk_strides[None, :] * stride_imsk, mask=sq_valid[:, None] & sk_valid[None, :], other=0.0)

            sparse_mask = tl.full([BLOCK_SQ, BLOCK_SK], float("-inf"), dtype=tl.float32)
            for i in range(BLOCK_SQ):
                sq_i = sq_block_id * BLOCK_SQ + i
                for topk_i in tl.range(tl.cdiv(TopK, BLOCK_TOPK)):
                    topk_off = topk_i * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
                    topk_valid = topk_off < TopK

                    topk_indices = tl.load(Topk_Indices_ptr + b * stride_tb + sq_i * stride_ts + topk_off * stride_tk, mask=(sq_i < Sq) & topk_valid, other=0)

                    topk_indices_norm = topk_indices - sk_start

                    topk_mask = tl.sum(topk_indices_norm[:, None] == sk_offs[None, :], axis=0) > 0

                    sparse_mask = tl.where(topk_mask, 0.0, sparse_mask)


            # sparse_mask = tl.full([BLOCK_SQ, BLOCK_SK], 0.0, dtype=tl.float32)
            
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
                
                aq_ptrs = aq_head_base + sq[:, None] * stride_asq + d_offs[None, :] * stride_aqd
                aq_vals = tl.load(aq_ptrs, mask=(sq_valid[:, None] & d_valid[None, :]), other=0.0)
                
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
            sq_strides = tl.arange(0, BLOCK_SQ)
            sk_strides = tl.arange(0, BLOCK_SK)

            # tl.store(Index_Mask_ptr + sq_strides[:, None] * stride_imsq + sk_strides[None, :] * stride_imsk, float("-inf"))

            # for topk_i in tl.range(tl.cdiv(TopK, BLOCK_TOPK)):
            #     topk_off = topk_i * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
            #     topk_valid = topk_off < TopK

            #     topk_indices = tl.load(Topk_Indices_ptr + b * stride_tb + sq[:, None] * stride_ts + topk_off[None, :] * stride_tk, mask=sq_valid[:, None] & topk_valid[None, :], other=0)  # [BLOCK_SQ, BLOCK_TOPK]

            #     topk_indices_norm = topk_indices - sk_start
            #     topk_idx_valid = (topk_indices_norm >= 0) & (topk_indices_norm < BLOCK_SK)
            #     # addr = topk_indices + sq_strides
            #     topk_indices_ptrs = topk_indices_norm * stride_imsk + sq_strides[:, None] * stride_imsq
            #     # index_mask: [BLOCK_SQ, BLOCK_SK]
            #     # tl.store(Index_Mask_ptr + topk_indices_ptrs, 0.0, mask=sq_valid[:, None] & topk_valid[None, :] & topk_idx_valid)
            #     tl.atomic_max(Index_Mask_ptr + topk_indices_ptrs, 0.0, mask=topk_idx_valid & sq_valid[:, None] & topk_valid[None, :])

            # tl.debug_barrier()

            # sparse_mask = tl.load(Index_Mask_ptr + sq_strides[:, None] * stride_imsq + sk_strides[None, :] * stride_imsk, mask=sq_valid[:, None] & sk_valid[None, :], other=0.0)

            # if (sparse_mask != 0).sum() > 0:
            #     tl.device_print("sparse_mask != 0", (sparse_mask != 0).sum())
            #     for i in range(BLOCK_SQ):
            #         for j in range(BLOCK_SK):
            #             val = tl.load(Index_Mask_ptr + i * stride_imsq + j * stride_imsk)
            #             if val != 0.0:
            #                 tl.device_print("i", i)
            #                 tl.device_print("j", j)
            #                 tl.device_print("val", val)
            #     tl.store(Index_Mask_ptr + sq_strides[:, None] * stride_imsq + sk_strides[None, :] * stride_imsk, sparse_mask)

            sparse_mask = tl.full([BLOCK_SQ, BLOCK_SK], float("-inf"), dtype=tl.float32)
            for i in range(BLOCK_SQ):
                sq_i = sq_block_id * BLOCK_SQ + i
                for topk_i in tl.range(tl.cdiv(TopK, BLOCK_TOPK)):
                    topk_off = topk_i * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
                    topk_valid = topk_off < TopK

                    topk_indices = tl.load(Topk_Indices_ptr + b * stride_tb + sq_i * stride_ts + topk_off * stride_tk, mask=(sq_i < Sq) & topk_valid, other=0)

                    topk_indices_norm = topk_indices - sk_start

                    if b == 0 and sq_i == 15 and topk_i == 0:
                        tl.device_print("topk_indices_norm", topk_indices_norm)

                    topk_mask = tl.sum(topk_indices_norm[:, None] == sk_offs[None, :], axis=0) > 0

                    sparse_mask = tl.where(topk_mask, 0.0, sparse_mask)

            tl.store(Index_Mask_ptr + sq_strides[:, None] * stride_imsq + sk_strides[None, :] * stride_imsk, sparse_mask)

            # sparse_mask = tl.full([BLOCK_SQ, BLOCK_SK], 0.0, dtype=tl.float32)
            
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
                
                aq_ptrs = aq_head_base + sq[:, None] * stride_asq + d_offs[None, :] * stride_aqd
                aq_vals = tl.load(aq_ptrs, mask=(sq_valid[:, None] & d_valid[None, :]), other=0.0)
                
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
                grad_q_part = tl.dot(grad_scores, k_vals)
                grad_q_base = Grad_Q_ptr + b * stride_gqb + h * stride_gqh
                grad_q_ptrs = grad_q_base + sq[:, None] * stride_gqs + d_offs[None, :] * stride_gqd
                tl.atomic_add(grad_q_ptrs, grad_q_part, mask=(sq_valid[:, None] & d_valid[None, :]))

                q_ptrs = q_head_base + sq[:, None] * stride_qs + d_offs[None, :] * stride_qd
                q_vals = tl.load(q_ptrs, mask=(sq_valid[:, None] & d_valid[None, :]), other=0.0)                

                # Compute partial grad_k: grad_scores.T @ q [BLOCK_SK, BLOCK_SQ] @ [BLOCK_SQ, BLOCK_D]
                partial_grad_k = tl.dot(tl.trans(grad_scores), q_vals)
                partial_base = Grad_K_ptr + b * stride_pgb
                partial_ptrs = partial_base + sk_offs[:, None] * stride_pgk + d_offs[None, :] * stride_pgd
                tl.atomic_add(partial_ptrs, partial_grad_k, mask=(sk_valid[:, None] & d_valid[None, :]))


def backward_triton_full(
    q, weights, k, query, key, topk_indices,
    softmax_scale, loss_coeff, sparse_loss,
    grad_loss
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
    sq, b, np, hn = query.size()
    sk = key.size(0)
    h = weights.size(2)  # indexer heads
    d = q.size(3)  # indexer dimension
    
    BLOCK_SQ = 16  # Must be >= 16 for Triton tl.dot
    BLOCK_SK = 64
    BLOCK_D = 64
    
    grad_q = torch.zeros_like(q, dtype=torch.float32)
    grad_weights = torch.zeros_like(weights, dtype=torch.float32)
    num_sq_blocks = triton.cdiv(sq, BLOCK_SQ)
    grad_k = torch.zeros(b, sk, d, device=q.device, dtype=torch.float32)
    index_mask = torch.zeros([BLOCK_SQ, BLOCK_SK], device=q.device, dtype=torch.float32)

    grid1 = (b, num_sq_blocks)
    grad_loss_scale = grad_loss.item() * loss_coeff / (b * sq)
    
    # Get topk
    topk = topk_indices.size(-1)
    BLOCK_TOPK = 8
    
    _compute_grad_index_logits_kernel[grid1](
        q, k, weights,
        query, key,
        topk_indices,
        index_mask,
        grad_q, grad_weights, grad_k,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2),
        weights.stride(0), weights.stride(1), weights.stride(2),
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        topk_indices.stride(0), topk_indices.stride(1), topk_indices.stride(2),
        index_mask.stride(0), index_mask.stride(1),
        grad_q.stride(0), grad_q.stride(1), grad_q.stride(2), grad_q.stride(3),
        grad_weights.stride(0), grad_weights.stride(1), grad_weights.stride(2),
        grad_k.stride(0), grad_k.stride(1), grad_k.stride(2),
        h, d, np, hn, sq, sk, topk,
        BLOCK_SQ, BLOCK_SK, BLOCK_D, BLOCK_TOPK,
        softmax_scale, loss_coeff, grad_loss_scale,
        sparse_loss,
    )

    grad_k = grad_k.permute(1, 0, 2)

    # print(f"{index_mask=}")
    print(torch.isinf(index_mask).nonzero())
    
    return grad_q.to(q.dtype), grad_weights.to(weights.dtype), grad_k.to(k.dtype)


def bench(fn, num_warmups: int = 5, num_tests: int = 30, post_fn=None, is_async=True):
    # Flush L2 cache with 256 MB data
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Flush L2
    cache.zero_()

    torch.cuda.synchronize()
    dist.barrier()

    if is_async:
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
        for i in range(num_tests):
            # Record
            start_events[i].record()
            fn()
            end_events[i].record()
            if post_fn is not None:
                post_fn()

        torch.cuda.synchronize()

        times = np.array([s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)])[1:]
        return np.median(times), np.min(times), np.max(times)
    else:
        start_events = torch.cuda.Event(enable_timing=True)
        end_events = torch.cuda.Event(enable_timing=True)
        start_events.record()
        for _ in range(num_tests):
            fn()
            if post_fn is not None:
                post_fn()
        end_events.record()
        
        torch.cuda.synchronize()

        times = start_events.elapsed_time(end_events) / 1e3 / num_tests

        return times, times, times

# TOPK
def _compute_index_scores_topk_native(
    q: torch.Tensor, 
    weights: torch.Tensor, 
    k: torch.Tensor, 
    mask: torch.Tensor, 
    topk: int
) -> torch.Tensor:
    scores = _compute_index_scores(q, weights, k)
    scores = scores + mask
    topk = min(topk, q.size(0))
    topk_indices = scores.topk(topk, dim=-1)[1]
    softmax_mi = scores.max(dim=-1)[0]
    softmax_di = (scores-softmax_mi.unsqueeze(-1)).exp().sum(dim=-1)
    return scores, topk_indices, softmax_mi, softmax_di



def _compute_index_scores(q: torch.Tensor, weights: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    index_scores = torch.einsum('sbhd,tbd->sbht', q.float(), k.float())

    # Apply ReLU activation.
    index_scores = torch.relu(index_scores)

    # Weight each head by attention weights.
    # [seqlen_q, batch, index_n_heads, seqlen_k] * [seqlen_q, batch, index_n_heads, 1]
    #   -> [seqlen_q, batch, index_n_heads, seqlen_k]
    index_scores = index_scores * weights.unsqueeze(-1)

    # Sum across attention heads.
    # [seqlen_q, batch, index_n_heads, seqlen_k] -> [seqlen_q, batch, seqlen_k]
    index_scores = index_scores.sum(dim=2)

    # Transpose to [batch, seqlen_q, seqlen_k].
    index_scores = index_scores.transpose(0, 1).contiguous()

    return index_scores

def compute_dsa_indexer_loss(
    index_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    pg_collection: ProcessGroupCollection = None,
    attention_scores: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute KL divergence loss between index_scores and true attention_scores.

    This loss trains the indexer to predict which tokens are important by matching the distribution
    of true attention scores.

    Reference: Section 2.1 of
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf

    Args:
        index_scores: Scores predicted by indexer [batch, seqlen_q, seqlen_k].
        topk_indices: Top-k indices [batch, seqlen_q, index_topk].
        query: Query tensor [seqlen_q, batch, heads, dim].
        key: Key tensor [seqlen_k, batch, heads, dim].
        softmax_scale: Scale coefficient after q @ k^T.
        loss_coeff: Coefficient for the indexer KL divergence loss.
        sparse_loss: bool, whether to use sparse indexer loss. If True, only the topk
            indices will be used to compute the loss.
        pg_collection: Process group collection, must have TP process group.

    Returns:
        index_loss: KL divergence loss (scalar).
    """
    sq, b, np, hn = query.size()
    sk = key.size(0)

    if attention_scores is None:
        # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
        query = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
        # [sk, b, np, hn] -> [b, np, hn, sk] -> [b * np, hn, sk]
        key = key.permute(1, 2, 3, 0).reshape(b * np, hn, sk)
        # Compute attention scores [b * np, sq, sk]
        attention_scores = torch.bmm(query.float(), key.float()) * softmax_scale
        # Reshape to [b, np, sq, sk]
        attention_scores = attention_scores.reshape(b, np, sq, sk)

    # causal_mask [sq, sk]
    causal_mask = torch.triu(
        torch.full((sq, sk), float('-inf'), dtype=torch.float32, device=attention_scores.device),
        diagonal=1,
    )
    # index_mask [b, sq, sk]
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=causal_mask.device
    ).scatter_(-1, topk_indices, 0)

    # [b, np, sq, skv] + [1, 1, sq, skv] -> [b, np, sq, skv]
    attention_scores += causal_mask.view(1, 1, sq, sk)
    if sparse_loss:
        # [b, np, sq, sk] + [b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores += index_mask.view(b, 1, sq, sk)
        # [b, sq, sk] + [b, sq, sk] -> [b, sq, sk]
        index_scores += index_mask

    # [b, np, sq, sk] -> [b, np, sq, sk]
    attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32)
    # [b, sq, sk] -> [b, sq, sk]
    index_scores = torch.nn.functional.softmax(index_scores, dim=-1, dtype=torch.float32)

    # Sum attention scores across heads.
    # [batch, heads, seqlen_q, seqlen_k] -> [batch, seqlen_q, seqlen_k]
    attention_scores = attention_scores.sum(dim=1)
    if pg_collection is not None and pg_collection.tp.size() > 1:
        # attention scores are scattered to TP ranks in head dimension.
        torch.distributed.all_reduce(attention_scores.contiguous(), group=pg_collection.tp)
    # L1 normalize target on the last dimension. Doesn't use abs() because attention_scores are
    # obtained from softmax so they are already non-negative.
    attention_scores = attention_scores / attention_scores.sum(dim=-1, keepdim=True)

    # Compute KL divergence: KL(target || index) = target(x) * log(target(x) / index(x))
    # kl_per_element [b, sq, sk]
    kl_per_element = attention_scores * (
        torch.log(attention_scores + 1e-10) - torch.log(index_scores + 1e-10)
    )

    # [b, sq, sk] -> [b, sq] -> [1]
    # Each token has same weight in the loss.
    kl_div = kl_per_element.sum(dim=-1).mean()

    # Scale by coefficient.
    indexer_loss = kl_div * loss_coeff

    return indexer_loss, kl_per_element


def compute_dsa_indexer_loss_1pass(
    index_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    # pg_collection: ProcessGroupCollection,
    attention_scores: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute KL divergence loss between index_scores and true attention_scores.

    This loss trains the indexer to predict which tokens are important by matching the distribution
    of true attention scores.

    Reference: Section 2.1 of
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf

    Args:
        index_scores: Scores predicted by indexer [batch, seqlen_q, seqlen_k].
        topk_indices: Top-k indices [batch, seqlen_q, index_topk].
        query: Query tensor [seqlen_q, batch, heads, dim].
        key: Key tensor [seqlen_k, batch, heads, dim].
        softmax_scale: Scale coefficient after q @ k^T.
        loss_coeff: Coefficient for the indexer KL divergence loss.
        sparse_loss: bool, whether to use sparse indexer loss. If True, only the topk
            indices will be used to compute the loss.
        pg_collection: Process group collection, must have TP process group.

    Returns:
        index_loss: KL divergence loss (scalar).
    """
    sq, b, np, hn = query.size()
    sk = key.size(0)

    if attention_scores is None:
        # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
        query = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
        # [sk, b, np, hn] -> [b, np, hn, sk] -> [b * np, hn, sk]
        key = key.permute(1, 2, 3, 0).reshape(b * np, hn, sk)
        # Compute attention scores [b * np, sq, sk]
        attention_scores = torch.bmm(query.float(), key.float()) * softmax_scale
        # Reshape to [b, np, sq, sk]
        attention_scores = attention_scores.reshape(b, np, sq, sk)

    m_i = torch.full((b, np, sq), float('-inf'), device=attention_scores.device, dtype=torch.float32)
    d_i = torch.zeros((b, np, sq), device=attention_scores.device, dtype=torch.float32)
    m1_i = torch.full((b, sq), float('-inf'), device=attention_scores.device, dtype=torch.float32)
    d1_i = torch.zeros((b, sq), device=attention_scores.device, dtype=torch.float32)
    loss_i = torch.zeros((b, sq), device=attention_scores.device, dtype=torch.float32)

    for i in range(sk):
        attn_scores_k = attention_scores[:, :, :, i]

        index_scores_k = index_scores[:, :, i]

        m_i_1 = m_i
        m1_i_1 = m1_i

        # attn scores with head dim
        m_i = torch.max(m_i, attn_scores_k)
        d_i = d_i * torch.exp(m_i_1 - m_i) + torch.exp(attn_scores_k - m_i)

        # index scores without head dim
        m1_i = torch.max(m1_i, index_scores_k)
        d1_i = d1_i * torch.exp(m1_i_1 - m1_i) + torch.exp(index_scores_k - m1_i)

    for i in range(sk):
        attn_scores_k = attention_scores[:, :, :, i]
        index_scores_k = index_scores[:, :, i]

        # softmax compute
        softmax_attn_i = torch.exp(attn_scores_k - m_i) / d_i
        softmax_index_i = torch.exp(index_scores_k - m1_i) / d1_i

        # reduce head dim
        softmax_attn_i = softmax_attn_i.sum(dim=1) / np

        # loss
        loss_i += softmax_attn_i * (torch.log(softmax_attn_i + 1e-10) - torch.log(softmax_index_i + 1e-10))

    loss = loss_i.mean() * loss_coeff

    return loss, loss_i



def forward_native(q, weights, k, mask, index_topk, query, key, softmax_scale, loss_coeff, sparse_loss, pg_collection=None):
    index_scores = _compute_index_scores(q, weights, k)

    if mask is not None:
        assert mask.dtype == index_scores.dtype, "Mask dtype must match index scores dtype"
        index_scores = index_scores + mask

    # =========================================
    # Select top-k indices
    # =========================================
    seqlen = index_scores.size(-1)
    topk_k = min(index_topk, seqlen)
    # [batch, seqlen, index_topk]
    topk_indices = index_scores.topk(topk_k, dim=-1)[1]

    indexer_loss, kl_per_element = compute_dsa_indexer_loss(
        index_scores.clone(), topk_indices, query, key, softmax_scale, loss_coeff, sparse_loss, pg_collection=pg_collection
    )

    return topk_indices, indexer_loss, kl_per_element, index_scores




def backward_native(
    q, weights, k, query, key, topk_indices, 
    softmax_scale, loss_coeff, sparse_loss,
    grad_loss
):
    """
    Pure PyTorch implementation of backward pass for DSA indexer loss.
    
    This function computes gradients of the KL divergence loss w.r.t. the indexer
    parameters (q, weights, k). It uses recomputation to save memory - forward
    values are recomputed during backward instead of being cached.
    
    Args:
        q: Indexer query embeddings [Sq, B, H, D]
        weights: Indexer attention weights [Sq, B, H]
        k: Indexer key embeddings [Sk, B, D]
        query: Attention query embeddings [Sq, B, AH, AD]
        key: Attention key embeddings [Sk, B, AH, AD]
        topk_indices: Top-k indices from forward pass [B, Sq, topk]
        softmax_scale: Scaling factor for attention scores
        loss_coeff: Coefficient for the loss
        sparse_loss: Whether to use sparse loss (only topk positions)
        grad_loss: Gradient from upstream (typically 1.0)
    
    Returns:
        grad_q: Gradient w.r.t. q [Sq, B, H, D]
        grad_weights: Gradient w.r.t. weights [Sq, B, H]
        grad_k: Gradient w.r.t. k [Sk, B, D]
    
    Algorithm:
        1. Recompute index_scores and attention_scores
        2. Apply masks (causal + optional sparse)
        3. Compute softmax for both
        4. Compute gradient of KL divergence w.r.t. index_scores_softmax
        5. Backpropagate through softmax
        6. Backpropagate through index_scores computation (einsum, relu, weights)
    """
    # Recompute index_scores (this is the "unfused" part in backward)
    # Trade-off: extra computation vs memory saving
    index_scores = _compute_index_scores(q, weights, k)  # [B, Sq, Sk]

    sq, b, np, hn = query.size()
    sk = key.size(0)

    # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
    query_reshaped = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
    # [sk, b, np, hn] -> [b, np, hn, sk] -> [b * np, hn, sk]
    key_reshaped = key.permute(1, 2, 3, 0).reshape(b * np, hn, sk)
    # Compute attention scores [b * np, sq, sk]
    attention_scores = torch.bmm(query_reshaped.float(), key_reshaped.float()) * softmax_scale
    # Reshape to [b, np, sq, sk]
    attention_scores = attention_scores.reshape(b, np, sq, sk)

    # causal_mask [sq, sk]
    causal_mask = torch.triu(
        torch.full((sq, sk), float('-inf'), dtype=torch.float32, device=attention_scores.device),
        diagonal=1,
    )
    # index_mask [b, sq, sk]
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=causal_mask.device
    ).scatter_(-1, topk_indices, 0)

    # Apply causal mask to both attention and index scores
    # [b, np, sq, skv] + [1, 1, sq, skv] -> [b, np, sq, skv]
    attention_scores = attention_scores + causal_mask.view(1, 1, sq, sk)
    # [b, sq, sk] + [1, sq, sk] -> [b, sq, sk]  
    index_scores = index_scores + causal_mask.unsqueeze(0)
    
    if sparse_loss:
        # [b, np, sq, sk] + [b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores = attention_scores + index_mask.view(b, 1, sq, sk)
        # [b, sq, sk] + [b, sq, sk] -> [b, sq, sk]
        index_scores = index_scores + index_mask
    
    # Compute softmax for both
    attention_scores_softmax = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32)
    index_scores_softmax = torch.nn.functional.softmax(index_scores, dim=-1, dtype=torch.float32)
    
    # Sum attention scores across heads: [b, np, sq, sk] -> [b, sq, sk]
    attention_scores_sum = attention_scores_softmax.sum(dim=1)
    # L1 normalize
    attention_scores_normalized = attention_scores_sum / attention_scores_sum.sum(dim=-1, keepdim=True)
    
    # # Compute KL per element
    # kl_per_element = attention_scores_normalized * (
    #     torch.log(attention_scores_normalized + 1e-10) - torch.log(index_scores_softmax + 1e-10)
    # )
    
    # Backward through loss = kl_div * loss_coeff
    # where kl_div = kl_per_element.sum(dim=-1).mean()
    grad_kl_div = grad_loss * loss_coeff  # scalar
    
    # Backward through mean: distribute gradient equally
    grad_kl_per_row = grad_kl_div / (b * sq)  # scalar value for each row
    
    # Backward through sum(dim=-1): broadcast back to [b, sq, sk]
    # Each element in a row contributes to the sum, so gradient is same for all
    grad_kl_per_element = torch.full((b, sq, sk), grad_kl_per_row.item(), 
                                      device=index_scores.device, dtype=torch.float32)
    
    # Backward through kl_per_element = target * (log(target) - log(index))
    # ∂kl/∂index_softmax = -target / index_softmax
    grad_index_scores_softmax = -attention_scores_normalized / (index_scores_softmax + 1e-10) * grad_kl_per_element
    
    # Backward through softmax: ∂L/∂x = softmax * (∂L/∂softmax - sum(∂L/∂softmax * softmax))
    sum_grad = (grad_index_scores_softmax * index_scores_softmax).sum(dim=-1, keepdim=True)
    grad_index_scores_logits = index_scores_softmax * (grad_index_scores_softmax - sum_grad)
    
    # Zero out gradients for masked positions
    # Create a mask for valid (non-masked) positions
    # Causal mask: position (i, j) is valid if j <= i
    causal_valid_mask = torch.tril(torch.ones((sq, sk), device=index_scores.device, dtype=torch.bool))  # [sq, sk]
    if sparse_loss:
        # Also apply index mask - only topk positions are valid
        index_valid_mask = (index_mask == 0)  # [b, sq, sk]
        valid_mask = causal_valid_mask.unsqueeze(0) & index_valid_mask  # [b, sq, sk]
    else:
        valid_mask = causal_valid_mask.unsqueeze(0).expand(b, sq, sk)  # [b, sq, sk]
    
    grad_index_scores_logits = grad_index_scores_logits * valid_mask.float()
    
    # Now backward through _compute_index_scores
    # index_scores = einsum('sbhd,tbd->sbht', q, k).relu() * weights -> sum(dim=2) -> transpose
    
    # Transpose from [b, sq, sk] to [sq, b, sk]
    grad_index_scores = grad_index_scores_logits.transpose(0, 1)  # [sq, b, sk]
    
    # Backward through sum over heads: expand gradient
    grad_weighted_scores = grad_index_scores.unsqueeze(2)  # [sq, b, 1, sk]
    
    # Compute forward values needed for backward
    scores = torch.einsum('sbhd,tbd->sbht', q.float(), k.float())  # [sq, b, h, sk]
    scores_after_relu = torch.relu(scores)
    
    # Backward through multiplication by weights: index_scores_per_head * weights
    # ∂L/∂weights = grad * relu_scores (sum over sk)
    grad_weights = (grad_weighted_scores * scores_after_relu).sum(dim=-1)  # [sq, b, h]
    
    # ∂L/∂relu_scores = grad * weights
    grad_scores_after_relu = grad_weighted_scores * weights.unsqueeze(-1)  # [sq, b, h, sk]
    
    # Backward through ReLU
    relu_mask = (scores > 0).float()
    grad_scores = grad_scores_after_relu * relu_mask  # [sq, b, h, sk]
    
    # Backward through einsum 'sbhd,tbd->sbht'
    # ∂L/∂q = einsum('sbht,tbd->sbhd', grad_scores, k)
    grad_q = torch.einsum('sbht,tbd->sbhd', grad_scores, k.float())  # [sq, b, h, d]
    # ∂L/∂k = einsum('sbht,sbhd->tbd', grad_scores, q)
    grad_k = torch.einsum('sbht,sbhd->tbd', grad_scores, q.float())  # [sk, b, d]

    return grad_q.to(q.dtype), grad_weights.to(weights.dtype), grad_k.to(k.dtype)

def backward_autograd(
    q, weights, k, mask, index_topk, query, key, softmax_scale, loss_coeff, sparse_loss, pg_collection
):
    indexer_loss = forward_native(q, weights, k, mask, index_topk, query, key, softmax_scale, loss_coeff, sparse_loss, pg_collection)
    indexer_loss.backward()
    return q.grad, weights.grad, k.grad


def test_loss():
    Sq = 4
    B = 2
    H = 2
    D = 256
    Sk = 16
    topk = 8

    attention_scores = torch.randn(B, H, Sq, Sk, device='cuda', dtype=torch.bfloat16)
    weights = torch.randn(Sq, B, H, device='cuda', dtype=torch.bfloat16)
    mask = torch.triu(
        torch.full((B, Sq, Sk), float('-inf'), dtype=torch.float32, device='cuda'),
        diagonal=1,
    )
    index_scores = torch.randn(B, Sq, Sk, device='cuda', dtype=torch.bfloat16)
    # topk_indices = torch.randint(0, Sk, (B, Sq, topk), device='cuda', dtype=torch.int32)
    query = torch.randn(Sq, B, H, D, device='cuda', dtype=torch.bfloat16)
    key = torch.randn(Sk, B, H, D, device='cuda', dtype=torch.bfloat16)
    softmax_scale = 1.0
    loss_coeff = 1.0
    sparse_loss = False

    # for i in torch.arange(Sk):
    #     attention_scores[:, :, :, i] = torch.log(i + 10) #torch.exp(i) + 10
    #     index_scores[:, :, i] = torch.log(i + 1) #torch.exp(i)

    topk_indices, softmax_mi, softmax_di = _compute_index_scores_topk_native(query, weights, key, mask, topk)

    indexer_loss, kl_per_element = compute_dsa_indexer_loss(
        index_scores, topk_indices, query, key, softmax_scale, loss_coeff, sparse_loss, attention_scores
    )

    indexer_loss_1pass, loss_i_1pass = compute_dsa_indexer_loss_1pass(
        index_scores, topk_indices, query, key, softmax_scale, loss_coeff, sparse_loss, attention_scores
    )

    indexer_loss_triton, kl_per_element_triton = compute_dsa_indexer_loss_triton(
        index_scores, topk_indices, query, key, softmax_scale, loss_coeff, sparse_loss, attention_scores
    )

    assert torch.allclose(indexer_loss, indexer_loss_1pass), "Indexer loss mismatch"

def benchmark_compute_index_scores_topk():
    """Benchmark compute_index_scores_topk + DSA indexer loss: native PyTorch vs Triton."""
    
    print("\n" + "=" * 80)
    print("Benchmark: DSA Indexer (TopK + Loss) - PyTorch vs Triton")
    print("=" * 80)
    
    configs = [
        # (Sq, Sk, B, H, D, topk)
        # (16, 80, 2, 8, 256, 8),
        # (32, 128, 2, 8, 256, 8),
        # (64, 256, 2, 8, 256, 8),
        # (128, 256, 2, 8, 256, 8),
        # (64, 512, 2, 8, 256, 8),
        # (128, 512, 2, 8, 256, 8),
        # (256, 1024, 2, 8, 256, 8),
        # (1024, 2048, 2, 8, 256, 8),
        # (4096, 4096, 2, 8, 256, 8),
        # (8192, 8192, 2, 8, 256, 8),
        # (16384, 16384, 2, 8, 256, 8),

        (2048, 2048, 1, 8, 128, 16),
        (2048, 2048, 1, 8, 128, 32),
        (2048, 2048, 1, 8, 128, 64),
        (2048, 2048, 1, 8, 128, 128),
        (2048, 2048, 1, 8, 128, 256),
        (2048, 2048, 1, 8, 128, 512),
        (2048, 2048, 1, 8, 128, 1024),
        (2048, 2048, 1, 8, 128, 2048),
        (8192, 8192, 1, 8, 128, 2048),
        (16384, 16384, 1, 8, 128, 2048),

        # (4096, 4096, 2, 8, 256, 64),
        # (8192, 8192, 2, 8, 256, 64),
        # (16384, 16384, 2, 8, 256, 64),
    ]
    
    print(f"\n{'Sq':>4} {'Sk':>5} {'B':>3} {'H':>3} {'D':>3} {'K':>4} {'Sparse':>7} | {'PyTorch (ms)':>14} {'Triton (ms)':>13} {'Speedup':>8}")
    print("-" * 95)
    
    for Sq, Sk, B, H, D, topk in configs:
        # Setup
        q = torch.randn(Sq, B, H, D, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(Sk, B, D, device='cuda', dtype=torch.bfloat16)
        weights = torch.randn(Sq, B, H, device='cuda', dtype=torch.bfloat16)
        mask = torch.triu(
            torch.full((B, Sq, Sk), float('-inf'), dtype=torch.float32, device='cuda'),
            diagonal=1,
        )

        attn_query = torch.randn(Sq, B, H, D, device='cuda', dtype=torch.bfloat16)
        attn_key = torch.randn(Sk, B, H, D, device='cuda', dtype=torch.bfloat16)
        softmax_scale = 1.0
        loss_coeff = 1.0
        for sparse_loss in [True]: #[False, True]:
            native_lambda = lambda: forward_native(q, weights, k, mask, topk, attn_query, attn_key, softmax_scale, loss_coeff, sparse_loss)
            triton_lambda = lambda: compute_dsa_indexer_loss_triton(q, weights, k, attn_query, attn_key, topk, softmax_scale, loss_coeff, mask=mask, sparse_loss=sparse_loss)

            # check correctness
            native_topk_indices, native_indexer_loss, native_kl_per_element, native_index_scores = native_lambda()
            triton_topk_indices, triton_indexer_loss, triton_kl_per_element = triton_lambda()

            topk_mask = ~torch.triu(
                torch.full((B, Sq, topk), 1, dtype=torch.bool, device='cuda'),
                diagonal=1,
            )
            # try:
            #     assert torch.allclose(native_topk_indices[topk_mask], triton_topk_indices[topk_mask])
            # except AssertionError as e:
            #     miss_match_indices = (native_topk_indices != triton_topk_indices).nonzero()
            #     # filter causal
            #     miss_match_indices = miss_match_indices[miss_match_indices[:, 1] >= miss_match_indices[:, 2]]
            #     '''
            #     # valify corner case
            #     a = torch.tensor([[1, 1, 1, 0, 0])
            #     a.topk(4)[1]
            #     # randomly returns [1, 2, 0, 3] or [1, 2, 0, 4]
            #     '''
            #     miss_match_native_indices = native_topk_indices[miss_match_indices[:, 0], miss_match_indices[:, 1], miss_match_indices[:, 2]]
            #     miss_match_triton_indices = triton_topk_indices[miss_match_indices[:, 0], miss_match_indices[:, 1], miss_match_indices[:, 2]]
            #     miss_match_native_values = native_index_scores[miss_match_indices[:, 0], miss_match_indices[:, 1], miss_match_native_indices]
            #     miss_match_triton_values = native_index_scores[miss_match_indices[:, 0], miss_match_indices[:, 1], miss_match_triton_indices]
            #     assert torch.allclose(miss_match_native_values, miss_match_triton_values)

            # assert torch.allclose(native_indexer_loss, triton_indexer_loss), "Indexer loss mismatch"
            # # assert torch.allclose(native_kl_per_element, triton_kl_per_element), "KL per element mismatch"

            # Warmup
            for _ in range(5):
                _ = native_lambda()
                _ = triton_lambda()
            torch.cuda.synchronize()
            
            # Benchmark PyTorch
            pytorch_time = triton.testing.do_bench(
                native_lambda
            ) * 1000
            
            # Benchmark Triton
            triton_time = triton.testing.do_bench(
                triton_lambda
            ) * 1000
            
            speedup = pytorch_time / triton_time
            marker = "🚀" if speedup > 1.0 else ""
            sparse_str = "Yes" if sparse_loss else "No"
            
            print(f"{Sq:>4} {Sk:>5} {B:>3} {H:>3} {D:>3} {topk:>4} {sparse_str:>7} | {pytorch_time:>12.2f}   {triton_time:>11.2f}   {speedup:>6.2f}x {marker}")
        
    print("\n" + "=" * 95)

def benchmark_tensor_parallel():
    from tests.unit_tests.test_utilities import Utils
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    tensor_model_parallel_size = 8

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size, pipeline_model_parallel_size=1
    )

    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    triton_pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
    native_pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

    torch.cuda.set_device(dist.get_rank())
    tp_rank = parallel_state.get_tensor_model_parallel_rank()

    """Benchmark compute_index_scores_topk + DSA indexer loss: native PyTorch vs Triton."""
    configs = [
        # (Sq, Sk, B, H, D, topk)
        # (16, 16, 2, 8, 256, 8),
        # (32, 32, 2, 8, 256, 8),
        # (64, 64, 2, 8, 256, 8),
        # (128, 128, 2, 8, 256, 8),
        # (256, 256, 2, 8, 256, 8),
        # (1024, 1024, 2, 8, 256, 8),
        # (4096, 4096, 2, 8, 256, 8),
        # (8192, 8192, 2, 8, 256, 8),
        # (16384, 16384, 2, 8, 256, 8),

        (8192, 8192, 2, 8, 256, 8),
        (8192, 8192, 2, 8, 256, 32),
        (8192, 8192, 2, 8, 256, 64),
        # (8192, 8192, 2, 8, 256, 512),
        # (8192, 8192, 2, 8, 256, 1024),
        # (8192, 8192, 2, 8, 256, 2048),
    ]
    
    if tp_rank == 0:
        metrics_collection = []
    
    for Sq, Sk, B, H, D, topk in configs:
        # Setup
        q = torch.randn(Sq, B, H, D, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(Sk, B, D, device='cuda', dtype=torch.bfloat16)
        weights = torch.randn(Sq, B, H, device='cuda', dtype=torch.bfloat16)
        mask = torch.triu(
            torch.full((B, Sq, Sk), float('-inf'), dtype=torch.float32, device='cuda'),
            diagonal=1,
        )

        attn_query = torch.zeros(Sq, B, H, D, device='cuda', dtype=torch.bfloat16)
        attn_key = torch.zeros(Sk, B, H, D, device='cuda', dtype=torch.bfloat16)
        for _h in range(H):
            attn_query[:, :, _h, :] = _h
            attn_key[:, :, _h, :] = _h

        # split attn heads by TP size
        assert H % tensor_model_parallel_size == 0
        head_per_rank = H // tensor_model_parallel_size
        start_head = tp_rank * head_per_rank
        end_head = (tp_rank + 1) * head_per_rank
        attn_query_tp = attn_query[:, :, start_head:end_head, :].clone()
        attn_key_tp = attn_key[:, :, start_head:end_head, :].clone()

        softmax_scale = 1.0
        loss_coeff = 1.0
        for sparse_loss in [False, True]:
            native_lambda = lambda: forward_native(q, weights, k, mask, topk, attn_query_tp, attn_key_tp, softmax_scale, loss_coeff, sparse_loss, native_pg_collection)
            triton_lambda = lambda: compute_dsa_indexer_loss_triton(q, weights, k, attn_query_tp, attn_key_tp, topk, softmax_scale, loss_coeff, mask=mask, sparse_loss=sparse_loss, pg_collection=triton_pg_collection)

            # check correctness
            native_topk_indices, native_indexer_loss, native_kl_per_element, native_index_scores = native_lambda()
            triton_topk_indices, triton_indexer_loss, triton_kl_per_element = triton_lambda()

            torch.cuda.synchronize()

            topk_mask = ~torch.triu(
                torch.full((B, Sq, topk), 1, dtype=torch.bool, device='cuda'),
                diagonal=1,
            )

            try:
                assert torch.allclose(native_topk_indices[topk_mask], triton_topk_indices[topk_mask])
            except AssertionError as e:
                miss_match_indices = (native_topk_indices != triton_topk_indices).nonzero()
                # filter causal
                miss_match_indices = miss_match_indices[miss_match_indices[:, 1] >= miss_match_indices[:, 2]]
                '''
                # valify corner case
                a = torch.tensor([[1, 1, 1, 0, 0])
                a.topk(4)[1]
                # randomly returns [1, 2, 0, 3] or [1, 2, 0, 4]
                '''
                miss_match_native_indices = native_topk_indices[miss_match_indices[:, 0], miss_match_indices[:, 1], miss_match_indices[:, 2]]
                miss_match_triton_indices = triton_topk_indices[miss_match_indices[:, 0], miss_match_indices[:, 1], miss_match_indices[:, 2]]
                miss_match_native_values = native_index_scores[miss_match_indices[:, 0], miss_match_indices[:, 1], miss_match_native_indices]
                miss_match_triton_values = native_index_scores[miss_match_indices[:, 0], miss_match_indices[:, 1], miss_match_triton_indices]
                assert torch.allclose(miss_match_native_values, miss_match_triton_values), f"{miss_match_native_values=}, {miss_match_triton_values=}"

            torch.cuda.synchronize()
            print(f"[Rank {tp_rank}] {Sq=}, {Sk=}, {B=}, {H=}, {D=}, {topk=} passed.")

            dist.barrier()

            pytorch_time, _, _ = bench(native_lambda)
            pytorch_time *= 1000

            triton_time, _, _ = bench(triton_lambda)
            triton_time *= 1000

            speedup = pytorch_time / triton_time
            marker = "🚀" if speedup > 1.0 else ""
            sparse_str = "Yes" if sparse_loss else "No"

            if tp_rank == 0:
                metrics_collection.append((Sq, Sk, B, H, D, topk, sparse_str, pytorch_time, triton_time, speedup))
        
    dist.barrier()
    if tp_rank == 0:
        print("\n" + "=" * 80)
        print("Benchmark: DSA Indexer (TopK + Loss) - PyTorch vs Triton")
        print("=" * 80)
        print(f"\n{'Sq':>4} {'Sk':>5} {'B':>3} {'H':>3} {'D':>3} {'K':>4} {'Sparse':>7} | {'PyTorch (ms)':>14} {'Triton (ms)':>13} {'Speedup':>8}")
        print("-" * 95)

        for Sq, Sk, B, H, D, topk, sparse_str, pytorch_time, triton_time, speedup in metrics_collection:
            print(f"{Sq:>4} {Sk:>5} {B:>3} {H:>3} {D:>3} {topk:>4} {sparse_str:>7} | {pytorch_time:>12.2f}   {triton_time:>11.2f}   {speedup:>6.2f}x {marker}")
        
        print("\n" + "=" * 95)

    dist.barrier()
    Utils.destroy_model_parallel()


def test_backward_native():
    """
    Test backward_native by comparing with PyTorch autograd.
    
    This test validates the manual backward implementation of the DSA indexer loss
    by comparing gradients with PyTorch's automatic differentiation.
    
    Tests multiple configurations with varying:
    - Sequence lengths (Sq, Sk)
    - Batch sizes (B)
    - Number of heads (H)
    - Head dimensions (D)
    - Top-k values
    - Sparse loss (True/False)
    
    Returns:
        bool: True if all tests pass, False otherwise.
    """
    print("\n" + "=" * 80)
    print("Test: backward_native vs autograd")
    print("=" * 80)
    
    # Test configurations: (Sq, Sk, B, H, D, topk, sparse_loss)
    configs = [
        # (Sq, Sk, B, H, D, topk, sparse_loss)
        (64, 128, 1, 2, 64, 64, False),
        (64, 128, 1, 2, 64, 64, True),
        (128, 256, 2, 4, 128, 128, False),
        (128, 256, 2, 4, 128, 128, True),
    ]
    
    all_passed = True
    
    for config_idx, (Sq, Sk, B, H, D, topk, sparse_loss) in enumerate(configs):
        print(f"\n[{config_idx+1}/{len(configs)}] Testing: Sq={Sq}, Sk={Sk}, B={B}, H={H}, D={D}, topk={topk}, sparse_loss={sparse_loss}")
        
        # Create inputs for autograd path (requires_grad=True)
        torch.manual_seed(42 + config_idx)
        q_autograd = torch.randn(Sq, B, H, D, device='cuda', dtype=torch.float32, requires_grad=True)
        weights_autograd = torch.randn(Sq, B, H, device='cuda', dtype=torch.float32, requires_grad=True)
        k_autograd = torch.randn(Sk, B, D, device='cuda', dtype=torch.float32, requires_grad=True)
        
        # Create mask
        mask = torch.triu(
            torch.full((B, Sq, Sk), float('-inf'), dtype=torch.float32, device='cuda'),
            diagonal=1,
        )
        
        # Create query and key for attention
        query = torch.randn(Sq, B, H, D, device='cuda', dtype=torch.float32)
        key = torch.randn(Sk, B, H, D, device='cuda', dtype=torch.float32)
        
        softmax_scale = 1.0 / (D ** 0.5)
        loss_coeff = 0.1
        
        # Compute forward
        index_scores = _compute_index_scores(q_autograd, weights_autograd, k_autograd)
        if mask is not None:
            index_scores = index_scores + mask
        topk_indices = index_scores.topk(topk, dim=-1)[1]
        
        indexer_loss, kl_per_element = compute_dsa_indexer_loss(
            index_scores.clone(), topk_indices, query, key, softmax_scale, loss_coeff, sparse_loss
        )
        
        # Get autograd gradients
        indexer_loss.backward()
        
        grad_q_autograd = q_autograd.grad.clone() if q_autograd.grad is not None else torch.zeros_like(q_autograd)
        grad_weights_autograd = weights_autograd.grad.clone() if weights_autograd.grad is not None else torch.zeros_like(weights_autograd)
        grad_k_autograd = k_autograd.grad.clone() if k_autograd.grad is not None else torch.zeros_like(k_autograd)
        
        # Create inputs for manual backward (no requires_grad)
        q_manual = q_autograd.detach().clone()
        weights_manual = weights_autograd.detach().clone()
        k_manual = k_autograd.detach().clone()
        
        # Compute manual gradients
        grad_loss = torch.ones_like(indexer_loss)
        grad_q_manual, grad_weights_manual, grad_k_manual = backward_native(
            q_manual, weights_manual, k_manual, query, key, topk_indices,
            softmax_scale, loss_coeff, sparse_loss, grad_loss
        )
        
        # Compare gradients
        rtol = 5e-2
        atol = 1e-4
        
        q_match = torch.allclose(grad_q_autograd, grad_q_manual, rtol=rtol, atol=atol)
        weights_match = torch.allclose(grad_weights_autograd, grad_weights_manual, rtol=rtol, atol=atol)
        k_match = torch.allclose(grad_k_autograd, grad_k_manual, rtol=rtol, atol=atol)
        
        # Print results
        if q_match and weights_match and k_match:
            print(f"  ✓ All gradients match! (loss={indexer_loss.item():.6f})")
        else:
            all_passed = False
            print(f"  ✗ Gradient mismatch detected:")
            if not q_match:
                q_rel_diff = (grad_q_autograd - grad_q_manual).abs() / (grad_q_autograd.abs() + 1e-8)
                print(f"    - grad_q: max_rel_diff={q_rel_diff.max():.6f}, max_abs_diff={(grad_q_autograd - grad_q_manual).abs().max():.6f}")
            if not weights_match:
                w_rel_diff = (grad_weights_autograd - grad_weights_manual).abs() / (grad_weights_autograd.abs() + 1e-8)
                print(f"    - grad_weights: max_rel_diff={w_rel_diff.max():.6f}, max_abs_diff={(grad_weights_autograd - grad_weights_manual).abs().max():.6f}")
            if not k_match:
                k_rel_diff = (grad_k_autograd - grad_k_manual).abs() / (grad_k_autograd.abs() + 1e-8)
                print(f"    - grad_k: max_rel_diff={k_rel_diff.max():.6f}, max_abs_diff={(grad_k_autograd - grad_k_manual).abs().max():.6f}")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All backward_native tests passed!")
    else:
        print("✗ Some backward_native tests failed")
    print("=" * 80)
    
    return all_passed


def test_backward_triton_full():
    """
    Test backward_triton_full (fully-fused) by comparing with PyTorch autograd.
    
    This validates the complete Triton implementation with all three kernels.
    """
    print("\n" + "=" * 80)
    print("Test: backward_triton_full vs autograd")
    print("=" * 80)
    
    # Test configurations
    configs = [
        # (Sq, Sk, B, H, D, topk, sparse_loss)
        (64, 128, 1, 2, 64, 64, False),
        (64, 128, 1, 2, 64, 64, True),
        (128, 256, 2, 4, 128, 128, False),
        (128, 256, 2, 4, 128, 128, True),
    ]
    
    all_passed = True
    
    for config_idx, (Sq, Sk, B, H, D, topk, sparse_loss) in enumerate(configs):
        print(f"\n[{config_idx+1}/{len(configs)}] Testing: Sq={Sq}, Sk={Sk}, B={B}, H={H}, D={D}, topk={topk}, sparse_loss={sparse_loss}")
        
        # Create inputs for autograd path (requires_grad=True)
        torch.manual_seed(42 + config_idx)
        q_autograd = torch.randn(Sq, B, H, D, device='cuda', dtype=torch.float32, requires_grad=True)
        weights_autograd = torch.randn(Sq, B, H, device='cuda', dtype=torch.float32, requires_grad=True)
        k_autograd = torch.randn(Sk, B, D, device='cuda', dtype=torch.float32, requires_grad=True)
        
        # Create mask
        mask = torch.triu(
            torch.full((B, Sq, Sk), float('-inf'), dtype=torch.float32, device='cuda'),
            diagonal=1,
        )
        
        # Create query and key for attention
        query = torch.randn(Sq, B, H, D, device='cuda', dtype=torch.float32)
        key = torch.randn(Sk, B, H, D, device='cuda', dtype=torch.float32)
        
        softmax_scale = 1.0 / (D ** 0.5)
        loss_coeff = 0.1
        
        # Compute forward
        index_scores = _compute_index_scores(q_autograd, weights_autograd, k_autograd)
        if mask is not None:
            index_scores = index_scores + mask
        topk_indices = index_scores.topk(topk, dim=-1)[1]
        
        indexer_loss, kl_per_element = compute_dsa_indexer_loss(
            index_scores.clone(), topk_indices, query, key, softmax_scale, loss_coeff, sparse_loss
        )
        
        # Get autograd gradients
        indexer_loss.backward()
        
        grad_q_autograd = q_autograd.grad.clone()
        grad_weights_autograd = weights_autograd.grad.clone()
        grad_k_autograd = k_autograd.grad.clone()
        
        # Create inputs for Triton backward (no requires_grad)
        q_triton = q_autograd.detach().clone()
        weights_triton = weights_autograd.detach().clone()
        k_triton = k_autograd.detach().clone()
        
        # Compute Triton full gradients
        grad_loss = torch.ones_like(indexer_loss)
        try:
            grad_q_triton, grad_weights_triton, grad_k_triton = backward_triton_full(
                q_triton, weights_triton, k_triton, query, key, topk_indices,
                softmax_scale, loss_coeff, sparse_loss, grad_loss
            )
            
            # Compare gradients
            rtol = 1e-1  # Relaxed tolerance for Triton
            atol = 1e-3
            
            q_match = torch.allclose(grad_q_autograd, grad_q_triton, rtol=rtol, atol=atol)
            weights_match = torch.allclose(grad_weights_autograd, grad_weights_triton, rtol=rtol, atol=atol)
            k_match = torch.allclose(grad_k_autograd, grad_k_triton, rtol=rtol, atol=atol)
            
            if q_match and weights_match and k_match:
                print(f"  ✓ All gradients match! (loss={indexer_loss.item():.6f})")
            else:
                all_passed = False
                print(f"  ✗ Gradient mismatch detected:")
                if not q_match:
                    q_rel_diff = (grad_q_autograd - grad_q_triton).abs() / (grad_q_autograd.abs() + 1e-8)
                    print(f"    - grad_q: max_rel_diff={q_rel_diff.max():.6f}, max_abs_diff={(grad_q_autograd - grad_q_triton).abs().max():.6f}")
                if not weights_match:
                    w_rel_diff = (grad_weights_autograd - grad_weights_triton).abs() / (grad_weights_autograd.abs() + 1e-8)
                    print(f"    - grad_weights: max_rel_diff={w_rel_diff.max():.6f}, max_abs_diff={(grad_weights_autograd - grad_weights_triton).abs().max():.6f}")
                if not k_match:
                    k_rel_diff = (grad_k_autograd - grad_k_triton).abs() / (grad_k_autograd.abs() + 1e-8)
                    print(f"    - grad_k: max_rel_diff={k_rel_diff.max():.6f}, max_abs_diff={(grad_k_autograd - grad_k_triton).abs().max():.6f}")
            
        except Exception as e:
            all_passed = False
            print(f"  ✗ Triton full backward failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All Triton full backward tests passed!")
    else:
        print("✗ Some Triton full backward tests failed")
    print("=" * 80)
    
    return all_passed


def benchmark_backward_only():
    """
    Benchmark ONLY the backward pass: autograd vs backward_triton_full.
    
    This isolates just the backward computation by precomputing all forward values.
    """
    print("\n" + "=" * 80)
    print("Benchmark: Backward Pass ONLY - PyTorch Autograd vs Triton Full")
    print("=" * 80)
    
    # Test configurations: (Sq, Sk, B, H, D, topk, sparse_loss)
    configs = [
        # Small to medium sizes
        (64, 128, 2, 8, 128, 16, False),
        (64, 128, 2, 8, 128, 16, True),
        (128, 256, 2, 8, 128, 32, False),
        (128, 256, 2, 8, 128, 32, True),
        
        # # Larger sizes
        # (256, 512, 2, 8, 128, 64, False),
        # (256, 512, 2, 8, 128, 64, True),
        # (512, 1024, 2, 8, 128, 128, False),
        # (512, 1024, 2, 8, 128, 128, True),
        
        # Very large sizes
        (1024, 2048, 2, 8, 128, 256, False),
        (1024, 2048, 2, 8, 128, 256, True),
        (2048, 4096, 2, 8, 128, 512, False),
        (2048, 4096, 2, 8, 128, 512, True),

        # Huge sizes
        (4096, 4096, 2, 8, 128, 2048, False),
        (4096, 4096, 2, 8, 128, 2048, True),
        (8192, 8192, 2, 8, 128, 2048, False),
        (8192, 8192, 2, 8, 128, 2048, True),
        (16384, 16384, 1, 8, 128, 2048, False),
        (16384, 16384, 1, 8, 128, 2048, True),
    ]
    
    print(f"\n{'Sq':>4} {'Sk':>5} {'B':>3} {'H':>3} {'D':>3} {'K':>4} {'Sparse':>7} | {'PyTorch (ms)':>14} {'Triton (ms)':>13} {'Speedup':>8}")
    print("-" * 95)
    
    for Sq, Sk, B, H, D, topk, sparse_loss in configs:
        torch.manual_seed(42)
        
        # Create inputs
        q = torch.randn(Sq, B, H, D, device='cuda', dtype=torch.float32)
        weights = torch.randn(Sq, B, H, device='cuda', dtype=torch.float32)
        k = torch.randn(Sk, B, D, device='cuda', dtype=torch.float32)
        query = torch.randn(Sq, B, H, D, device='cuda', dtype=torch.float32)
        key = torch.randn(Sk, B, H, D, device='cuda', dtype=torch.float32)
        
        mask = torch.triu(
            torch.full((B, Sq, Sk), float('-inf'), dtype=torch.float32, device='cuda'),
            diagonal=1,
        )
        
        softmax_scale = 1.0 / (D ** 0.5)
        loss_coeff = 0.1
        
        # Precompute forward pass
        with torch.no_grad():
            index_scores = _compute_index_scores(q, weights, k)
            index_scores_masked = index_scores + mask
            topk_indices = index_scores_masked.topk(topk, dim=-1)[1]
            indexer_loss, _ = compute_dsa_indexer_loss(
                index_scores_masked.clone(), topk_indices, query, key,
                softmax_scale, loss_coeff, sparse_loss
            )
            grad_loss = torch.ones_like(indexer_loss)
        
        # Benchmark PyTorch autograd backward
        def pytorch_backward_only():
            grad_q, grad_weights, grad_k = backward_native(
                q, weights, k, query, key, topk_indices,
                softmax_scale, loss_coeff, sparse_loss, grad_loss
            )
        
        # Benchmark Triton backward only
        def triton_backward_only():
            grad_q, grad_weights, grad_k = backward_triton_full(
                q, weights, k, query, key, topk_indices,
                softmax_scale, loss_coeff, sparse_loss, grad_loss
            )
            torch.cuda.synchronize()
        
        # Warmup
        for _ in range(5):
            pytorch_backward_only()
            triton_backward_only()
        torch.cuda.synchronize()
        
        # Benchmark
        pytorch_time = triton.testing.do_bench(pytorch_backward_only) * 1000
        triton_time = triton.testing.do_bench(triton_backward_only) * 1000
        
        speedup = pytorch_time / triton_time
        marker = "🚀" if speedup > 1.0 else "⚠️"
        sparse_str = "Yes" if sparse_loss else "No"
        
        print(f"{Sq:>4} {Sk:>5} {B:>3} {H:>3} {D:>3} {topk:>4} {sparse_str:>7} | {pytorch_time:>12.2f}   {triton_time:>11.2f}   {speedup:>6.2f}x {marker}")
    
    print("\n" + "=" * 95)
    print("Note: PyTorch time includes forward recomputation (required for autograd)")
    print("      Triton time is pure backward pass (no forward recomputation)")
    print("=" * 95)


if __name__ == "__main__":
    # # Test the backward_native implementation
    # test_backward_native()
    
    # # Test the fully-fused Triton backward implementation
    # print("\n")
    test_backward_triton_full()
    
    # print("\n")
    # benchmark_backward_only()
    
    # Uncomment to run other benchmarks:
    # benchmark_compute_index_scores_topk()
    # benchmark_tensor_parallel()
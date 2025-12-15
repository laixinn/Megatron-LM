# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.distributed as dist

import triton
import triton.language as tl

from megatron.core.process_groups_config import ProcessGroupCollection


# =============================================================================
# _compute_index_scores Triton kernel (full output)
# =============================================================================

@triton.jit
def _compute_index_scores_kernel(
    Q_ptr,
    K_ptr,
    W_ptr,
    Mask_ptr,
    Out_ptr,
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
    # Mask strides: [B, Sq, Sk]
    stride_mb,
    stride_ms,
    stride_mk,
    # Out strides: [B, Sq, Sk]
    stride_ob,
    stride_os,
    stride_ok,
    # Dimensions
    H,
    D,
    Sq,
    Sk,
    BLOCK_SK: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    """
    Computes: out[b, sq, sk] = sum_h(relu(Q[sq,b,h,:] · K[sk,b,:]) * W[sq,b,h]) + mask[b, sq, sk]
    
    Grid: (B * Sq, ceil(Sk / BLOCK_SK))
    
    This fuses:
      1. Q @ K^T (batched dot products)
      2. ReLU activation
      3. Weighting by W
      4. Sum over heads
      5. Mask addition (optional)
    
    Avoiding materialization of the [Sq, B, H, Sk] intermediate tensor.
    """
    # Program IDs
    pid_seq = tl.program_id(0)  # Encodes (b, sq)
    pid_sk = tl.program_id(1)   # Sk chunk index
    
    # Decompose pid_seq -> (b, sq)
    sq = pid_seq % Sq
    b = pid_seq // Sq
    
    # Sk chunk offsets
    sk_start = pid_sk * BLOCK_SK
    sk_offs = sk_start + tl.arange(0, BLOCK_SK)
    sk_valid = sk_offs < Sk
    
    # Accumulator for output [BLOCK_SK]
    acc = tl.zeros([BLOCK_SK], dtype=tl.float32)
    
    # Base pointers for this (b, sq)
    q_base = Q_ptr + sq * stride_qs + b * stride_qb
    k_base = K_ptr + b * stride_kb
    w_base = W_ptr + sq * stride_ws + b * stride_wb
    
    # Loop over heads
    for h in range(H):
        # Load weight scalar W[sq, b, h]
        w_val = tl.load(w_base + h * stride_wh)
        
        # Q base for this head
        q_head_base = q_base + h * stride_qh
        
        # Compute dot product: Q[sq,b,h,:] · K[sk_offs,b,:]
        dot = tl.zeros([BLOCK_SK], dtype=tl.float32)
        
        # Loop over D dimension
        # Each iteration: load Q[d] (scalar), K[sk_offs, d] (vector), accumulate
        for d in range(D):
            # Load Q[sq, b, h, d] - scalar
            q_val = tl.load(q_head_base + d * stride_qd)
            
            # Load K[sk_offs, b, d] - vector [BLOCK_SK]
            k_ptrs = k_base + sk_offs * stride_ks + d * stride_kd
            k_vals = tl.load(k_ptrs, mask=sk_valid, other=0.0)
            
            # Accumulate: dot[i] += Q[d] * K[i, d]
            dot += q_val.to(tl.float32) * k_vals.to(tl.float32)
        
        # Apply ReLU
        dot = tl.maximum(dot, 0.0)
        
        # Weight and accumulate across heads
        acc += dot * w_val.to(tl.float32)
    
    # Add mask if provided
    if HAS_MASK:
        mask_ptrs = Mask_ptr + b * stride_mb + sq * stride_ms + sk_offs * stride_mk
        mask_vals = tl.load(mask_ptrs, mask=sk_valid, other=0.0)
        acc += mask_vals
    
    # Store output [BLOCK_SK]
    out_ptrs = Out_ptr + b * stride_ob + sq * stride_os + sk_offs * stride_ok
    tl.store(out_ptrs, acc, mask=sk_valid)


def compute_index_scores_triton(q: torch.Tensor, weights: torch.Tensor, k: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute index scores using Triton kernel.
    
    This is a fused implementation that avoids materializing the 
    intermediate [Sq, B, H, Sk] tensor.
    
    Formula: out[b, sq, sk] = sum_h(relu(Q[sq,b,h,:] · K[sk,b,:]) * W[sq,b,h]) + mask[b, sq, sk]
    
    Args:
        q: [Sq, B, H, D] query tensor
        weights: [Sq, B, H] attention weights
        k: [Sk, B, D] key tensor
        mask: [B, Sq, Sk] mask tensor (optional)
        
    Returns:
        index_scores: [B, Sq, Sk] float32 tensor
    """
    Sq, B, H, D = q.shape
    Sk = k.shape[0]
    
    # Output tensor
    out = torch.empty((B, Sq, Sk), dtype=torch.float32, device=q.device)
    
    # Block size for Sk dimension
    BLOCK_SK = 128
    
    # Grid: parallelize over (B * Sq) and Sk chunks
    grid = (B * Sq, triton.cdiv(Sk, BLOCK_SK))
    
    # Handle mask strides
    if mask is not None:
        stride_mb = mask.stride(0)
        stride_ms = mask.stride(1)
        stride_mk = mask.stride(2)
        has_mask = True
    else:
        stride_mb = stride_ms = stride_mk = 0
        has_mask = False
    
    _compute_index_scores_kernel[grid](
        Q_ptr=q,
        K_ptr=k,
        W_ptr=weights,
        Mask_ptr=mask,
        Out_ptr=out,
        # Q strides
        stride_qs=q.stride(0),
        stride_qb=q.stride(1),
        stride_qh=q.stride(2),
        stride_qd=q.stride(3),
        # K strides
        stride_ks=k.stride(0),
        stride_kb=k.stride(1),
        stride_kd=k.stride(2),
        # W strides
        stride_ws=weights.stride(0),
        stride_wb=weights.stride(1),
        stride_wh=weights.stride(2),
        # Mask strides
        stride_mb=stride_mb,
        stride_ms=stride_ms,
        stride_mk=stride_mk,
        # Out strides
        stride_ob=out.stride(0),
        stride_os=out.stride(1),
        stride_ok=out.stride(2),
        # Dimensions
        H=H,
        D=D,
        Sq=Sq,
        Sk=Sk,
        BLOCK_SK=BLOCK_SK,
        HAS_MASK=has_mask,
    )
    
    return out


# =============================================================================
# Fused _compute_index_scores + TopK kernel (avoids [B, Sq, Sk] materialization)
# =============================================================================
@triton.jit
def topk_parallel_kernel(
    x_ptr,
    out_val_ptr,
    out_idx_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
):
    """
    Optimized TopK with mask-based processing.
    Uses a processed mask to avoid repeated max finding.
    
    Grid: (1,)
    """
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    vals = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))
    
    # Running TopK buffer
    topk_vals = tl.full([TOPK], float("-inf"), dtype=tl.float32)
    topk_idxs = tl.full([TOPK], -1, dtype=tl.int32)
    
    # Use mask to track processed elements
    vals_working = vals
    
    # Iterate TOPK times
    for k in tl.static_range(TOPK):
        # Find max
        max_val = tl.max(vals_working, axis=0)
        
        # Find argmax
        is_max = (vals_working == max_val)
        argmax = tl.min(tl.where(is_max, offsets, BLOCK_SIZE), axis=0)
        
        # Store result
        topk_vals = tl.where(tl.arange(0, TOPK) == k, max_val, topk_vals)
        topk_idxs = tl.where(tl.arange(0, TOPK) == k, argmax.to(tl.int32), topk_idxs)
        
        # Mask out selected element
        vals_working = tl.where(offsets == argmax, float("-inf"), vals_working)
    
    tl.store(out_val_ptr + tl.arange(0, TOPK), topk_vals)
    tl.store(out_idx_ptr + tl.arange(0, TOPK), topk_idxs)


@triton.jit
def _compute_index_scores_topk_kernel(
    Q_ptr,
    K_ptr,
    W_ptr,
    Mask_ptr,
    Out_Idx_ptr,
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
    # Mask strides: [B, Sq, Sk]
    stride_mb,
    stride_ms,
    stride_mk,
    # Out strides: [B, Sq, TopK]
    stride_ib,
    stride_is,
    stride_ik,
    # Dimensions
    H,
    D,
    Sq,
    Sk,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TOPK: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    """
    Fused index score computation + TopK selection.
    
    Grid: (B * Sq,) - one thread block per (b, sq) pair
    
    For each (b, sq):
      1. Iterate over Sk in chunks
      2. Compute scores for each chunk
      3. Maintain running TopK in registers
      4. Output TopK indices
    
    This avoids materializing the full [B, Sq, Sk] tensor.
    """
    b = tl.program_id(0)
    sq_block_id = tl.program_id(1)
    sq = sq_block_id * BLOCK_SQ + tl.arange(0, BLOCK_SQ)
    sq_valid = sq < Sq
    
    # Initialize TopK buffers in registers
    # topk_vals[i] holds the i-th largest value seen so far
    # topk_idxs[i] holds the corresponding index
    topk_vals = tl.full([BLOCK_SQ, TOPK], float("-inf"), dtype=tl.float32)
    topk_idxs = tl.full([BLOCK_SQ, TOPK], -1, dtype=tl.int32)
    
    # Base pointers for this (b, sq)
    q_base = Q_ptr + b * stride_qb
    k_base = K_ptr + b * stride_kb
    w_base = W_ptr + sq * stride_ws + b * stride_wb

    # Loop over Sk in chunks
    for sk_start in range(0, Sk, BLOCK_SK):
        sk_offs = sk_start + tl.arange(0, BLOCK_SK)
        sk_valid = sk_offs < Sk
        
        # Compute scores for this chunk
        scores = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
        
        for h in range(H):
            w_val = tl.load(w_base + h * stride_wh, mask=sq_valid, other=0.0)
            q_head_base = q_base + h * stride_qh
            
            dot = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            
            # Process D dimension in blocks for better memory access
            for d_start in range(0, D, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < D
                
                # Load Q values for this D block
                q_ptrs = q_head_base + sq[:, None] * stride_qs + d_offs[None, :] * stride_qd
                q_vals = tl.load(q_ptrs, mask=(sq_valid[:, None] & d_valid[None, :]), other=0.0)

                # Load K values for this D block
                k_ptrs = k_base + sk_offs[None, :] * stride_ks + d_offs[:, None] * stride_kd
                k_vals = tl.load(k_ptrs, mask=(sk_valid[None, :] & d_valid[None, :]), other=0.0)

                # Compute dot product for this D block
                dot += tl.dot(q_vals, k_vals)
            
            # ReLU
            dot = tl.maximum(dot, 0.0)
            scores += dot * w_val[:, None]
        
        # Add mask if provided
        if HAS_MASK:
            mask_ptrs = Mask_ptr + b * stride_mb + sq[:, None] * stride_ms + sk_offs[None, :] * stride_mk
            mask_vals = tl.load(mask_ptrs, mask=(sq_valid[:, None] & sk_valid[None, :]), other=float("-inf"))
            scores = tl.where((sq_valid[:, None] & sk_valid[None, :]), scores + mask_vals, float("-inf"))
        else:
            scores = tl.where((sq_valid[:, None] & sk_valid[None, :]), scores, float("-inf"))

        '''streaming topk'''
        new_topk_vals = tl.full([BLOCK_SQ, TOPK], float("-inf"), dtype=tl.float32)
        new_topk_idxs = tl.full([BLOCK_SQ, TOPK], -1, dtype=tl.int32)
        
        topk_vals_working = topk_vals
        chunk_vals_working = scores
        
        for k in tl.static_range(TOPK):
            # Find max from both topk buffer and current chunk
            max_from_topk = tl.max(topk_vals_working, axis=1)
            max_from_chunk = tl.max(chunk_vals_working, axis=1)
            
            # Determine which has the larger max
            from_topk = (max_from_topk >= max_from_chunk)
            max_val = tl.where(from_topk, max_from_topk, max_from_chunk)
            
            # Find argmax from topk buffer
            is_max_topk = (topk_vals_working == max_from_topk[:, None])
            argmax_topk = tl.min(tl.where(is_max_topk, topk_idxs, Sk), axis=1)
            
            # Find argmax from chunk
            is_max_chunk = (chunk_vals_working == max_from_chunk[:, None])
            argmax_chunk = tl.min(tl.where(is_max_chunk, sk_offs, Sk), axis=1)
            
            # Select index based on which source we chose
            # For topk source, we need to get the actual stored index
            max_idx = tl.where(from_topk, argmax_topk, argmax_chunk)
            
            # Mask out from both sources (only one will actually affect the next iteration)
            topk_vals_working = tl.where(max_val[:, None] == topk_vals_working, float("-inf"), topk_vals_working)
            chunk_vals_working = tl.where(max_val[:, None] == chunk_vals_working, float("-inf"), chunk_vals_working)
            
            # Store in new topk
            new_topk_vals = tl.where(tl.arange(0, TOPK)[None, :] == k, max_val[:, None], new_topk_vals)
            new_topk_idxs = tl.where(tl.arange(0, TOPK)[None, :] == k, max_idx[:, None], new_topk_idxs)

        # Update global topk
        topk_vals = new_topk_vals
        topk_idxs = new_topk_idxs
    
    # Store TopK indices
    idx_base = Out_Idx_ptr + b * stride_ib + sq[:, None] * stride_is + tl.arange(0, TOPK)[None, :] * stride_ik
    tl.store(idx_base, topk_idxs, mask=(sq_valid[:, None] & (tl.arange(0, TOPK)[None, :] < TOPK)))


def compute_index_scores_topk_triton(
    q: torch.Tensor,
    weights: torch.Tensor, 
    k: torch.Tensor,
    topk: int,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused index score computation + TopK selection using Triton.
    
    This avoids materializing the full [B, Sq, Sk] index_scores tensor.
    
    Args:
        q: [Sq, B, H, D] query tensor
        weights: [Sq, B, H] attention weights  
        k: [Sk, B, D] key tensor
        topk: number of top indices to return
        mask: [B, Sq, Sk] mask tensor (optional)
        
    Returns:
        topk_indices: [B, Sq, TopK] int64 tensor of top-k indices
    """
    Sq, B, H, D = q.shape
    Sk = k.shape[0]
    
    # Clamp topk to valid range
    topk = min(topk, Sk)
    
    # Output tensor
    out_idx = torch.empty((B, Sq, topk), dtype=torch.int64, device=q.device)
    
    BLOCK_SQ = 16
    BLOCK_SK = 128
    BLOCK_D = 128

    num_sq_blocks = (Sq + BLOCK_SQ - 1) // BLOCK_SQ
    grid = (B, num_sq_blocks,)
    
    # Handle mask strides
    if mask is not None:
        stride_mb = mask.stride(0)
        stride_ms = mask.stride(1)
        stride_mk = mask.stride(2)
        has_mask = True
    else:
        stride_mb = stride_ms = stride_mk = 0
        has_mask = False
    
    _compute_index_scores_topk_kernel[grid](
        Q_ptr=q,
        K_ptr=k,
        W_ptr=weights,
        Mask_ptr=mask,
        Out_Idx_ptr=out_idx,
        # Q strides
        stride_qs=q.stride(0),
        stride_qb=q.stride(1),
        stride_qh=q.stride(2),
        stride_qd=q.stride(3),
        # K strides
        stride_ks=k.stride(0),
        stride_kb=k.stride(1),
        stride_kd=k.stride(2),
        # W strides
        stride_ws=weights.stride(0),
        stride_wb=weights.stride(1),
        stride_wh=weights.stride(2),
        # Mask strides
        stride_mb=stride_mb,
        stride_ms=stride_ms,
        stride_mk=stride_mk,
        # Out strides
        stride_ib=out_idx.stride(0),
        stride_is=out_idx.stride(1),
        stride_ik=out_idx.stride(2),
        # Dimensions
        H=H,
        D=D,
        Sq=Sq,
        Sk=Sk,
        BLOCK_SQ=BLOCK_SQ,
        BLOCK_SK=BLOCK_SK,
        BLOCK_D=BLOCK_D,
        TOPK=topk,
        HAS_MASK=has_mask,
    )
    
    return out_idx


@triton.jit
def _compute_dsa_indexer_loss_kernel(
    Q_ptr,
    K_ptr,
    W_ptr,
    Mask_ptr,
    Out_Idx_ptr,
    Attn_Query_ptr,
    Attn_Key_ptr,
    Loss_ptr,
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
    # Mask strides: [B, Sq, Sk]
    stride_mb,
    stride_ms,
    stride_mk,
    # Out strides: [B, Sq, TopK]
    stride_ib,
    stride_is,
    stride_ik,
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
    # Dimensions
    H: tl.constexpr,
    D: tl.constexpr,
    AH: tl.constexpr,
    AD: tl.constexpr,
    Sq: tl.constexpr,
    Sk: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TOPK: tl.constexpr,
    HAS_MASK: tl.constexpr,
    SPARSE_LOSS: tl.constexpr,
    Softmax_Scale: tl.constexpr,
):
    """
    Fused index score computation + TopK selection.
    
    Grid: (B * Sq,) - one thread block per (b, sq) pair
    
    For each (b, sq):
      1. Iterate over Sk in chunks
      2. Compute scores for each chunk
      3. Maintain running TopK in registers
      4. Output TopK indices
    
    This avoids materializing the full [B, Sq, Sk] tensor.
    """
    b = tl.program_id(0)
    sq_block_id = tl.program_id(1)
    sq = sq_block_id * BLOCK_SQ + tl.arange(0, BLOCK_SQ)
    sq_valid = sq < Sq
    
    # Initialize TopK buffers in registers
    # topk_vals[i] holds the i-th largest value seen so far
    # topk_idxs[i] holds the corresponding index
    topk_vals = tl.full([BLOCK_SQ, TOPK], float("-inf"), dtype=tl.float32)
    topk_idxs = tl.full([BLOCK_SQ, TOPK], -1, dtype=tl.int32)
    
    # Base pointers for this (b, sq)
    q_base = Q_ptr + b * stride_qb
    k_base = K_ptr + b * stride_kb
    w_base = W_ptr + sq * stride_ws + b * stride_wb

    # TODO: consider TP rank offset
    aq_base = Attn_Query_ptr + b * stride_aqb
    ak_base = Attn_Key_ptr + b * stride_akb

    # 1-pass loss recursion
    m_i = tl.full([AH, BLOCK_SQ], float("-inf"), dtype=tl.float32)
    m1_i = tl.full([BLOCK_SQ], float("-inf"), dtype=tl.float32)
    d_i = tl.zeros([AH, BLOCK_SQ], dtype=tl.float32)
    d1_i = tl.zeros([BLOCK_SQ], dtype=tl.float32)
    loss_i = tl.zeros([BLOCK_SQ], dtype=tl.float32)

    # compute topk indices
    for sk_start in tl.range(0, Sk, BLOCK_SK):
        sk_offs = sk_start + tl.arange(0, BLOCK_SK)
        sk_valid = sk_offs < Sk
        
        '''
        compute index scores
        '''
        # Compute index_scores for this chunk
        index_scores = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
        
        for h in tl.range(H):
            w_val = tl.load(w_base + h * stride_wh, mask=sq_valid, other=0.0)
            q_head_base = q_base + h * stride_qh
            
            dot = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            
            # Process D dimension in blocks for better memory access
            for d_start in tl.range(0, D, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < D
                
                # Load Q values for this D block
                q_ptrs = q_head_base + sq[:, None] * stride_qs + d_offs[None, :] * stride_qd
                q_vals = tl.load(q_ptrs, mask=(sq_valid[:, None] & d_valid[None, :]), other=0.0)

                # Load K values for this D block
                k_ptrs = k_base + sk_offs[None, :] * stride_ks + d_offs[:, None] * stride_kd
                k_vals = tl.load(k_ptrs, mask=(sk_valid[None, :] & d_valid[None, :]), other=0.0)

                # Compute dot product for this D block
                dot += tl.dot(q_vals, k_vals)
            
            # ReLU
            dot = tl.maximum(dot, 0.0)
            index_scores += dot * w_val[:, None]
        
        if HAS_MASK:
            mask_ptrs = Mask_ptr + b * stride_mb + sq[:, None] * stride_ms + sk_offs[None, :] * stride_mk
            mask_vals = tl.load(mask_ptrs, mask=(sq_valid[:, None] & sk_valid[None, :]), other=float("-inf"))
            index_scores = tl.where((sq_valid[:, None] & sk_valid[None, :]), index_scores + mask_vals, float("-inf"))
        else:
            index_scores = tl.where((sq_valid[:, None] & sk_valid[None, :]), index_scores, float("-inf"))

        '''streaming topk'''
        new_topk_vals = tl.full([BLOCK_SQ, TOPK], float("-inf"), dtype=tl.float32)
        new_topk_idxs = tl.full([BLOCK_SQ, TOPK], -1, dtype=tl.int32)
        
        topk_vals_working = topk_vals
        chunk_vals_working = index_scores
        
        for k in tl.range(TOPK):
            # Find max from both topk buffer and current chunk
            max_from_topk = tl.max(topk_vals_working, axis=1)
            max_from_chunk = tl.max(chunk_vals_working, axis=1)
            
            # Determine which has the larger max
            from_topk = (max_from_topk >= max_from_chunk)
            max_val = tl.where(from_topk, max_from_topk, max_from_chunk)
            
            # Find argmax from topk buffer
            is_max_topk = (topk_vals_working == max_from_topk[:, None])
            argmax_topk = tl.min(tl.where(is_max_topk, topk_idxs, Sk), axis=1)
            # argmax_topk = tl.max(tl.where(is_max_topk, topk_idxs, -1), axis=1)

            # Find argmax from chunk
            is_max_chunk = (chunk_vals_working == max_from_chunk[:, None])
            argmax_chunk = tl.min(tl.where(is_max_chunk, sk_offs, Sk), axis=1)
            # argmax_chunk = tl.max(tl.where(is_max_chunk, sk_offs, -1), axis=1)
            
            # Select index based on which source we chose
            # For topk source, we need to get the actual stored index
            max_idx = tl.where(from_topk, argmax_topk, argmax_chunk)
            
            # Mask out from both sources (only one will actually affect the next iteration)
            topk_vals_working = tl.where((max_val[:, None] == topk_vals_working) & (argmax_topk[:, None] == topk_idxs) & (from_topk[:, None]), float("-inf"), topk_vals_working)
            chunk_vals_working = tl.where((max_val[:, None] == chunk_vals_working) & (argmax_chunk[:, None] == sk_offs) & (~from_topk[:, None]), float("-inf"), chunk_vals_working)
            
            # Store in new topk
            new_topk_vals = tl.where(tl.arange(0, TOPK)[None, :] == k, max_val[:, None], new_topk_vals)
            new_topk_idxs = tl.where(tl.arange(0, TOPK)[None, :] == k, max_idx[:, None], new_topk_idxs)

        # Update global topk
        topk_vals = new_topk_vals
        topk_idxs = new_topk_idxs

    # Store TopK indices
    idx_base = Out_Idx_ptr + b * stride_ib + sq[:, None] * stride_is + tl.arange(0, TOPK)[None, :] * stride_ik
    tl.store(idx_base, topk_idxs, mask=(sq_valid[:, None] & (tl.arange(0, TOPK)[None, :] < TOPK)))

    # compute the first pass for attn softmax and index softmax
    # apply causal mask by loop trunctation
    causal_sk = tl.minimum(tl.min(sq) + 1, Sk)
    for sk_start in tl.range(0, causal_sk, BLOCK_SK):
        sk_offs = sk_start + tl.arange(0, BLOCK_SK)
        sk_valid = sk_offs < Sk

        # Compute index_scores for this chunk
        index_scores = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
        
        for h in tl.range(H):
            w_val = tl.load(w_base + h * stride_wh, mask=sq_valid, other=0.0)
            q_head_base = q_base + h * stride_qh
            
            dot = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            
            # Process D dimension in blocks for better memory access
            for d_start in tl.range(0, D, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < D
                
                # Load Q values for this D block
                q_ptrs = q_head_base + sq[:, None] * stride_qs + d_offs[None, :] * stride_qd
                q_vals = tl.load(q_ptrs, mask=(sq_valid[:, None] & d_valid[None, :]), other=0.0)

                # Load K values for this D block
                k_ptrs = k_base + sk_offs[None, :] * stride_ks + d_offs[:, None] * stride_kd
                k_vals = tl.load(k_ptrs, mask=(sk_valid[None, :] & d_valid[None, :]), other=0.0)

                # Compute dot product for this D block
                dot += tl.dot(q_vals, k_vals)
            
            # ReLU
            dot = tl.maximum(dot, 0.0)
            index_scores += dot * w_val[:, None]
        
        if HAS_MASK:
            mask_ptrs = Mask_ptr + b * stride_mb + sq[:, None] * stride_ms + sk_offs[None, :] * stride_mk
            mask_vals = tl.load(mask_ptrs, mask=(sq_valid[:, None] & sk_valid[None, :]), other=float("-inf"))
            index_scores = tl.where((sq_valid[:, None] & sk_valid[None, :]), index_scores + mask_vals, float("-inf"))
        else:
            index_scores = tl.where((sq_valid[:, None] & sk_valid[None, :]), index_scores, float("-inf"))

        if SPARSE_LOSS:
            # [BLOCK_SQ, TOPK, 1] vs [1, 1, BLOCK_SK] -> [BLOCK_SQ, TOPK, BLOCK_SK]
            is_in_topk = (topk_idxs[:, :, None] == sk_offs[None, None, :])
            # [BLOCK_SQ, TOPK, BLOCK_SK] -> [BLOCK_SQ, BLOCK_SK]
            is_in_topk = tl.sum(is_in_topk, axis=1) > 0

            sparse_mask = tl.where(is_in_topk, 0.0, float("-inf"))
            index_scores += sparse_mask
        
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

                aq_ptrs = aq_head_base + sq[:, None] * stride_asq + d_offs[None, :] * stride_aqd
                ak_ptrs = ak_head_base + sk_offs[None, :] * stride_ask + d_offs[:, None] * stride_akd

                aq_vals = tl.load(aq_ptrs, mask=(sq_valid[:, None] & d_valid[None, :]), other=0.0)
                ak_vals = tl.load(ak_ptrs, mask=(sk_valid[None, :] & d_valid[:, None]), other=0.0)

                dot += tl.dot(aq_vals, ak_vals)

            dot *= Softmax_Scale

            attn_scores = tl.where(h_ids[:, None, None] == h, dot[None, :, :], attn_scores)

        # apply causal mask
        casual_mask = tl.full([BLOCK_SQ, BLOCK_SK], float("-inf"), dtype=tl.float32)
        casual_mask = tl.where((sq[:, None] < sk_offs[None, :]), casual_mask, 0.0)
        attn_scores += casual_mask[None, :, :]

        if SPARSE_LOSS:
            attn_scores += sparse_mask[None, :, :]

        m_i_1 = m_i
        m_i = tl.maximum(m_i, tl.max(attn_scores, axis=-1))
        m_i = tl.where(m_i <= float("-inf"), 0.0, m_i)
        d_i = d_i * tl.exp(m_i_1 - m_i) + tl.exp(attn_scores - m_i[:, :, None]).sum(axis=-1)
    
    # recompute for the second pass of attn softmax
    for sk_start in tl.range(0, causal_sk, BLOCK_SK):
        sk_offs = sk_start + tl.arange(0, BLOCK_SK)
        sk_valid = sk_offs < Sk
        
        '''
        compute index scores
        '''
        # Compute index_scores for this chunk
        index_scores = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
        
        for h in tl.range(H):
            w_val = tl.load(w_base + h * stride_wh, mask=sq_valid, other=0.0)
            q_head_base = q_base + h * stride_qh
            
            dot = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            
            # Process D dimension in blocks for better memory access
            for d_start in tl.range(0, D, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < D
                
                # Load Q values for this D block
                q_ptrs = q_head_base + sq[:, None] * stride_qs + d_offs[None, :] * stride_qd
                q_vals = tl.load(q_ptrs, mask=(sq_valid[:, None] & d_valid[None, :]), other=0.0)

                # Load K values for this D block
                k_ptrs = k_base + sk_offs[None, :] * stride_ks + d_offs[:, None] * stride_kd
                k_vals = tl.load(k_ptrs, mask=(sk_valid[None, :] & d_valid[None, :]), other=0.0)

                # Compute dot product for this D block
                dot += tl.dot(q_vals, k_vals)
            
            # ReLU
            dot = tl.maximum(dot, 0.0)
            index_scores += dot * w_val[:, None]
        
        if HAS_MASK:
            mask_ptrs = Mask_ptr + b * stride_mb + sq[:, None] * stride_ms + sk_offs[None, :] * stride_mk
            mask_vals = tl.load(mask_ptrs, mask=(sq_valid[:, None] & sk_valid[None, :]), other=float("-inf"))
            index_scores = tl.where((sq_valid[:, None] & sk_valid[None, :]), index_scores + mask_vals, float("-inf"))
        else:
            index_scores = tl.where((sq_valid[:, None] & sk_valid[None, :]), index_scores, float("-inf"))

        if SPARSE_LOSS:
            # [BLOCK_SQ, TOPK, 1] vs [1, 1, BLOCK_SK] -> [BLOCK_SQ, TOPK, BLOCK_SK]
            is_in_topk = (topk_idxs[:, :, None] == sk_offs[None, None, :])
            # [BLOCK_SQ, TOPK, BLOCK_SK] -> [BLOCK_SQ, BLOCK_SK]
            is_in_topk = tl.sum(is_in_topk, axis=1) > 0

            sparse_mask = tl.where(is_in_topk, 0.0, float("-inf"))
            index_scores += sparse_mask

        '''
        compute loss - online softmax with head summation and L1 normalization
        '''
        h_ids = tl.arange(0, AH)

        attn_scores = tl.zeros([AH, BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
        for h in tl.range(AH):
            aq_head_base = aq_base + h * stride_aqh
            ak_head_base = ak_base + h * stride_akh

            dot = tl.zeros([BLOCK_SQ, BLOCK_SK], dtype=tl.float32)
            for d_start in tl.range(0, AD, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_valid = d_offs < AD

                aq_ptrs = aq_head_base + sq[:, None] * stride_asq + d_offs[None, :] * stride_aqd
                ak_ptrs = ak_head_base + sk_offs[None, :] * stride_ask + d_offs[:, None] * stride_akd

                aq_vals = tl.load(aq_ptrs, mask=(sq_valid[:, None] & d_valid[None, :]), other=0.0)
                ak_vals = tl.load(ak_ptrs, mask=(sk_valid[None, :] & d_valid[:, None]), other=0.0)

                dot += tl.dot(aq_vals, ak_vals)

            dot *= Softmax_Scale

            attn_scores = tl.where(h_ids[:, None, None] == h, dot[None, :, :], attn_scores)

        # apply causal mask
        casual_mask = tl.full([BLOCK_SQ, BLOCK_SK], float("-inf"), dtype=tl.float32)
        casual_mask = tl.where((sq[:, None] < sk_offs[None, :]), casual_mask, 0.0)
        attn_scores += casual_mask[None, :, :]

        if SPARSE_LOSS:
            attn_scores += sparse_mask[None, :, :]

        # softmax
        softmax_attn_i = tl.exp(attn_scores - m_i[:, :, None]) / d_i[:, :, None]
        softmax_index_i = tl.exp(index_scores - m1_i[:, None]) / d1_i[:, None]

        # reduce head dim
        softmax_attn_i = tl.sum(softmax_attn_i, axis=0) / AH

        # loss
        loss_sk = softmax_attn_i * (tl.log(softmax_attn_i + 1e-10) - tl.log(softmax_index_i + 1e-10))
        loss_i += loss_sk.sum(axis=-1)

    # Store loss
    tl.store(Loss_ptr + b * stride_lb + sq * stride_ls, loss_i, mask=sq_valid)


def compute_dsa_indexer_loss_triton(
    q: torch.Tensor,
    weights: torch.Tensor, 
    k: torch.Tensor,
    attn_query: torch.Tensor,
    attn_key: torch.Tensor,
    topk: int,
    softmax_scale: float,
    loss_coeff: float,
    mask: Optional[torch.Tensor] = None,
    sparse_loss: Optional[bool] = False,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> torch.Tensor:
    """
    Fused index score computation + TopK selection using Triton.
    
    This avoids materializing the full [B, Sq, Sk] index_scores tensor.
    
    Args:
        q: [Sq, B, H, D] query tensor
        weights: [Sq, B, H] attention weights  
        k: [Sk, B, D] key tensor
        attn_query: [Sq, B, H, D] attention query tensor
        attn_key: [Sq, B, H, D] attention key tensor
        topk: number of top indices to return
        softmax_scale: softmax scale
        mask: [B, Sq, Sk] mask tensor (optional)
        sparse_loss: whether to use sparse loss
        loss_coeff: loss coefficient
    Returns:
        topk_indices: [B, Sq, TopK] int64 tensor of top-k indices
    """
    Sq, B, H, D = q.shape
    Sk = k.shape[0]

    ASq, AB, AH, AD = attn_query.shape
    ASk = attn_key.shape[0]

    assert Sq == ASq and Sk == ASk and AB == B and AD == D

    # Clamp topk to valid range
    topk = min(topk, Sk)

    if pg_collection is not None and pg_collection.tp.size() > 1:
        assert pg_collection.cp is None or pg_collection.cp.size() == 1, "This kernel does not support TP and CP together."
        tp_size = pg_collection.tp.size()

        if ASq == ASk:
            assert ASq % tp_size == 0
            packed_qk = torch.stack([attn_query, attn_key], dim=-1)
            # [ASq, B, H, D, 2] -> [tp_size, H, ASq // tp_size, B, D, 2]
            packed_qk = (
                packed_qk.permute(2, 0, 1, 3, 4)
                .reshape(AH, tp_size, ASq // tp_size, AB, AD, 2)
                .transpose(0, 1)
                .contiguous()
            )
            output_qk = torch.empty_like(packed_qk)
            torch.distributed.all_to_all_single(
                output_qk, 
                packed_qk,
                group=pg_collection.tp
            )
            # [tp_size, H, ASq // tp_size, B, D, 2] -> [H * tp_size, ASq // tp_size, B, D, 2]
            output_qk = output_qk.reshape(AH * tp_size, ASq // tp_size, AB, AD, 2)
            # [H * tp_size, ASq // tp_size, B, D, 2] -> [ASq // tp_size, B, H * tp_size, D, 2]
            output_qk = output_qk.permute(1, 2, 0, 3, 4).contiguous()
            attn_query, attn_key = output_qk.unbind(dim=-1)
            attn_query = attn_query.contiguous()
            attn_key = attn_key.contiguous()
        else:
            # TODO: support different ASq and ASk
            raise ValueError(f"ASq {ASq} != ASk {ASk}")

        ASq, AB, AH, AD = attn_query.shape
        ASk = attn_key.shape[0]
        
        # TODO: do not split index scores, it introduces extra problem in communication and casual mask
    
    # Output tensor
    out_idx = torch.empty((B, Sq, topk), dtype=torch.int64, device=q.device)
    out_loss = torch.empty((B, Sq), dtype=torch.float32, device=q.device)
    
    BLOCK_SQ = 16
    BLOCK_SK = 128
    BLOCK_D = 128

    num_sq_blocks = (Sq + BLOCK_SQ - 1) // BLOCK_SQ
    grid = (B, num_sq_blocks,)
    
    # Handle mask strides
    if mask is not None:
        stride_mb = mask.stride(0)
        stride_ms = mask.stride(1)
        stride_mk = mask.stride(2)
        has_mask = True
    else:
        stride_mb = stride_ms = stride_mk = 0
        has_mask = False
    
    _compute_dsa_indexer_loss_kernel[grid](
        Q_ptr=q,
        K_ptr=k,
        W_ptr=weights,
        Mask_ptr=mask,
        Out_Idx_ptr=out_idx,
        Attn_Query_ptr=attn_query,
        Attn_Key_ptr=attn_key,
        Loss_ptr=out_loss,
        # Q strides
        stride_qs=q.stride(0),
        stride_qb=q.stride(1),
        stride_qh=q.stride(2),
        stride_qd=q.stride(3),
        # K strides
        stride_ks=k.stride(0),
        stride_kb=k.stride(1),
        stride_kd=k.stride(2),
        # W strides
        stride_ws=weights.stride(0),
        stride_wb=weights.stride(1),
        stride_wh=weights.stride(2),
        # Mask strides
        stride_mb=stride_mb,
        stride_ms=stride_ms,
        stride_mk=stride_mk,
        # Out strides
        stride_ib=out_idx.stride(0),
        stride_is=out_idx.stride(1),
        stride_ik=out_idx.stride(2),
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
        # Dimensions
        H=H,
        D=D,
        AH=AH,
        AD=AD,
        Sq=Sq,
        Sk=Sk,
        BLOCK_SQ=BLOCK_SQ,
        BLOCK_SK=BLOCK_SK,
        BLOCK_D=BLOCK_D,
        TOPK=topk,
        HAS_MASK=has_mask,
        SPARSE_LOSS=sparse_loss,
        Softmax_Scale=softmax_scale,
    )

    indexer_loss = out_loss.mean() * loss_coeff

    if pg_collection is not None and pg_collection.tp.size() > 1:
        # reduce loss
        torch.distributed.all_reduce(indexer_loss, group=pg_collection.tp)
        indexer_loss /= pg_collection.tp.size()
    
    return out_idx, indexer_loss, out_loss

class FusedDSAIndexerLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, weights, k, query, key, softmax_scale, topk, loss_coeff, sparse_loss):
        """
        Fused forward: index_scores never materialized in full.
        """
        # Run fused Triton kernel
        topk_indices, loss = dsa_fwd_kernel(
            q, weights, k, query, key, softmax_scale, topk, loss_coeff, sparse_loss
        )
        
        # Save for backward (recomputation strategy)
        ctx.save_for_backward(q, weights, k, query, key, topk_indices)
        ctx.softmax_scale = softmax_scale
        ctx.loss_coeff = loss_coeff
        ctx.sparse_loss = sparse_loss
        
        return topk_indices, loss
    
    @staticmethod
    def backward(ctx, grad_topk_indices, grad_loss):
        """
        Backward: Recompute what we need.
        """
        q, weights, k, query, key, topk_indices = ctx.saved_tensors

        grad_q , grad_weights, grad_k = dsa_bwd_kernel(
            q, weights, k, query, key, topk_indices, 
            ctx.softmax_scale, ctx.loss_coeff * grad_loss, ctx.sparse_loss,
        )
        
        # query and key are detached in forward, so return None for their gradients
        return grad_q, grad_weights, grad_k, None, None, None, None, None, None
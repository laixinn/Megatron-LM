# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""TileLang port of _fwd_fused_indexer_loss_stage2_kernel.

Stage 1 (computing softmax_m/d and m1/d1) reuses the existing Triton kernel.
Stage 2 (KL-divergence loss accumulation) is re-implemented here using TileLang.

The computation for each (batch b, sq_block) is:
    softmax_attn[sq, sk] = mean_h( exp(Q[sq,h,:] @ K[sk,h,:]^T * scale
                                       [+ index_mask]  (sparse only)
                                       + causal_mask
                                       - m[b,h,sq]) / d[b,h,sq] )
    softmax_index[sq, sk] = exp(index_scores[b,sq,sk]
                                [+ index_mask]  (sparse only)
                                - m1[b,sq]) / d1[b,sq]
    loss[b, sq] = sum_sk( softmax_attn * (log(softmax_attn+eps) - log(softmax_index+eps)) )

Implementation notes:
  - Triton stage-1 writes softmax_m/d/m1/d1 at GLOBAL sq = Sq_offset + aq_local.
    Therefore the buffers are allocated with size (Sq_offset + ASq) so that
    stage-1's writes are in-bounds and TileLang stage-2 can read from the same
    addresses (sq_off + aq_start + i).
  - Index_Scores / Index_Mask tensors are LOCAL (shape [B, ASq, Sk]), accessed at
    aq_start + i (0-indexed local), which is correct.
  - Fragment-to-1D-fragment accumulation inside T.Parallel(2D) is incorrect in
    TileLang because the 1D fragment's thread-to-element layout doesn't match.
  - The correct pattern for a 2D->1D reduction is:
      1. T.copy(2D_fragment, shared_2D)   -- layout-conversion via T.copy
      2. for i in T.Parallel(BLOCK_SQ): x = shared[i,0]; for j in serial(1,SK): x += ...
"""
from functools import lru_cache
from typing import Optional, Tuple

import torch

import tilelang.language as T
from tilelang import JITKernel

from megatron.core.transformer.experimental_attention_variant.fused_loss import (
    _fwd_fused_indexer_loss_stage1_kernel,
)


# ---------------------------------------------------------------------------
# Factory: builds and JIT-compiles the stage-2 TileLang kernel
# ---------------------------------------------------------------------------

def _make_stage2_kernel(
    B: int,
    ASq: int,
    Sk: int,
    AH: int,
    AD: int,
    Sq_offset: int,
    full_Sq: int,
    BLOCK_SQ: int,
    BLOCK_SK: int,
    BLOCK_D: int,
    sparse_loss: bool,
    softmax_scale: float,
    dtype: str,
) -> JITKernel:
    """Return a compiled TileLang JITKernel for stage 2.

    All parameters are compile-time constants enabling full specialisation.

    Tensor sizes:
      softmax_m, softmax_d : (B, AH, ASq_total)  where ASq_total = Sq_offset + ASq
      m1, d1               : (B, ASq_total)
      Index_Scores/Mask    : (B, ASq, Sk)   -- LOCAL, 0-indexed
      Attn_Query           : (ASq, B, AH, AD) -- LOCAL
      Loss                 : (B, ASq)         -- LOCAL output
    """
    num_sq_blocks = (ASq + BLOCK_SQ - 1) // BLOCK_SQ
    num_sk_blocks = (Sk + BLOCK_SK - 1) // BLOCK_SK
    num_dk_blocks = (AD + BLOCK_D - 1) // BLOCK_D
    scale_f32  = float(softmax_scale)
    sq_off     = int(Sq_offset)
    ASq_total  = sq_off + ASq   # size of the m/d/m1/d1 dimension

    if sparse_loss:
        @T.prim_func
        def _kernel_sparse(
            Attn_Query:  T.Tensor((ASq, B, AH, AD), dtype),
            Attn_Key:    T.Tensor((Sk,  B, AH, AD), dtype),
            Index_Scores: T.Tensor((B, ASq, Sk), 'float32'),
            Index_Mask:   T.Tensor((B, ASq, Sk), 'float32'),
            softmax_m:   T.Tensor((B, AH, ASq_total), 'float32'),
            softmax_d:   T.Tensor((B, AH, ASq_total), 'float32'),
            m1:          T.Tensor((B, ASq_total), 'float32'),
            d1:          T.Tensor((B, ASq_total), 'float32'),
            Loss:        T.Tensor((B, ASq), 'float32'),
        ):
            with T.Kernel(B, num_sq_blocks, threads=1024) as (b, sq_blk):
                Q_shared  = T.alloc_shared((BLOCK_SQ, BLOCK_D), dtype)
                K_shared  = T.alloc_shared((BLOCK_SK, BLOCK_D), dtype)
                IS_shared = T.alloc_shared((BLOCK_SQ, BLOCK_SK), 'float32')
                IM_shared = T.alloc_shared((BLOCK_SQ, BLOCK_SK), 'float32')
                SA_shared = T.alloc_shared((BLOCK_SQ, BLOCK_SK), 'float32')
                m1_shared = T.alloc_shared((BLOCK_SQ,), 'float32')
                d1_shared = T.alloc_shared((BLOCK_SQ,), 'float32')
                loss_sh   = T.alloc_shared((BLOCK_SQ,), 'float32')

                h_scores     = T.alloc_fragment((BLOCK_SQ, BLOCK_SK), 'float32')
                softmax_attn = T.alloc_fragment((BLOCK_SQ, BLOCK_SK), 'float32')

                aq_start = sq_blk * BLOCK_SQ

                # Load m1/d1 from global (sq_off + aq_start + i) position
                for i in T.Parallel(BLOCK_SQ):
                    m1_shared[i] = m1[b, sq_off + aq_start + i]
                    d1_shared[i] = d1[b, sq_off + aq_start + i]
                    loss_sh[i]   = T.float32(0.0)

                for sk_blk in T.serial(num_sk_blocks):
                    sk_start = sk_blk * BLOCK_SK
                    if sk_start >= sq_off + aq_start + 1:
                        T.loop_break()

                    # Load index mask and masked index scores (local indexing)
                    for i, j in T.Parallel(BLOCK_SQ, BLOCK_SK):
                        IM_shared[i, j] = Index_Mask[b, aq_start + i, sk_start + j]
                    for i, j in T.Parallel(BLOCK_SQ, BLOCK_SK):
                        IS_shared[i, j] = (
                            Index_Scores[b, aq_start + i, sk_start + j] + IM_shared[i, j]
                        )

                    T.clear(softmax_attn)

                    for h in T.Parallel(AH):
                        T.clear(h_scores)
                        for d_blk in T.Pipelined(num_dk_blocks):
                            d_off = d_blk * BLOCK_D
                            for i, j in T.Parallel(BLOCK_SQ, BLOCK_D):
                                Q_shared[i, j] = Attn_Query[aq_start + i, b, h, d_off + j]
                            for i, j in T.Parallel(BLOCK_SK, BLOCK_D):
                                K_shared[i, j] = Attn_Key[sk_start + i, b, h, d_off + j]
                            T.gemm(Q_shared, K_shared, h_scores, transpose_B=True)

                        # Accumulate softmax_attn over heads (fragment+fragment, works correctly)
                        for i, j in T.Parallel(BLOCK_SQ, BLOCK_SK):
                            softmax_attn[i, j] = softmax_attn[i, j] + T.exp(
                                h_scores[i, j] * T.float32(scale_f32)
                                + IM_shared[i, j]     # sparse mask on attention scores
                                + T.Select(           # causal mask (global coords)
                                    (sq_off + aq_start + i) < (sk_start + j),
                                    T.float32(-1e30),
                                    T.float32(0.0),
                                )
                                - softmax_m[b, h, sq_off + aq_start + i]
                            ) / softmax_d[b, h, sq_off + aq_start + i]

                    # Copy 2D fragment -> shared for 2D->1D KL reduction
                    T.copy(softmax_attn, SA_shared)

                    for i in T.Parallel(BLOCK_SQ):
                        p0 = SA_shared[i, 0] / T.float32(AH)
                        q0 = T.exp(IS_shared[i, 0] - m1_shared[i]) / d1_shared[i]
                        loss_sh[i] = loss_sh[i] + p0 * (
                            T.log(p0 + T.float32(1e-10)) - T.log(q0 + T.float32(1e-10))
                        )
                        for j in T.serial(1, BLOCK_SK):
                            p = SA_shared[i, j] / T.float32(AH)
                            q = T.exp(IS_shared[i, j] - m1_shared[i]) / d1_shared[i]
                            loss_sh[i] = loss_sh[i] + p * (
                                T.log(p + T.float32(1e-10)) - T.log(q + T.float32(1e-10))
                            )

                for i in T.Parallel(BLOCK_SQ):
                    Loss[b, aq_start + i] = loss_sh[i]

        return JITKernel(_kernel_sparse, out_idx=[8], target='cuda')

    else:
        @T.prim_func
        def _kernel_dense(
            Attn_Query:   T.Tensor((ASq, B, AH, AD), dtype),
            Attn_Key:     T.Tensor((Sk,  B, AH, AD), dtype),
            Index_Scores: T.Tensor((B, ASq, Sk), 'float32'),
            softmax_m:    T.Tensor((B, AH, ASq_total), 'float32'),
            softmax_d:    T.Tensor((B, AH, ASq_total), 'float32'),
            m1:           T.Tensor((B, ASq_total), 'float32'),
            d1:           T.Tensor((B, ASq_total), 'float32'),
            Loss:         T.Tensor((B, ASq), 'float32'),
        ):
            with T.Kernel(B, num_sq_blocks, threads=1024) as (b, sq_blk):
                Q_shared  = T.alloc_shared((BLOCK_SQ, BLOCK_D), dtype)
                K_shared  = T.alloc_shared((BLOCK_SK, BLOCK_D), dtype)
                IS_shared = T.alloc_shared((BLOCK_SQ, BLOCK_SK), 'float32')
                SA_shared = T.alloc_shared((BLOCK_SQ, BLOCK_SK), 'float32')
                m1_shared = T.alloc_shared((BLOCK_SQ,), 'float32')
                d1_shared = T.alloc_shared((BLOCK_SQ,), 'float32')
                loss_sh   = T.alloc_shared((BLOCK_SQ,), 'float32')

                h_scores     = T.alloc_fragment((BLOCK_SQ, BLOCK_SK), 'float32')
                softmax_attn = T.alloc_fragment((BLOCK_SQ, BLOCK_SK), 'float32')

                aq_start = sq_blk * BLOCK_SQ

                # Load m1/d1 from global (sq_off + aq_start + i) position
                for i in T.Parallel(BLOCK_SQ):
                    m1_shared[i] = m1[b, sq_off + aq_start + i]
                    d1_shared[i] = d1[b, sq_off + aq_start + i]
                    loss_sh[i]   = T.float32(0.0)

                for sk_blk in T.serial(num_sk_blocks):
                    sk_start = sk_blk * BLOCK_SK
                    if sk_start >= sq_off + aq_start + 1:
                        T.loop_break()

                    # Load index scores (local indexing)
                    for i, j in T.Parallel(BLOCK_SQ, BLOCK_SK):
                        IS_shared[i, j] = Index_Scores[b, aq_start + i, sk_start + j]

                    T.clear(softmax_attn)

                    for h in T.serial(AH):
                        T.clear(h_scores)
                        for d_blk in T.Pipelined(num_dk_blocks):
                            d_off = d_blk * BLOCK_D
                            for i, j in T.Parallel(BLOCK_SQ, BLOCK_D):
                                Q_shared[i, j] = Attn_Query[aq_start + i, b, h, d_off + j]
                            for i, j in T.Parallel(BLOCK_SK, BLOCK_D):
                                K_shared[i, j] = Attn_Key[sk_start + i, b, h, d_off + j]
                            T.gemm(Q_shared, K_shared, h_scores, transpose_B=True)

                        # Accumulate softmax_attn over heads (fragment+fragment, works correctly)
                        for i, j in T.Parallel(BLOCK_SQ, BLOCK_SK):
                            softmax_attn[i, j] = softmax_attn[i, j] + T.exp(
                                h_scores[i, j] * T.float32(scale_f32)
                                + T.Select(          # causal mask (global coords)
                                    (sq_off + aq_start + i) < (sk_start + j),
                                    T.float32(-1e30),
                                    T.float32(0.0),
                                )
                                - softmax_m[b, h, sq_off + aq_start + i]
                            ) / softmax_d[b, h, sq_off + aq_start + i]

                    # Copy 2D fragment -> shared for 2D->1D KL reduction
                    T.copy(softmax_attn, SA_shared)

                    for i in T.Parallel(BLOCK_SQ):
                        p0 = SA_shared[i, 0] / T.float32(AH)
                        q0 = T.exp(IS_shared[i, 0] - m1_shared[i]) / d1_shared[i]
                        loss_sh[i] = loss_sh[i] + p0 * (
                            T.log(p0 + T.float32(1e-10)) - T.log(q0 + T.float32(1e-10))
                        )
                        for j in T.serial(1, BLOCK_SK):
                            p = SA_shared[i, j] / T.float32(AH)
                            q = T.exp(IS_shared[i, j] - m1_shared[i]) / d1_shared[i]
                            loss_sh[i] = loss_sh[i] + p * (
                                T.log(p + T.float32(1e-10)) - T.log(q + T.float32(1e-10))
                            )

                for i in T.Parallel(BLOCK_SQ):
                    Loss[b, aq_start + i] = loss_sh[i]

        return JITKernel(_kernel_dense, out_idx=[7], target='cuda')


# Cache compiled kernels keyed by all compile-time parameters
@lru_cache(maxsize=32)
def _get_stage2_kernel(
    B, ASq, Sk, AH, AD, Sq_offset, full_Sq,
    BLOCK_SQ, BLOCK_SK, BLOCK_D,
    sparse_loss, softmax_scale, dtype,
) -> JITKernel:
    return _make_stage2_kernel(
        B=B, ASq=ASq, Sk=Sk, AH=AH, AD=AD,
        Sq_offset=Sq_offset, full_Sq=full_Sq,
        BLOCK_SQ=BLOCK_SQ, BLOCK_SK=BLOCK_SK, BLOCK_D=BLOCK_D,
        sparse_loss=sparse_loss, softmax_scale=softmax_scale,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Public wrapper – same interface as fwd_fused_indexer_loss
# ---------------------------------------------------------------------------

def fwd_fused_indexer_loss_tilelang(
    index_scores: torch.Tensor,
    attn_query: torch.Tensor,
    attn_key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: Optional[bool] = False,
    index_mask: Optional[torch.Tensor] = None,
    Sq_offset: Optional[int] = 0,
    full_Sq: Optional[int] = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """TileLang stage-2 replacement for fwd_fused_indexer_loss.

    Stage 1 (online softmax statistics) is computed by the existing Triton kernel
    to keep identical numerics for m/d/m1/d1.  Stage 2 (KL-divergence reduction)
    uses the TileLang kernel defined above.

    Args / Returns: identical to
        megatron.core.transformer.experimental_attention_variant.fused_loss.fwd_fused_indexer_loss

    Note on Sq_offset:
        Triton stage-1 writes softmax statistics at GLOBAL positions
        (sq = Sq_offset + aq_local).  The m/d/m1/d1 buffers are therefore
        allocated with size (Sq_offset + ASq) so that both Triton stage-1
        writes and TileLang stage-2 reads refer to valid, in-bounds locations.
    """
    ASq, AB, AH, AD = attn_query.shape
    ASk = attn_key.shape[0]
    assert AH <= 128, "Kernel currently requires AH <= 128."

    BLOCK_SQ  = 128
    BLOCK_SK  = 128
    BLOCK_D   = 64
    num_warps  = 8
    num_stages = 3

    num_sq_blocks = (ASq + BLOCK_SQ - 1) // BLOCK_SQ

    # Prepare index_mask strides for Triton stage-1
    if sparse_loss:
        stride_imb = index_mask.stride(0)
        stride_ims = index_mask.stride(1)
        stride_imk = index_mask.stride(2)
    else:
        index_mask  = torch.empty((0,), dtype=torch.float32, device=attn_query.device)
        stride_imb = stride_ims = stride_imk = 0

    # Triton stage-1 writes at sq = Sq_offset + aq, so allocate (Sq_offset + ASq)
    ASq_total  = Sq_offset + ASq
    softmax_m  = torch.full((AB, AH, ASq_total), float('-inf'), dtype=torch.float32, device=attn_query.device)
    softmax_d  = torch.full((AB, AH, ASq_total), 0.0,           dtype=torch.float32, device=attn_query.device)
    softmax_m1 = torch.full((AB, ASq_total),     float('-inf'), dtype=torch.float32, device=attn_query.device)
    softmax_d1 = torch.full((AB, ASq_total),     0.0,           dtype=torch.float32, device=attn_query.device)
    out_loss_placeholder = torch.empty((AB, ASq), dtype=torch.float32, device=attn_query.device)

    stage1_grid = (AB, num_sq_blocks, AH)
    _fwd_fused_indexer_loss_stage1_kernel[stage1_grid](
        Attn_Query_ptr=attn_query,
        Attn_Key_ptr=attn_key,
        Loss_ptr=out_loss_placeholder,
        Index_Scores_ptr=index_scores,
        Index_Mask_ptr=index_mask,
        m_ptr=softmax_m,
        d_ptr=softmax_d,
        m1_ptr=softmax_m1,
        d1_ptr=softmax_d1,
        stride_asq=attn_query.stride(0),
        stride_aqb=attn_query.stride(1),
        stride_aqh=attn_query.stride(2),
        stride_aqd=attn_query.stride(3),
        stride_ask=attn_key.stride(0),
        stride_akb=attn_key.stride(1),
        stride_akh=attn_key.stride(2),
        stride_akd=attn_key.stride(3),
        stride_lb=out_loss_placeholder.stride(0),
        stride_ls=out_loss_placeholder.stride(1),
        stride_ibs=index_scores.stride(0),
        stride_isq=index_scores.stride(1),
        stride_isk=index_scores.stride(2),
        stride_imb=stride_imb,
        stride_ims=stride_ims,
        stride_imk=stride_imk,
        stride_smmb=softmax_m.stride(0),
        stride_smmh=softmax_m.stride(1),
        stride_smmq=softmax_m.stride(2),
        stride_smdb=softmax_d.stride(0),
        stride_smdh=softmax_d.stride(1),
        stride_smdq=softmax_d.stride(2),
        stride_sm1b=softmax_m1.stride(0),
        stride_sm1q=softmax_m1.stride(1),
        stride_sd1b=softmax_d1.stride(0),
        stride_sd1q=softmax_d1.stride(1),
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

    # ---- Stage 2: TileLang ----
    _dtype_map = {torch.float16: 'float16', torch.bfloat16: 'bfloat16'}
    tl_dtype = _dtype_map.get(attn_query.dtype, 'float16')

    kernel = _get_stage2_kernel(
        B=AB, ASq=ASq, Sk=ASk, AH=AH, AD=AD,
        Sq_offset=Sq_offset, full_Sq=full_Sq,
        BLOCK_SQ=BLOCK_SQ, BLOCK_SK=BLOCK_SK, BLOCK_D=BLOCK_D,
        sparse_loss=bool(sparse_loss),
        softmax_scale=softmax_scale,
        dtype=tl_dtype,
    )

    if sparse_loss:
        out_loss = kernel(
            attn_query, attn_key,
            index_scores, index_mask,
            softmax_m, softmax_d,
            softmax_m1, softmax_d1,
        )
    else:
        out_loss = kernel(
            attn_query, attn_key,
            index_scores,
            softmax_m, softmax_d,
            softmax_m1, softmax_d1,
        )

    indexer_loss = out_loss.mean() * loss_coeff
    return indexer_loss, out_loss

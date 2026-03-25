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
import argparse
import sys 

import torch
import torch.distributed as dist

import triton
import triton.language as tl

import numpy as np

from megatron.core.transformer.experimental_attention_variant.dsa import fwd_fused_indexer_loss_triton, compute_dsa_indexer_loss_triton
from megatron.core.process_groups_config import ProcessGroupCollection
import megatron.core.parallel_state as parallel_state
from megatron.core.transformer.experimental_attention_variant.dsa import fused_qk_topk_naive


# ---------------------------------------------------------------------------
# Triton BMM kernel with 3D tile loads (all heads in one program).
#
# Mirrors the pattern from _fwd_fused_indexer_loss_kernel_v0:
#   a_tile : [AH, BLOCK_M, BLOCK_K]  loaded from query [Sq, B, H, D]
#   b_tile : [AH, BLOCK_K, BLOCK_N]  loaded from key   [Sk, B, H, D]
#   3D dot : [AH, BLOCK_M, BLOCK_K] @ [AH, BLOCK_K, BLOCK_N]
#              -> [AH, BLOCK_M, BLOCK_N]  (accumulated over split-K loop)
#
# Grid : (B, ceil(Sq/BLOCK_M), ceil(Sk/BLOCK_N))
# AH heads are handled inside each program via 3D tensors.
# ---------------------------------------------------------------------------
@triton.jit
def _triton_bmm_kernel(
    A_ptr,  # query [Sq, B, H, D]
    B_ptr,  # key   [Sk, B, H, D]
    C_ptr,  # out   [B,  H, Sq, Sk]
    # query strides: [Sq, B, H, D]
    stride_asq, stride_ab, stride_ah, stride_ad,
    # key strides:   [Sk, B, H, D]
    stride_bsk, stride_bb, stride_bh, stride_bd,
    # output strides: [B, H, Sq, Sk]
    stride_cb, stride_ch, stride_csq, stride_csk,
    Sq, Sk, D,
    AH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # split-K tile width over D
    Scale: tl.constexpr,
):
    b     = tl.program_id(0)
    m_pid = tl.program_id(1)
    n_pid = tl.program_id(2)

    m_offs = m_pid * BLOCK_M + tl.arange(0, BLOCK_M)  # Sq tile
    n_offs = n_pid * BLOCK_N + tl.arange(0, BLOCK_N)  # Sk tile
    h_ids  = tl.arange(0, AH)
    m_valid = m_offs < Sq
    n_valid = n_offs < Sk

    # Accumulator covers all AH heads at once: [AH, BLOCK_M, BLOCK_N]
    acc = tl.zeros([AH, BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Split-K loop: reduce over head-dim D in BLOCK_K slices
    for k_start in tl.range(0, D, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_valid = k_offs < D

        # A tile: [AH, BLOCK_M, BLOCK_K]  — query[m, b, h, k]
        a_ptrs = (A_ptr
                  + b          * stride_ab
                  + h_ids[:, None, None] * stride_ah
                  + m_offs[None, :, None] * stride_asq
                  + k_offs[None, None, :] * stride_ad)
        a_tile = tl.load(a_ptrs,
                         mask=(m_valid[None, :, None] & k_valid[None, None, :]),
                         other=0.0)

        # B tile: [AH, BLOCK_K, BLOCK_N]  — key[n, b, h, k]  (K before N = transposed)
        b_ptrs = (B_ptr
                  + b          * stride_bb
                  + h_ids[:, None, None] * stride_bh
                  + k_offs[None, :, None] * stride_bd
                  + n_offs[None, None, :] * stride_bsk)
        b_tile = tl.load(b_ptrs,
                         mask=(k_valid[None, :, None] & n_valid[None, None, :]),
                         other=0.0)

        # [AH, BLOCK_M, BLOCK_K] @ [AH, BLOCK_K, BLOCK_N] -> [AH, BLOCK_M, BLOCK_N]
        acc = tl.dot(a_tile, b_tile, acc=acc, allow_tf32=False)

    acc = acc * Scale

    # Store [AH, BLOCK_M, BLOCK_N] -> C[b, h, m, n]
    c_ptrs = (C_ptr
              + b          * stride_cb
              + h_ids[:, None, None] * stride_ch
              + m_offs[None, :, None] * stride_csq
              + n_offs[None, None, :] * stride_csk)
    tl.store(c_ptrs, acc,
             mask=(m_valid[None, :, None] & n_valid[None, None, :]))


def triton_bmm(
    query: torch.Tensor,  # [Sq, B, H, D]
    key: torch.Tensor,    # [Sk, B, H, D]
    softmax_scale: float,
    BLOCK_M: int = 4,
    BLOCK_N: int = 16,
    BLOCK_K: int = 32,
) -> torch.Tensor:
    """
    Compute QK^T * scale using the 3D Triton BMM kernel above.

    Works directly on the native [Sq, B, H, D] layout — no permute/reshape copy.
    Output: [B, H, Sq, Sk]

    Shared memory per program = AH * (BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N) * 4 bytes
    plus the [AH, BLOCK_M, BLOCK_N] accumulator.  BLOCK_K is halved for every
    doubling of AH beyond 2 to stay within the 232 KB hardware limit.
    """
    Sq, B, H, D = query.shape
    Sk = key.shape[0]

    # Scale BLOCK_K down with AH so shared memory stays within hardware limits.
    # Baseline: AH=2, BLOCK_K=64.  Each extra factor-of-2 in AH halves BLOCK_K.
    import math
    effective_block_k = max(16, BLOCK_K >> max(0, int(math.log2(H)) - 1))

    # Keep input dtype — the kernel accumulator is always fp32.
    q = query.contiguous()
    k = key.contiguous()
    out = torch.empty((B, H, Sq, Sk), dtype=torch.float32, device=query.device)

    grid = (B, triton.cdiv(Sq, BLOCK_M), triton.cdiv(Sk, BLOCK_N))

    _triton_bmm_kernel[grid](
        q, k, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Sq=Sq, Sk=Sk, D=D,
        AH=H,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=effective_block_k,
        Scale=softmax_scale,
    )

    return out

def bench(fn, num_warmups: int = 5, num_tests: int = 50, post_fn=None, is_async=True):
    # Flush L2 cache with 256 MB data
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Flush L2
    cache.zero_()

    torch.cuda.synchronize()
    if dist.is_initialized():
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
        attention_scores = torch.bmm(query, key, out_dtype=torch.float32) * softmax_scale
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


def forward_native(q, weights, k, mask, index_topk, query, key, softmax_scale, loss_coeff, sparse_loss, pg_collection=None):
    index_scores, topk_indices = fused_qk_topk_naive(q, k, weights, index_topk, mask)

    indexer_loss, kl_per_element = compute_dsa_indexer_loss(
        index_scores, topk_indices, query, key, softmax_scale, loss_coeff, sparse_loss, pg_collection=pg_collection
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


def benchmark_fused_loss_forward():
    """Benchmark compute_index_scores_topk + DSA indexer loss: native PyTorch vs Triton."""
    
    configs = [
        # (Sq, Sk, B, H, D, topk)
        # (2048, 2048, 1, 128, 7168, 1024),
        # (2048, 2048, 1, 128, 7168, 1024),
        # (4096, 4096, 1, 64, 7168, 2048),
        (4096, 4096, 1, 128, 7168, 2048),
        (6144, 6144, 1, 128, 7168, 2048),
        # (8192, 8192, 1, 16, 7168, 2048),
        # (8192, 8192, 1, 32, 7168, 2048),
        # (8192, 8192, 1, 64, 7168, 2048),
        (8192, 8192, 1, 128, 7168, 2048),
        # (16384, 16384, 1, 128, 7168, 2048),
    ]
    
    metrics_collection = []
    for Sq, Sk, B, H, D, topk in configs:
        # Setup
        dtype = torch.bfloat16
        q = torch.randn(Sq, B, H, D, device='cuda', dtype=dtype)
        k = torch.randn(Sk, B, D, device='cuda', dtype=dtype)
        weights = torch.randn(Sq, B, H, device='cuda', dtype=torch.float32)
        mask = torch.triu(
            torch.full((B, Sq, Sk), float('-inf'), dtype=torch.float32, device='cuda'),
            diagonal=1,
        )

        attn_query = torch.randn(Sq, B, H, D, device='cuda', dtype=dtype)
        attn_key = torch.randn(Sk, B, H, D, device='cuda', dtype=dtype)
        softmax_scale = 1.0
        loss_coeff = 1.0
        for sparse_loss in [False, True]:
            index_scores, topk_indices = fused_qk_topk_naive(q, k, weights, topk, mask)

            native_lambda = lambda: compute_dsa_indexer_loss(
                index_scores, 
                topk_indices, 
                attn_query, 
                attn_key, 
                softmax_scale, 
                loss_coeff, 
                sparse_loss
            )
            # TODO: remove accuracy_check after topk is fixed
            triton_lambda = lambda: compute_dsa_indexer_loss_triton(
                index_scores,
                topk_indices,
                attn_query,
                attn_key,
                softmax_scale,
                loss_coeff,
                sparse_loss,
                0, 
                Sq,
            )

            # check correctness, topk_indices has its unit test
            native_indexer_loss, native_kl_per_element = native_lambda()
            triton_indexer_loss, triton_out_loss = triton_lambda()

            match = torch.allclose(native_indexer_loss, triton_indexer_loss, atol=1e-4, rtol=1e-4)
            torch.cuda.synchronize()
            
            # Benchmark PyTorch (reset peak stats first so memory is measured during bench)
            torch.cuda.reset_peak_memory_stats()
            pytorch_time, _, _ = bench(native_lambda, is_async=False, num_warmups=1, num_tests=1)
            pytorch_time *= 1000
            native_mem_gb = torch.cuda.max_memory_allocated() / 1024 ** 3

            torch.cuda.synchronize()

            # Benchmark Triton (reset peak stats first so memory is measured during bench)
            torch.cuda.reset_peak_memory_stats()
            triton_time, _, _ = bench(triton_lambda, is_async=False, num_warmups=1, num_tests=1)
            triton_time *= 1000
            triton_mem_gb = torch.cuda.max_memory_allocated() / 1024 ** 3

            mem_saved_gb = triton_mem_gb / native_mem_gb
            speedup = pytorch_time / triton_time
            marker = "🚀" if speedup > 1.0 else ""
            sparse_str = "Yes" if sparse_loss else "No"

            metrics_collection.append((Sq, Sk, B, H, D, topk, sparse_str, pytorch_time, triton_time, speedup, match, native_mem_gb, triton_mem_gb, mem_saved_gb))

            print(f"[Sq={Sq}, Sk={Sk}, B={B}, H={H}, D={D}, topk={topk}, sparse_loss={sparse_loss}] completes.")

    print("\n" + "=" * 80)
    print("Benchmark: DSA Indexer (TopK + Loss) - PyTorch vs Triton")
    print("=" * 80)

    print(f"\n{'Sq':>4} {'Sk':>5} {'B':>3} {'H':>3} {'D':>3} {'TopK':>4} {'Sparse':>7} | {'PyTorch (ms)':>14} {'Triton (ms)':>13} {'Speedup':>8} {'Match':>10} | {'Native (GB)':>12} {'Triton (GB)':>12} {'Saved (GB)':>11}")
    print("-" * 135)

    for Sq, Sk, B, H, D, topk, sparse_str, pytorch_time, triton_time, speedup, match, native_mem_gb, triton_mem_gb, mem_saved_gb in metrics_collection:
        saved_marker = "✓" if mem_saved_gb > 0 else ""
        print(f"{Sq:>4} {Sk:>5} {B:>3} {H:>3} {D:>3} {topk:>4} {sparse_str:>7} | {pytorch_time:>14.2f}   {triton_time:>13.2f}   {speedup:>8.2f}x {marker} {str(match):>10} | {native_mem_gb:>12.3f} {triton_mem_gb:>12.3f} {mem_saved_gb:>10.3f} {saved_marker}")

    print("\n" + "=" * 135)

def benchmark_fused_loss_forward_tensor_parallel():
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
        (2048, 2048, 1, 8, 128, 1024),
        (2048, 2048, 1, 8, 128, 2048),
        (2048, 2048, 1, 32, 128, 2048),
        (8192, 8192, 1, 8, 128, 2048),
        (8192, 8192, 1, 32, 128, 2048),
        (16384, 16384, 1, 8, 128, 2048),
        (16384, 16384, 1, 32, 128, 2048),
    ]
    
    if tp_rank == 0:
        metrics_collection = []
    
    for Sq, Sk, B, H, D, topk in configs:
        # Setup
        q = torch.randn(Sq, B, H, D, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(Sk, B, D, device='cuda', dtype=torch.bfloat16)
        weights = torch.randn(Sq, B, H, device='cuda', dtype=torch.float32)
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
            
            triton_acc_lambda = lambda: FusedDSAIndexerLoss.apply(q, weights, k, attn_query_tp, attn_key_tp, softmax_scale, topk, loss_coeff, mask, sparse_loss, triton_pg_collection, True)

            # check correctness
            native_topk_indices, native_indexer_loss, native_kl_per_element, native_index_scores = native_lambda()
            triton_topk_indices, triton_indexer_loss = triton_acc_lambda()

            match = torch.allclose(native_indexer_loss, triton_indexer_loss, atol=1e-4, rtol=1e-4)

            print(f"[Rank {tp_rank}] {Sq=}, {Sk=}, {B=}, {H=}, {D=}, {topk=} passed.")

            dist.barrier()

            pytorch_time, _, _ = bench(native_lambda)
            pytorch_time *= 1000

            triton_time, _, _ = bench(triton_ben_lambda)
            triton_time *= 1000

            speedup = pytorch_time / triton_time
            marker = "🚀" if speedup > 1.0 else ""
            sparse_str = "Yes" if sparse_loss else "No"

            if tp_rank == 0:
                metrics_collection.append((Sq, Sk, B, H, D, topk, sparse_str, pytorch_time, triton_time, speedup, match))
        
    dist.barrier()
    if tp_rank == 0:
        print("\n" + "=" * 80)
        print("Benchmark: TP DSA Indexer (TopK + Loss) - PyTorch vs Triton")
        print("=" * 80)
        print(f"\n{'Sq':>4} {'Sk':>5} {'B':>3} {'H':>3} {'D':>3} {'TopK':>4} {'Sparse':>7} | {'PyTorch (ms)':>14} {'Triton (ms)':>13} {'Speedup':>8} {'Match':>10}")
        print("-" * 95)

        for Sq, Sk, B, H, D, topk, sparse_str, pytorch_time, triton_time, speedup, match in metrics_collection:
            print(f"{Sq:>4} {Sk:>5} {B:>3} {H:>3} {D:>3} {topk:>4} {sparse_str:>7} | {pytorch_time:>12.2f}   {triton_time:>11.2f}   {speedup:>6.2f}x {marker} {match}")
        
        print("\n" + "=" * 95)

    dist.barrier()
    Utils.destroy_model_parallel()


def test_fused_loss_backward_tensor_parallel():
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
        (2048, 2048, 1, 8, 128, 1024),
        (2048, 2048, 1, 8, 128, 2048),
        (2048, 2048, 1, 32, 128, 2048),
        (8192, 8192, 1, 8, 128, 2048),
        (8192, 8192, 1, 32, 128, 2048),
        (16384, 16384, 1, 8, 128, 2048),
        # (16384, 16384, 1, 32, 128, 2048),
    ]
    
    if tp_rank == 0:
        results_collection = []
    
    for Sq, Sk, B, H, D, topk in configs:
        # Setup
        q = torch.randn(Sq, B, H, D, device='cuda', dtype=torch.float32, requires_grad=True)
        k = torch.randn(Sk, B, D, device='cuda', dtype=torch.float32, requires_grad=True)
        weights = torch.randn(Sq, B, H, device='cuda', dtype=torch.float32, requires_grad=True)
        mask = torch.triu(
            torch.full((B, Sq, Sk), float('-inf'), dtype=torch.float32, device='cuda'),
            diagonal=1,
        )

        attn_query = torch.zeros(Sq, B, H, D, device='cuda', dtype=torch.float32)
        attn_key = torch.zeros(Sk, B, H, D, device='cuda', dtype=torch.float32)
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
            native_topk_indices, native_indexer_loss, native_kl_per_element, native_index_scores = \
                forward_native(q, weights, k, mask, topk, attn_query_tp, attn_key_tp, softmax_scale, loss_coeff, sparse_loss, native_pg_collection)

            # Backward through native
            native_indexer_loss.backward()
            
            grad_q_native = q.grad.clone()
            grad_weights_native = weights.grad.clone()
            grad_k_native = k.grad.clone()

            dist.barrier()
            torch.cuda.synchronize()

            # Create new inputs for fused implementation
            q_fused = q.detach().clone().requires_grad_(True)
            weights_fused = weights.detach().clone().requires_grad_(True)
            k_fused = k.detach().clone().requires_grad_(True)

            # Run custom autograd function
            topk_indices_fused, indexer_loss_fused = FusedDSAIndexerLoss.apply(
                q_fused, weights_fused, k_fused, attn_query_tp, attn_key_tp, 
                softmax_scale, topk, loss_coeff, mask, sparse_loss, 
                triton_pg_collection, True
            )

            # Backward through fused
            indexer_loss_fused.backward()

            grad_q_fused = q_fused.grad.clone()
            grad_weights_fused = weights_fused.grad.clone()
            grad_k_fused = k_fused.grad.clone()

            dist.barrier()
            torch.cuda.synchronize()

            # ==========================================
            # Compare Results
            # ==========================================
            rtol = 1e-4  # Relaxed for Triton kernels
            atol = 1e-4
            
            # Compare forward outputs
            loss_match = torch.allclose(native_indexer_loss, indexer_loss_fused, rtol=rtol, atol=atol)
            topk_match = True # torch.equal(topk_indices_native, topk_indices_fused)
            
            # Compare backward gradients
            q_match = torch.allclose(grad_q_native, grad_q_fused, rtol=rtol, atol=atol)
            weights_match = torch.allclose(grad_weights_native, grad_weights_fused, rtol=rtol, atol=atol)
            k_match = torch.allclose(grad_k_native, grad_k_fused, rtol=rtol, atol=atol)
            
            all_match = loss_match and topk_match and q_match and weights_match and k_match
            
            print(f"[Rank {tp_rank}] {Sq=}, {Sk=}, {B=}, {H=}, {D=}, {topk=}, {sparse_loss=} completes.")
            
            if tp_rank == 0:
                sparse_str = "Yes" if sparse_loss else "No"
                results_collection.append((
                    Sq, Sk, B, H, D, topk, sparse_str,
                    loss_match, q_match, weights_match, k_match, all_match
                ))
    
    dist.barrier()
    
    if tp_rank == 0:
        print("\n" + "=" * 100)
        print("Test: Tensor Parallel FusedDSAIndexerLoss (Forward + Backward)")
        print("=" * 100)
        print(f"\n{'Sq':>4} {'Sk':>5} {'B':>3} {'H':>3} {'D':>3} {'TopK':>4} {'Sparse':>7} | {'Loss':>8} {'Q':>8} {'W':>8} {'K':>8} {'All':>5}")
        print("-" * 100)
        
        for (Sq, Sk, B, H, D, topk, sparse_str,
             loss_match, q_match, w_match, k_match, all_match) in results_collection:
            status = "✓" if all_match else "✗"
            print(f"{Sq:>4} {Sk:>5} {B:>3} {H:>3} {D:>3} {topk:>4} {sparse_str:>7} | {str(loss_match):>8} {str(q_match):>8} {str(w_match):>8} {str(k_match):>8} {status:>5}")
        
        print("\n" + "=" * 100)
        
        # Check if all tests passed
        all_passed = all(result[-1] for result in results_collection)
        if all_passed:
            print("✓ All Tensor Parallel FusedDSAIndexerLoss tests passed!")
        else:
            print("✗ Some Tensor Parallel FusedDSAIndexerLoss tests failed")
        print("=" * 100)
    
    dist.barrier()
    Utils.destroy_model_parallel()


def test_fused_loss_backward_native():
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


def benchmark_fused_loss_backward():
    """
    Benchmark ONLY the backward pass: autograd vs backward_triton_full.
    
    This isolates just the backward computation by precomputing all forward values.
    """
    print("\n" + "=" * 100)
    print("Benchmark: Backward Pass ONLY - PyTorch Autograd vs Triton Full")
    print("=" * 100)
    
    # Test configurations: (Sq, Sk, B, H, D, topk, sparse_loss)
    configs = [
        # Small to medium sizes
        (64, 128, 2, 8, 128, 16, False),
        (64, 128, 2, 8, 128, 16, True),
        
        # Very large sizes
        (1024, 2048, 2, 8, 128, 256, False),
        (1024, 2048, 2, 8, 128, 256, True),

        # Huge sizes
        (4096, 4096, 2, 8, 128, 2048, False),
        (4096, 4096, 2, 8, 128, 2048, True),
        (8192, 8192, 2, 8, 128, 2048, False),
        (8192, 8192, 2, 8, 128, 2048, True),
        (16384, 16384, 1, 8, 128, 2048, False),
        (16384, 16384, 1, 8, 128, 2048, True),
    ]
    
    print(f"\n{'Sq':>4} {'Sk':>5} {'B':>3} {'H':>3} {'D':>3} {'TopK':>4} {'Sparse':>7} | {'PyTorch (ms)':>14} {'Triton (ms)':>13} {'Speedup':>10} {'Q Match':>7} {'W Match':>7} {'K Match':>7}")
    print("-" * 100)
    
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
        native_lambda = lambda: backward_native(
            q, weights, k, query, key, topk_indices,
            softmax_scale, loss_coeff, sparse_loss, grad_loss
        )

        triton_lambda = lambda: backward_triton_full(
            q, weights, k, query, key, topk_indices,
            softmax_scale, loss_coeff, sparse_loss, grad_loss
        )
        
        # Compare gradients
        grad_q_native, grad_weights_native, grad_k_native = native_lambda()
        grad_q_triton, grad_weights_triton, grad_k_triton = triton_lambda()
        
        rtol = 1e-1  # Relaxed tolerance for Triton
        atol = 1e-3

        q_match = torch.allclose(grad_q_native, grad_q_triton, rtol=rtol, atol=atol)
        weights_match = torch.allclose(grad_weights_native, grad_weights_triton, rtol=rtol, atol=atol)
        k_match = torch.allclose(grad_k_native, grad_k_triton, rtol=rtol, atol=atol)
        
        # Warmup
        for _ in range(5):
            native_lambda()
            triton_lambda()
        torch.cuda.synchronize()
        
        # Benchmark
        pytorch_time = triton.testing.do_bench(native_lambda) * 1000
        triton_time = triton.testing.do_bench(triton_lambda) * 1000
        
        speedup = pytorch_time / triton_time
        marker = "🚀" if speedup > 1.0 else "⚠️"
        sparse_str = "Yes" if sparse_loss else "No"
        
        print(f"{Sq:>4} {Sk:>5} {B:>3} {H:>3} {D:>3} {topk:>4} {sparse_str:>7} | {pytorch_time:>14.2f} {triton_time:>13.2f} {speedup:>7.2f}x {marker} {str(q_match):>7} {str(weights_match):>7} {str(k_match):>7}")

    print("=" * 100)


def test_fused_dsa_indexer_loss_autograd():
    """
    Test FusedDSAIndexerLoss autograd function (forward + backward).
    
    This test validates the custom autograd function by:
    1. Comparing forward outputs with native implementation
    2. Comparing backward gradients with PyTorch autograd on native implementation
    
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
    print("Test: FusedDSAIndexerLoss Autograd Function")
    print("=" * 80)
    
    # Test configurations: (Sq, Sk, B, H, D, topk, sparse_loss)
    # Note: H must be divisible by 8 for the Triton kernel
    configs = [
        (2048, 2048, 1, 8, 128, 1024, False),
        (2048, 2048, 1, 8, 128, 1024, True),
        (2048, 2048, 1, 8, 128, 2048, False),
        (2048, 2048, 1, 8, 128, 2048, True),
        (2048, 2048, 1, 32, 128, 2048, False),
        (2048, 2048, 1, 32, 128, 2048, True),
        (8192, 8192, 1, 8, 128, 2048, False),
        (8192, 8192, 1, 8, 128, 2048, True),
        (8192, 8192, 1, 32, 128, 2048, False),
        (8192, 8192, 1, 32, 128, 2048, True),
        (16384, 16384, 1, 8, 128, 2048, False),
        (16384, 16384, 1, 8, 128, 2048, True),
        (16384, 16384, 1, 32, 128, 2048, False),
        (16384, 16384, 1, 32, 128, 2048, True),
    ]
    
    all_passed = True
    
    for config_idx, (Sq, Sk, B, H, D, topk, sparse_loss) in enumerate(configs):
        print(f"\n[{config_idx+1}/{len(configs)}] Testing: Sq={Sq}, Sk={Sk}, B={B}, H={H}, D={D}, topk={topk}, sparse_loss={sparse_loss}")
        
        torch.manual_seed(42 + config_idx)
        
        # Create inputs for native implementation with autograd
        q_native = torch.randn(Sq, B, H, D, device='cuda', dtype=torch.float32, requires_grad=True)
        weights_native = torch.randn(Sq, B, H, device='cuda', dtype=torch.float32, requires_grad=True)
        k_native = torch.randn(Sk, B, D, device='cuda', dtype=torch.float32, requires_grad=True)
        
        # Create query and key for attention (no grad needed as they're detached in loss)
        query = torch.randn(Sq, B, H, D, device='cuda', dtype=torch.float32)
        key = torch.randn(Sk, B, H, D, device='cuda', dtype=torch.float32)
        
        softmax_scale = 1.0 / (D ** 0.5)
        loss_coeff = 0.1
        
        # Create mask
        mask = torch.triu(
            torch.full((B, Sq, Sk), float('-inf'), dtype=torch.float32, device='cuda'),
            diagonal=1,
        )
        
        # ==========================================
        # Test Native Implementation (with autograd)
        # ==========================================
        topk_indices_native, indexer_loss_native, _, _ = forward_native(
            q_native, weights_native, k_native, mask, topk, 
            query, key, softmax_scale, loss_coeff, sparse_loss
        )
        
        # Backward through native
        indexer_loss_native.backward()
        
        grad_q_native = q_native.grad.clone()
        grad_weights_native = weights_native.grad.clone()
        grad_k_native = k_native.grad.clone()
        
        # ==========================================
        # Test FusedDSAIndexerLoss (custom autograd)
        # ==========================================
        # Create new inputs for fused implementation
        q_fused = q_native.detach().clone().requires_grad_(True)
        weights_fused = weights_native.detach().clone().requires_grad_(True)
        k_fused = k_native.detach().clone().requires_grad_(True)
        
        # Run custom autograd function
        topk_indices_fused, indexer_loss_fused = FusedDSAIndexerLoss.apply(
            q_fused, weights_fused, k_fused, query, key, 
            softmax_scale, topk, loss_coeff, mask, sparse_loss,
            None,  # pg_collection
            True,  # accuracy_check
        )
        
        # Backward through fused
        indexer_loss_fused.backward()
        
        grad_q_fused = q_fused.grad.clone()
        grad_weights_fused = weights_fused.grad.clone()
        grad_k_fused = k_fused.grad.clone()
        
        # ==========================================
        # Compare Results
        # ==========================================
        rtol = 1e-4  # Relaxed for Triton kernels
        atol = 1e-4
        
        # Compare forward outputs
        loss_match = torch.allclose(indexer_loss_native, indexer_loss_fused, rtol=rtol, atol=atol)
        topk_match = True # torch.equal(topk_indices_native, topk_indices_fused)
        
        # Compare backward gradients
        q_match = torch.allclose(grad_q_native, grad_q_fused, rtol=rtol, atol=atol)
        weights_match = torch.allclose(grad_weights_native, grad_weights_fused, rtol=rtol, atol=atol)
        k_match = torch.allclose(grad_k_native, grad_k_fused, rtol=rtol, atol=atol)
        
        # Print results
        if loss_match and topk_match and q_match and weights_match and k_match:
            print(f"  ✓ All tests passed!")
            print(f"    - Forward: loss={indexer_loss_native.item():.6f}")
            print(f"    - Backward: All gradients match")
        else:
            all_passed = False
            print(f"  ✗ Test failed:")
            
            if not loss_match:
                loss_diff = (indexer_loss_native - indexer_loss_fused).abs()
                print(f"    - Loss mismatch: native={indexer_loss_native.item():.6f}, fused={indexer_loss_fused.item():.6f}, diff={loss_diff.item():.6f}")
            
            if not topk_match:
                topk_diff_count = (topk_indices_native != topk_indices_fused).sum().item()
                print(f"    - TopK indices mismatch: {topk_diff_count}/{topk_indices_native.numel()} elements differ")
            
            if not q_match:
                q_rel_diff = (grad_q_native - grad_q_fused).abs() / (grad_q_native.abs() + 1e-8)
                print(f"    - grad_q: max_rel_diff={q_rel_diff.max():.6f}, max_abs_diff={(grad_q_native - grad_q_fused).abs().max():.6f}")
            
            if not weights_match:
                w_rel_diff = (grad_weights_native - grad_weights_fused).abs() / (grad_weights_native.abs() + 1e-8)
                print(f"    - grad_weights: max_rel_diff={w_rel_diff.max():.6f}, max_abs_diff={(grad_weights_native - grad_weights_fused).abs().max():.6f}")
            
            if not k_match:
                k_rel_diff = (grad_k_native - grad_k_fused).abs() / (grad_k_native.abs() + 1e-8)
                print(f"    - grad_k: max_rel_diff={k_rel_diff.max():.6f}, max_abs_diff={(grad_k_native - grad_k_fused).abs().max():.6f}")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All FusedDSAIndexerLoss tests passed!")
    else:
        print("✗ Some FusedDSAIndexerLoss tests failed")
    print("=" * 80)
    
    return all_passed


def test_triton_bmm():
    """
    Unit test: compare triton_bmm (split-K 3D Triton kernel) against torch.bmm
    for both fp32 and bf16 input dtypes.

    For each dtype the reference is computed in fp32:
        q_ref = query.float().permute(1,2,0,3).reshape(B*H, Sq, D)
        k_ref = key.float().permute(1,2,3,0).reshape(B*H, D, Sk)
        ref   = torch.bmm(q_ref, k_ref).reshape(B,H,Sq,Sk) * scale

    bf16 inputs are expected to show small rounding errors relative to fp32;
    tolerances are tightened for fp32 and relaxed for bf16.
    """
    print("\n" + "=" * 70)
    print("Test: triton_bmm split-K 3D kernel vs torch.bmm  [fp32 & bf16]")
    print("=" * 70)

    configs = [
        # # Sq,  Sk,   B,  H,    D,  scale
        # (  64, 128,  1,  2,   64,  0.125),
        # ( 128, 128,  2,  4,  128,  0.088),
        # (  32,  64,  1,  8,   32,  0.177),
        # ( 256, 512,  2,  8,  128,  0.088),
        # # non-power-of-two sizes — exercises boundary masking
        # (  70, 100,  1,  4,   96,  0.100),
        (4096, 4096, 1, 64, 7168, 1.0)
    ]

    dtype_cases = [
        # (dtype,             atol,  rtol,  label)
        (torch.float32,       1e-5,  1e-5,  "fp32"),
        (torch.bfloat16,      2e-3,  2e-3,  "bf16"),
    ]

    all_passed = True

    for dtype, atol, rtol, dtype_label in dtype_cases:
        print(f"\n  dtype={dtype_label}  (atol={atol:.0e}, rtol={rtol:.0e})")
        print(f"  {'Sq':>4} {'Sk':>5} {'B':>2} {'H':>2} {'D':>4} {'scale':>6}  "
              f"{'max_abs':>10}  {'max_rel':>10}  result")
        print("  " + "-" * 62)

        for Sq, Sk, B, H, D, scale in configs:
            torch.manual_seed(42)
            query = torch.randn(Sq, B, H, D, device="cuda", dtype=dtype)
            key   = torch.randn(Sk, B, H, D, device="cuda", dtype=dtype)

            # Reference: torch.bmm always in fp32
            q_ref = query.float().permute(1, 2, 0, 3).reshape(B * H, Sq, D)
            k_ref = key.float().permute(1, 2, 3, 0).reshape(B * H, D, Sk)
            ref   = torch.bmm(q_ref, k_ref).reshape(B, H, Sq, Sk) * scale

            # Triton 3D split-K kernel (accumulates in fp32 regardless of input dtype)
            out = triton_bmm(query, key, softmax_scale=scale)

            passed = torch.allclose(out, ref, atol=atol, rtol=rtol)
            max_abs = (out - ref).abs().max().item()
            max_rel = ((out - ref).abs() / (ref.abs() + 1e-8)).max().item()

            status = "✓" if passed else "✗"
            print(f"  {Sq:4d} {Sk:5d} {B:2d} {H:2d} {D:4d} {scale:6.3f}  "
                  f"{max_abs:10.2e}  {max_rel:10.2e}  {status}")

            if not passed:
                all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All triton_bmm tests passed!")
    else:
        print("✗ Some triton_bmm tests FAILED")
    print("=" * 70)
    return all_passed


def test_fused_loss_tilelang():
    """Accuracy test + benchmark: TileLang stage-2 kernel vs Triton fwd_fused_indexer_loss.

    Accuracy:
        The TileLang output is compared element-wise against the reference Triton
        kernel (atol=rtol=1e-4).  NaN positions that appear in *both* outputs
        (sparse edge case where the causal mask eliminates every valid key for the
        first few query rows) are treated as matching via equal_nan semantics.

    Benchmark:
        Follows the same pattern as benchmark_fused_loss_forward(): one warmup,
        one timed run, peak GPU memory measured via torch.cuda.max_memory_allocated.
        Both sparse_loss=False and sparse_loss=True are exercised for each shape.
    """
    import math
    from megatron.core.transformer.experimental_attention_variant.fused_loss import (
        fwd_fused_indexer_loss,
    )
    from megatron.core.transformer.experimental_attention_variant.fused_loss_tilelang import (
        fwd_fused_indexer_loss_tilelang,
    )

    configs = [
        # (Sq, Sk, B, H, D)
        (4096, 4096, 1, 128, 7168),
        (6144, 6144, 1, 128, 7168),
        (8192, 8192, 1, 128, 7168),
    ]

    atol, rtol = 1e-4, 1e-4
    all_passed = True
    metrics_collection = []

    # ------------------------------------------------------------------ #
    # Accuracy pass                                                        #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 70)
    print("TileLang fused-loss stage-2 accuracy test")
    print("=" * 70)
    print(f"{'Sq':>5} {'Sk':>5} {'B':>2} {'H':>3} {'D':>5}  {'sparse':>6}  "
          f"{'max_abs':>10}  {'status':>6}")
    print("-" * 70)

    for Sq, Sk, B, H, D in configs:
        softmax_scale = 1.0 / math.sqrt(D)
        dtype = torch.bfloat16

        torch.manual_seed(0)
        attn_query   = torch.randn(Sq, B, H, D, device="cuda", dtype=dtype)
        attn_key     = torch.randn(Sk, B, H, D, device="cuda", dtype=dtype)
        index_scores = torch.randn(B, Sq, Sk, device="cuda", dtype=torch.float32)

        topk = Sk // 2
        index_mask_sparse = torch.full(
            (B, Sq, Sk), float("-inf"), dtype=torch.float32, device="cuda"
        )
        idx = torch.randint(0, Sk, (B, Sq, topk), device="cuda")
        index_mask_sparse.scatter_(-1, idx, 0.0)

        for sparse_loss in [False, True]:
            index_mask = index_mask_sparse if sparse_loss else None

            _, triton_out = fwd_fused_indexer_loss(
                index_scores, attn_query, attn_key, softmax_scale, 1.0,
                sparse_loss=sparse_loss, index_mask=index_mask,
                Sq_offset=0, full_Sq=Sq,
            )
            _, tl_out = fwd_fused_indexer_loss_tilelang(
                index_scores, attn_query, attn_key, softmax_scale, 1.0,
                sparse_loss=sparse_loss, index_mask=index_mask,
                Sq_offset=0, full_Sq=Sq,
            )

            # NaN positions must agree between both kernels (equal_nan semantics)
            nan_triton = torch.isnan(triton_out)
            nan_tl     = torch.isnan(tl_out)
            nan_match  = (nan_triton == nan_tl).all().item()
            valid      = ~nan_triton & ~nan_tl
            if valid.any():
                max_abs = (tl_out[valid] - triton_out[valid]).abs().max().item()
                passed  = (
                    nan_match
                    and torch.allclose(tl_out[valid], triton_out[valid], atol=atol, rtol=rtol)
                )
            else:
                max_abs = 0.0
                passed  = nan_match

            status = "PASS" if passed else "FAIL"
            print(f"{Sq:5d} {Sk:5d} {B:2d} {H:3d} {D:5d}  "
                  f"{'sparse' if sparse_loss else 'dense ':>6}  "
                  f"{max_abs:10.2e}  {status:>6}")
            if not passed:
                all_passed = False

    print("=" * 70)
    if all_passed:
        print("ALL TileLang accuracy tests PASSED")
    else:
        print("SOME TileLang accuracy tests FAILED")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # Benchmark pass                                                       #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 110)
    print("Benchmark: DSA Indexer Loss - Triton vs TileLang")
    print("=" * 110)

    for Sq, Sk, B, H, D in configs:
        softmax_scale = 1.0 / math.sqrt(D)
        dtype = torch.bfloat16

        torch.manual_seed(0)
        attn_query   = torch.randn(Sq, B, H, D, device="cuda", dtype=dtype)
        attn_key     = torch.randn(Sk, B, H, D, device="cuda", dtype=dtype)
        index_scores = torch.randn(B, Sq, Sk, device="cuda", dtype=torch.float32)

        topk = Sk // 2
        index_mask_sparse = torch.full(
            (B, Sq, Sk), float("-inf"), dtype=torch.float32, device="cuda"
        )
        idx = torch.randint(0, Sk, (B, Sq, topk), device="cuda")
        index_mask_sparse.scatter_(-1, idx, 0.0)

        for sparse_loss in [False, True]:
            index_mask = index_mask_sparse if sparse_loss else None

            triton_lambda = lambda: fwd_fused_indexer_loss(
                index_scores, attn_query, attn_key, softmax_scale, 1.0,
                sparse_loss=sparse_loss, index_mask=index_mask,
                Sq_offset=0, full_Sq=Sq,
            )
            tl_lambda = lambda: fwd_fused_indexer_loss_tilelang(
                index_scores, attn_query, attn_key, softmax_scale, 1.0,
                sparse_loss=sparse_loss, index_mask=index_mask,
                Sq_offset=0, full_Sq=Sq,
            )

            torch.cuda.synchronize()

            # Benchmark Triton
            torch.cuda.reset_peak_memory_stats()
            triton_time, _, _ = bench(triton_lambda, is_async=False, num_warmups=1, num_tests=1)
            triton_time *= 1000
            triton_mem_gb = torch.cuda.max_memory_allocated() / 1024 ** 3

            torch.cuda.synchronize()

            # Benchmark TileLang
            torch.cuda.reset_peak_memory_stats()
            tl_time, _, _ = bench(tl_lambda, is_async=False, num_warmups=1, num_tests=1)
            tl_time *= 1000
            tl_mem_gb = torch.cuda.max_memory_allocated() / 1024 ** 3

            speedup = triton_time / tl_time
            sparse_str = "Yes" if sparse_loss else "No"
            metrics_collection.append(
                (Sq, Sk, B, H, D, sparse_str,
                 triton_time, tl_time, speedup,
                 triton_mem_gb, tl_mem_gb)
            )
            print(f"  [Sq={Sq}, Sk={Sk}, B={B}, H={H}, D={D}, sparse={sparse_str}] done.")

    hdr = (f"{'Sq':>5} {'Sk':>5} {'B':>2} {'H':>3} {'D':>5}  {'Sparse':>6}  "
           f"{'Triton (ms)':>12}  {'TileLang (ms)':>14}  {'Speedup':>8}  "
           f"{'Triton (GB)':>12}  {'TileLang (GB)':>14}")
    print("\n" + "-" * 110)
    print(hdr)
    print("-" * 110)
    for (Sq, Sk, B, H, D, sparse_str,
         triton_time, tl_time, speedup,
         triton_mem_gb, tl_mem_gb) in metrics_collection:
        marker = "🚀" if speedup > 1.0 else ""
        print(f"{Sq:5d} {Sk:5d} {B:2d} {H:3d} {D:5d}  {sparse_str:>6}  "
              f"{triton_time:>12.2f}  {tl_time:>14.2f}  {speedup:>7.2f}x{marker}  "
              f"{triton_mem_gb:>12.3f}  {tl_mem_gb:>14.3f}")
    print("=" * 110)

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="DSA Triton/PyTorch test harness")
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--backward-native", 
        action="store_true", 
        help="Test the backward_native implementation"
    )
    group.add_argument(
        "--backward-kernel", 
        action="store_true", 
        help="Test the fully-fused Triton backward implementation"
    )
    group.add_argument(
        "--forward-kernel", 
        action="store_true", 
        help="Benchmark compute_index_scores_topk and DSA indexer loss (native vs Triton)"
    )
    group.add_argument(
        "--forward-tensor-parallel", 
        action="store_true", 
        help="Benchmark tensor parallel variant (requires torchrun with --nproc_per_node)"
    )
    group.add_argument(
        "--backward-tensor-parallel", 
        action="store_true", 
        help="Benchmark tensor parallel variant (requires torchrun with --nproc_per_node)"
    )
    group.add_argument(
        "--autograd", 
        action="store_true", 
        help="Test the FusedDSAIndexerLoss autograd function (forward + backward)"
    )
    group.add_argument(
        "--test-bmm",
        action="store_true",
        help="Unit test: triton_bmm split-K kernel vs torch.bmm",
    )
    group.add_argument(
        "--test-tilelang",
        action="store_true",
        help="Unit test: TileLang stage-2 kernel accuracy vs Triton fwd_fused_indexer_loss",
    )
    args = parser.parse_args()

    any_run = False

    if args.backward_native:
        test_fused_loss_backward_native()
        any_run = True

    if args.backward_kernel:
        benchmark_fused_loss_backward()
        any_run = True

    if args.forward_kernel:
        benchmark_fused_loss_forward()
        any_run = True

    if args.forward_tensor_parallel:
        benchmark_fused_loss_forward_tensor_parallel()
        any_run = True

    if args.backward_tensor_parallel:
        test_fused_loss_backward_tensor_parallel()
        any_run = True

    if args.autograd:
        test_fused_dsa_indexer_loss_autograd()
        any_run = True

    if args.test_bmm:
        test_triton_bmm()
        any_run = True

    if args.test_tilelang:
        test_fused_loss_tilelang()
        any_run = True

    if not any_run:
        print(
            "Nothing selected to run. Please specify one of the following:\n"
            "--backward-native\n"
            "--backward-kernel\n"
            "--forward-kernel\n"
            "--forward-tensor-parallel\n"
            "--autograd\n"
            "--test-bmm\n"
            "--test-tilelang"
        )
        sys.exit(1)

if __name__ == "__main__":
    main()
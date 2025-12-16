import torch
import torch.distributed as dist

import triton

import numpy as np

from megatron.core.transformer.experimental_attention_variant.dsa_triton import compute_dsa_indexer_loss_triton
from megatron.core.process_groups_config import ProcessGroupCollection
import megatron.core.parallel_state as parallel_state

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
    # Recompute index_scores (this is the "unfused" part in backward)
    # Trade-off: extra computation vs memory saving
    index_scores = _compute_index_scores(q, weights, k)  # [B, Sq, Sk]

    sq, b, np, hn = query.size()
    sk = key.size(0)

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
    
    # Compute gradient: ∂KL/∂index_scores = -attention_scores / index_scores
    grad_index_scores = -attention_scores / (index_scores + 1e-10)
    grad_index_scores = grad_index_scores * grad_loss * loss_coeff
    
    # Backward through index_scores computation
    # gradient to weights
    def _gradient_from_index_scores():
        scores = torch.einsum('sbhd,tbd->sbht', q.float(), k.float())
        scores_after_relu = torch.relu(scores)
        # gradient to weights
        grad_weights = (grad_index_scores * scores_after_relu).sum(dim=-1)
        # gradient to scores after relu
        grad_scores_after_relu = grad_index_scores * weights.unsqueeze(-1)
        relu_mask = (scores_after_relu > 0).float()
        grad_scores = grad_scores_after_relu * relu_mask
        # gradient to q
        grad_q = torch.einsum('sbht,tbd->sbhd', grad_scores, k.float())
        # gradient to k
        grad_k = torch.einsum('sbht,tbd->sbhd', grad_scores, q.float())

        return grad_q, grad_weights, grad_k
    grad_q, grad_weights, grad_k = _gradient_from_index_scores()

    return grad_q, grad_weights, grad_k

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

        (128, 128, 2, 8, 256, 32),
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
        for sparse_loss in [False, True]:
            native_lambda = lambda: forward_native(q, weights, k, mask, topk, attn_query, attn_key, softmax_scale, loss_coeff, sparse_loss)
            triton_lambda = lambda: compute_dsa_indexer_loss_triton(q, weights, k, attn_query, attn_key, topk, softmax_scale, loss_coeff, mask=mask, sparse_loss=sparse_loss)

            # check correctness
            native_topk_indices, native_indexer_loss, native_kl_per_element, native_index_scores = native_lambda()
            triton_topk_indices, triton_indexer_loss, triton_kl_per_element = triton_lambda()

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
                assert torch.allclose(miss_match_native_values, miss_match_triton_values)

            assert torch.allclose(native_indexer_loss, triton_indexer_loss), "Indexer loss mismatch"
            # assert torch.allclose(native_kl_per_element, triton_kl_per_element), "KL per element mismatch"

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


if __name__ == "__main__":
    # benchmark_compute_index_scores_topk()

    benchmark_tensor_parallel()
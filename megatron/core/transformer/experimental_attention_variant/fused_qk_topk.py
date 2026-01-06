import torch
import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True, # logits computation needs
}


def convert_to_uint16(x):
    hval = T.Cast(T.float16, x)
    bits_uint = T.reinterpret(T.uint16, hval)
    bits_uint = T.if_then_else(x < 0, ~bits_uint & (0xFFFF), bits_uint | (0x8000))
    return bits_uint >> 8


def convert_to_uint32(x):
    bits_uint = T.reinterpret(T.uint32, x)
    bits_uint = T.if_then_else(
        x < 0,
        ~bits_uint & T.Cast(T.uint32, (0xFFFFFFFF)),
        bits_uint | T.Cast(T.uint32, (0x80000000)),
    )
    return bits_uint


@tilelang.jit(pass_configs=pass_configs)
def tl_topk_impl(
    heads,
    index_dim,
    topk,
    num_stages=2,
    threads=512,
    debug=False,
    dtype=T.float8_e4m3fn,
):
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")
    RADIX = 1 << 8
    BLOCK_SIZE = threads
    SMEM_INPUT_SIZE = 4096  # assume the threshold bucket size after first pass is less than 4K

    # logits compute
    block_Q = 1 # restricted
    block_N = 256
    # dtype=T.float8_e4m3fn
    accum_dtype = T.float32
    index_dtype = T.int32

    index_q_shape = [seq_len * heads, index_dim]
    index_k_shape = [seq_len_kv, index_dim]
    index_k_scale_shape = [seq_len_kv]
    logits_shape = [seq_len, topk]

    block_TOPK = topk + block_N

    @T.prim_func
    def tl_topk_kernel(
        # input: T.Tensor[(seq_len, seq_len_kv), accum_dtype],
        topk_index: T.Tensor[(seq_len, topk), index_dtype],
        topk_logits: T.Tensor[(seq_len, topk), accum_dtype],
        starts: T.Tensor[(seq_len), index_dtype],
        ends: T.Tensor[(seq_len), index_dtype],
        # logits compute
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        IndexK: T.Tensor(index_k_shape, dtype),  # type: ignore
        IndexKScale: T.Tensor(index_k_scale_shape, accum_dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        LogitsIdx: T.Tensor(logits_shape, index_dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], accum_dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], index_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], index_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, block_Q), threads=threads) as (bx):
            # logits compute
            index_q_shared = T.alloc_shared([block_Q * heads, index_dim], dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
            index_k_scale_fragment = T.alloc_fragment([block_N], accum_dtype)
            s_shared = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)
            s_reshaped = T.reshape(s_shared, (block_N, block_Q, heads))
            logits = T.alloc_fragment([block_N, block_Q], accum_dtype)
            weights = T.alloc_fragment([block_Q, heads], accum_dtype)

            seq_len_i = bx * block_Q

            cu_k_s_min = T.alloc_var(index_dtype)
            cu_k_e_max = T.alloc_var(index_dtype)

            cu_k_s_min = 2147483647
            cu_k_e_max = -2147483648

            for bq_i in T.serial(block_Q):
                cu_k_s_min = T.min(cu_k_s_min, T.min(CuSeqLenKS[seq_len_i + bq_i], seq_len_kv))
            for bq_i in T.serial(block_Q):
                cu_k_e_max = T.max(cu_k_e_max, T.min(CuSeqLenKE[seq_len_i + bq_i], seq_len_kv))

            s_threshold_bin_id = T.alloc_shared([1], T.int32)
            s_histogram = T.alloc_shared([2, RADIX + 1], T.int32)
            s_num_input = T.alloc_shared([2], T.int32)
            s_input_idx = T.alloc_shared([2, SMEM_INPUT_SIZE], T.int32)

            l_threshold_bin_id = T.alloc_var(T.int32)
            l_new_topk = T.alloc_var(T.int32)
            l_num_input = T.alloc_var(T.int32)
            l_bin_id32 = T.alloc_var(T.int32)
            l_val = T.alloc_var(T.int32)
            l_start_pos = T.alloc_var(T.int32)
            l_out_pos = T.alloc_var(T.int32)

            l_new_topk = topk

            # sync
            tx = T.get_thread_binding()
            copy_done = T.alloc_barrier(arrive_count=512)
            gemm_done = T.alloc_barrier(arrive_count=512)

            T.fill(s_histogram[0, :], 0)
            T.fill(s_num_input[0], 0)

            nbn_i = T.alloc_var(T.int32)
            pos = T.alloc_var(T.int32)
            s_val = T.alloc_var(accum_dtype)
            s_idx = T.alloc_var(index_dtype)
            input_idx = T.alloc_var(T.int32)

            T.copy(IndexQ[seq_len_i * heads, 0], index_q_shared)
            T.copy(Weights[seq_len_i, 0], weights)

            # fill block_TOPK logits
            fill_size = T.min(topk, cu_k_e_max - cu_k_s_min)
            T.barrier_arrive(gemm_done)

            for nbn_i in T.serial(T.ceildiv(fill_size, block_N)):
                T.barrier_wait(gemm_done, nbn_i % 2)

                T.copy(IndexK[cu_k_s_min + nbn_i * block_N, 0], index_k_shared)
                T.copy(IndexKScale[cu_k_s_min + nbn_i * block_N], index_k_scale_fragment)

                if debug and bx == 0 and tx == 0:
                    T.print(nbn_i, "copy done")

                T.barrier_arrive(copy_done)
                T.barrier_wait(copy_done, nbn_i % 2)

                if debug and bx == 0 and tx == 0:
                    T.print(nbn_i, "start gemm")

                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    s_shared,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                if debug and bx == 0 and tx == 0:
                    T.print(nbn_i, "gemm done")

                T.barrier_arrive(gemm_done)

                if debug and bx == 0 and tx == 0:
                    T.print(nbn_i, "pass gemm barrier")

                for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
                    s_reshaped[bn_i, bq_i, h_i] = (T.max(s_shared[bn_i, bq_i * heads + h_i], 0) * weights[bq_i, h_i]) * index_k_scale_fragment[
                        bn_i
                    ]

                T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                # update histogram for topk stage 1
                for s in T.Parallel(block_N):
                    input_idx = cu_k_s_min + nbn_i * block_N + s
                    if input_idx < cu_k_e_max and input_idx >= cu_k_s_min and s < block_N:
                        inval_int16 = convert_to_uint16(logits[s, 0])
                        T.atomic_add(s_histogram[0, inval_int16], 1)

                # store topk logits and index first
                for s in T.Parallel(block_N):
                    if input_idx < cu_k_e_max and input_idx >= cu_k_s_min and cu_k_s_min + nbn_i * block_N + s < fill_size:
                        Logits[bx, cu_k_s_min + nbn_i * block_N + s] = logits[s, 0]
                        LogitsIdx[bx, cu_k_s_min + nbn_i * block_N + s] = cu_k_s_min + nbn_i * block_N + s

            T.sync_threads(1, 512)

            if cu_k_e_max - cu_k_s_min > topk:
                # update topk for each logits block compute
                cu_k_s_min = cu_k_s_min + fill_size
                T.fill(s_histogram[1, :], 0)

                for nbn_i in T.serial(T.ceildiv(cu_k_e_max - cu_k_s_min, block_N)):
                    T.fill(s_num_input[0], 0)

                    # logits compute
                    T.barrier_wait(gemm_done, nbn_i % 2)

                    T.copy(IndexK[cu_k_s_min + nbn_i * block_N, 0], index_k_shared)
                    T.copy(IndexKScale[cu_k_s_min + nbn_i * block_N], index_k_scale_fragment)

                    if debug and bx == 0 and tx == 0:
                        T.print(nbn_i, "copy done")

                    T.barrier_arrive(copy_done)
                    T.barrier_wait(copy_done, nbn_i % 2)

                    if debug and bx == 0 and tx == 0:
                        T.print(nbn_i, "start gemm")

                    T.gemm(
                        index_k_shared,
                        index_q_shared,
                        s_shared,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    if debug and bx == 0 and tx == 0:
                        T.print(nbn_i, "gemm done")

                    T.barrier_arrive(gemm_done)

                    if debug and bx == 0 and tx == 0:
                        T.print(nbn_i, "pass gemm barrier")

                    for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
                        s_reshaped[bn_i, bq_i, h_i] = (T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i]) * index_k_scale_fragment[
                            bn_i
                        ]

                    T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                    # block_Q is restricted to 1
                    for s in T.Parallel(block_N):
                        input_idx = cu_k_s_min + nbn_i * block_N + s
                        if input_idx < cu_k_e_max and input_idx >= cu_k_s_min and s < block_N:
                            inval_int16 = convert_to_uint16(logits[s, 0])
                            T.atomic_add(s_histogram[nbn_i % 2, inval_int16], 1)

                    # maintain s_histogram for the next block
                    T.copy(s_histogram[nbn_i % 2, :], s_histogram[(nbn_i % 2) ^ 1, :])

                    # topk compute

                    # cumsum
                    s_threshold_bin_id[0] = -1
                    T.sync_threads(1, 512)
                    if tx < RADIX:
                        for i in T.serial(8):
                            offset = 1 << i
                            T.sync_threads(3, RADIX)
                            if tx < RADIX - offset:
                                l_val = s_histogram[nbn_i % 2, tx] + s_histogram[nbn_i % 2, tx + offset]
                            T.sync_threads(3, RADIX)
                            if tx < RADIX - offset:
                                s_histogram[nbn_i % 2, tx] = l_val

                        # find threshold bin id
                        T.sync_threads(3, RADIX)
                        if s_histogram[nbn_i % 2, tx] > l_new_topk and s_histogram[nbn_i % 2, tx + 1] <= l_new_topk:
                            s_threshold_bin_id[0] = tx
                    T.sync_threads(1, 512)
                    l_threshold_bin_id = s_threshold_bin_id[0]
                    l_new_topk = l_new_topk - s_histogram[nbn_i % 2, l_threshold_bin_id + 1]
                    T.sync_threads(1, 512)

                    if debug and bx == 0 and tx == 0 and l_threshold_bin_id < 0:
                        T.print(l_threshold_bin_id, "stage 1l_threshold_bin_id < 0")

                    if debug and bx == 0 and tx == 0:
                        T.print(l_new_topk, "l_new_topk 0")

                    # reset counter greater than topk to topk
                    # TODO: check accuracy issue
                    s_histogram[(nbn_i % 2) ^ 1, l_threshold_bin_id] = topk - s_histogram[nbn_i % 2, l_threshold_bin_id + 1]
                    T.fill(s_histogram[(nbn_i % 2) ^ 1, 0 : l_threshold_bin_id], 0)
                    if debug and bx == 0 and tx == 0:
                        T.print(s_histogram[(nbn_i % 2) ^ 1, l_threshold_bin_id], "s_histogram[(nbn_i % 2) ^ 1, l_threshold_bin_id]")

                    # collect previous topk elements with exponent ≥ threshold
                    for s in T.serial(T.ceildiv(topk, BLOCK_SIZE)):
                        T.sync_threads(1, 512)
                        input_idx = s * BLOCK_SIZE + tx
                        if input_idx < topk:
                            bin_id = convert_to_uint16(Logits[bx, input_idx])
                            l_bin_id32 = T.Cast(T.int32, bin_id)
                            if l_bin_id32 > l_threshold_bin_id:
                                # need a pos = T.atomic_add(s_histogram[bin_id32+1], 1)
                                pos = T.atomic_add(s_histogram[nbn_i % 2, l_bin_id32 + 1], 1, return_prev=True)
                                topk_index[bx, pos] = LogitsIdx[bx, input_idx]
                                topk_logits[bx, pos] = Logits[bx, input_idx]

                            elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                                pos = T.atomic_add(s_num_input[0], 1, return_prev=True)
                                s_input_idx[0, pos] = input_idx

                    # collect current block elements with exponent ≥ threshold
                    for s in T.Parallel(block_N):
                        input_idx = cu_k_s_min + nbn_i * block_N + s
                        if input_idx < cu_k_e_max and input_idx >= cu_k_s_min and s < block_N:
                            inval_int16 = convert_to_uint16(logits[s, 0])
                            l_bin_id32 = T.Cast(T.int32, inval_int16)
                            if l_bin_id32 > l_threshold_bin_id:
                                pos = T.atomic_add(s_histogram[nbn_i % 2, l_bin_id32 + 1], 1, return_prev=True)
                                topk_index[bx, pos] = input_idx
                                topk_logits[bx, pos] = logits[s, 0]

                            elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                                pos = T.atomic_add(s_num_input[0], 1, return_prev=True)
                                s_input_idx[0, pos] = input_idx

                    # stage 2: tail pass
                    for round in T.serial(4):
                        if l_new_topk <= 0:
                            T.loop_break()

                        r_idx = round % 2
                        l_start_pos = topk - l_new_topk

                        T.sync_threads(1, 512)
                        T.fill(s_histogram[nbn_i % 2, :], 0)
                        if tx == 0:
                            s_num_input[r_idx ^ 1] = 0
                        T.sync_threads(1, 512)

                        if debug and bx == 0 and tx == 0:
                            T.print(s_num_input[r_idx], "s_num_input[r_idx]")

                        l_num_input = s_num_input[r_idx]
                        for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                            T.sync_threads(1, 512)
                            if s * BLOCK_SIZE + tx < l_num_input:
                                input_idx = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                                if input_idx < cu_k_s_min + nbn_i * block_N:
                                    s_val = Logits[bx, input_idx]
                                else:
                                    s_val = logits[input_idx - cu_k_s_min - nbn_i * block_N, 0]
                                l_bin_id32 = T.Cast(
                                    T.int32, ((convert_to_uint32(s_val) >> (24 - round * 8)) & 0xFF)
                                )
                                T.atomic_add(s_histogram[nbn_i % 2, l_bin_id32], 1)
                        T.sync_threads(1, 512)

                        # cumsum
                        s_threshold_bin_id[0] = -1
                        if tx < RADIX:
                            for i in T.serial(8):
                                offset = 1 << i
                                T.sync_threads(3, RADIX)
                                if tx < RADIX - offset:
                                    l_val = s_histogram[nbn_i % 2, tx] + s_histogram[nbn_i % 2, tx + offset]
                                T.sync_threads(3, RADIX)
                                if tx < RADIX - offset:
                                    s_histogram[nbn_i % 2, tx] = l_val

                            # find threshold bin id
                            T.sync_threads(3, RADIX)
                            if s_histogram[nbn_i % 2, tx] > l_new_topk and s_histogram[nbn_i % 2, tx + 1] <= l_new_topk:
                                s_threshold_bin_id[0] = tx
                        T.sync_threads(1, 512)

                        l_threshold_bin_id = s_threshold_bin_id[0]
                        l_new_topk = l_new_topk - s_histogram[nbn_i % 2, l_threshold_bin_id + 1]
                        T.sync_threads(1, 512)

                        for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                            T.sync_threads(1, 512)
                            if s * BLOCK_SIZE + tx < l_num_input:
                                input_idx = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                                if input_idx < cu_k_s_min + nbn_i * block_N:
                                    s_val = Logits[bx, input_idx]
                                    s_idx = LogitsIdx[bx, input_idx]
                                else:
                                    s_val = logits[input_idx - cu_k_s_min - nbn_i * block_N, 0]
                                    s_idx = input_idx

                                l_bin_id32 = T.Cast(
                                    T.int32, ((convert_to_uint32(s_val) >> (24 - round * 8)) & 0xFF)
                                )

                                if l_bin_id32 > l_threshold_bin_id:
                                    pos = T.atomic_add(s_histogram[nbn_i % 2, l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                                    topk_logits[bx, pos] = s_val
                                    topk_index[bx, pos] = s_idx
                                elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                                    if round == 3:
                                        l_out_pos = T.atomic_add(s_histogram[nbn_i % 2, l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                                        if l_out_pos < topk:
                                            topk_logits[bx, l_out_pos] = s_val
                                            topk_index[bx, l_out_pos] = s_idx
                                    else:
                                        pos = T.atomic_add(s_num_input[r_idx ^ 1], 1, return_prev=True)
                                        s_input_idx[r_idx ^ 1, pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]



                    # dump topk to Logits
                    T.copy(topk_index[bx, :], LogitsIdx[bx, :])
                    T.copy(topk_logits[bx, :], Logits[bx, :])
            else:
                T.copy(LogitsIdx[bx, :], topk_index[bx, :])
                T.copy(Logits[bx, :], topk_logits[bx, :])


    return tl_topk_kernel


def tl_topk(
    q, kv, weights, cu_seqlen_ks, cu_seqlen_ke,
    starts, ends, topk, debug=False, kv_scales=None, input=None,
):
    seq_len, heads, index_dim = q.shape
    seq_len_kv = kv.shape[0]

    topk_indexes = torch.zeros(seq_len, topk, device=q.device, dtype=torch.int32)
    topk_logits = torch.empty([seq_len, topk], device=q.device, dtype=torch.float32)
    logits = torch.empty([seq_len, topk], device=q.device, dtype=torch.float32)
    logits_idx = torch.empty([seq_len, topk], device=q.device, dtype=torch.int32)

    if kv_scales is None:
        kv_scales = torch.ones(seq_len_kv, device=q.device, dtype=torch.float32)

    kernel = tl_topk_impl(heads=heads, index_dim=index_dim, topk=topk, debug=debug, dtype=q.dtype)
    kernel(
        # input, 
        topk_indexes, 
        topk_logits,
        starts, 
        ends,
        # logits
        q.view(seq_len * heads, index_dim),
        kv,
        kv_scales,
        logits,
        logits_idx,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )
    return topk_indexes, topk_logits, logits
from torch._subclasses.fake_utils import output_alias_each_other
import triton
import triton.language as tl
import torch

@triton.jit
def test_kernel(
    x_ptr,
    stride_n: tl.constexpr,
    stride_m: tl.constexpr,
    N: tl.constexpr,
    M: tl.constexpr,
):
    x_ptrs = x_ptr + tl.arange(0, N)[:, None] * stride_n + tl.arange(0, M)[None, :] * stride_m
    x_mask = (tl.arange(0, N) < N)[:, None] & (tl.arange(0, M) < M)[None, :]
    data = tl.load(x_ptrs, mask=x_mask, other=0.0)
    val_range = (data > 3) & (data < 7)
    hist = tl.histogram(data, 32, mask=val_range)
    tl.device_print("hist", hist)


@triton.jit
def convert_to_uint16(x):
    hval = x.cast(tl.float16)
    bits_uint = hval.cast(tl.uint16, bitcast=True)

    # the negative values are flipped, so that the minimum one is 0
    # the positive values are enlarged to greater than any negative value
    bits_uint = tl.where(x < 0, ~bits_uint & (0xFFFF), bits_uint | (0x8000))

    # only keep the sign and the exponent
    bits_uint = bits_uint >> 8
    
    return bits_uint

@triton.jit
def convert_to_uint32(x):
    fval = x.cast(tl.float32)
    bits_uint = fval.cast(tl.uint32, bitcast=True)
    bits_uint = tl.where(
        x < 0, 
        ~bits_uint & tl.full((1,), 0xFFFFFFFF, dtype=tl.uint32), 
        bits_uint | tl.full((1,), 0x80000000, dtype=tl.uint32)
    )
    return bits_uint


@triton.jit
def test_convert_kernel(
    x_ptr,
    uint16_x_ptr,
    out_vals_ptr,
    out_idxs_ptr,
    hist_ptr,
    mask_ptr,
    index_ptr,
    position_ptr,
    s_out_vals_ptr,
    s_out_idxs_ptr,
    s_bin_threshold_ptr,
    s_num_input_ptr,
    TOPK_THRESHOLD: tl.constexpr,
    stride_n: tl.constexpr,
    stride_uint16_x: tl.constexpr,
    stride_on: tl.constexpr,
    stride_oi: tl.constexpr,
    stride_hist: tl.constexpr,
    stride_mask: tl.constexpr,
    stride_index: tl.constexpr,
    stride_son: tl.constexpr,
    stride_soi: tl.constexpr,
    stride_position: tl.constexpr,
    stride_s_bin_threshold: tl.constexpr,
    stride_s_num_input: tl.constexpr,
    N: tl.constexpr,
    UINT16_X_N: tl.constexpr,
    ON: tl.constexpr,
    OI: tl.constexpr,
    HIST_N: tl.constexpr,
    MASK_N: tl.constexpr,
    INDEX_N: tl.constexpr,
    SON: tl.constexpr,
    SOI: tl.constexpr,
    POSITION_N: tl.constexpr,
    S_BIN_THRESHOLD_N: tl.constexpr,
    S_NUM_INPUT_N: tl.constexpr,
):
    # TODO: maybe need BLOCK_SIZE for loading
    x_ptrs = x_ptr + tl.arange(0, N) * stride_n
    x = tl.load(x_ptrs, mask=tl.arange(0, N) < N, other=0.0)
    x_idx = tl.arange(0, N)
    x_dtype = x.dtype

    # # initialize
    # tl.store(s_out_vals_ptr, x, mask=tl.arange(0, N) < N)
    # tl.store(s_out_idxs_ptr, x_idx, mask=tl.arange(0, N) < N)
    # l_new_topk = TOPK_THRESHOLD

    # Stage-1: coarse-grained topk selection
    # consider the first 8 bits
    uint16_x = convert_to_uint16(x)

    # 256 = 2^8
    histogram = tl.histogram(uint16_x.cast(tl.int32), 256)

    cumsum_histogram = tl.cumsum(histogram, reverse=True)

    # bin threshold
    tl.store(hist_ptr + tl.arange(0, 256) * stride_hist, cumsum_histogram, mask=tl.arange(0, 256) < 256)
    offset_histogram = tl.load(hist_ptr + (tl.arange(0, 256) + 1) * stride_hist, mask=tl.arange(0, 256) < 256, other=0)
    threshold_mask = (cumsum_histogram > TOPK_THRESHOLD) & (offset_histogram <= TOPK_THRESHOLD + 1)
    bin_threshold = tl.max(tl.where(threshold_mask, tl.arange(0, 256), -1))
    tl.store(s_bin_threshold_ptr, bin_threshold)

    # greater than threshold, to output
    gt_mask = (uint16_x.cast(tl.int32) > bin_threshold)
    tl.store(mask_ptr + tl.arange(0, N) * stride_mask, gt_mask, mask=tl.arange(0, N) < N)
    tl.store(index_ptr + tl.arange(0, N) * stride_index, x_idx, mask=tl.arange(0, N) < N)
    tl.store(position_ptr + tl.arange(0, 256) * stride_position, cumsum_histogram, mask=tl.arange(0, 256) < 256)
    tl.store(uint16_x_ptr + tl.arange(0, N) * stride_uint16_x, uint16_x, mask=tl.arange(0, N) < N)
    for i in tl.static_range(0, N):
        is_valid = tl.load(mask_ptr + i * stride_mask)
        if is_valid:
            uint16_val = tl.load(uint16_x_ptr + i * stride_uint16_x).to(tl.int32)
            val = tl.load(x_ptr + i * stride_n)
            idx = tl.load(index_ptr + i * stride_index)
            out_ptr_offset = tl.load(position_ptr + (uint16_val + 1) * stride_position, mask=(uint16_val + 1) < 256, other=0)
            tl.store(out_vals_ptr + out_ptr_offset * stride_on, val)
            tl.store(out_idxs_ptr + out_ptr_offset * stride_oi, idx)
            tl.store(position_ptr + (uint16_val + 1) * stride_position, out_ptr_offset + 1, mask=(uint16_val + 1) < 256)
            # tl.device_print("increases by 1: ", out_ptr_offset)

    # equal to threshold, to fine-grained selection
    eq_mask = (uint16_x.cast(tl.int32) == bin_threshold)
    tl.store(mask_ptr + tl.arange(0, N) * stride_mask, eq_mask, mask=tl.arange(0, N) < N)
    tl.store(index_ptr + tl.arange(0, N) * stride_index, x_idx, mask=tl.arange(0, N) < N)
    sout_ptr_offset = 0
    for i in tl.static_range(0, N):
        is_valid = tl.load(mask_ptr + i * stride_mask)
        if is_valid:
            val = tl.load(x_ptr + i * stride_n)
            idx = tl.load(index_ptr + i * stride_index)
            tl.store(s_out_vals_ptr + sout_ptr_offset * stride_son, val.to(x_dtype))
            tl.store(s_out_idxs_ptr + sout_ptr_offset * stride_soi, idx)
            sout_ptr_offset += 1
    tl.store(s_num_input_ptr, sout_ptr_offset)

    # Stage-2: fine-grained topk selection
    l_new_topk = tl.sum(eq_mask)
    if l_new_topk == 0:
        return

    for round in tl.static_range(4):
        if l_new_topk > 0:
            # interleaved buffer
            r_idx = round % 2

            # compute the start position of the current round
            l_start_pos = TOPK_THRESHOLD - l_new_topk

            # clean up buffer
            tl.store(position_ptr + tl.arange(0, 256) * stride_position, tl.zeros((256,), dtype=tl.int32), mask=tl.arange(0, 256) < 256)
            tl.store(s_num_input_ptr + (r_idx ^ 1) * stride_s_num_input, 0)

            l_num_input = tl.load(s_num_input_ptr + r_idx * stride_s_num_input)

            # build current 8-bit histogram
            cur_x = tl.load(s_out_vals_ptr + tl.arange(0, N) * stride_son)
            cur_uint32_x = convert_to_uint32(cur_x)
            cur_8bit_x = ((cur_uint32_x >> (24 - round * 8)) & (0xFF)).cast(tl.int32)
            cur_histogram = tl.histogram(cur_8bit_x, 256)

            # prefix sum
            cur_cumsum_histogram = tl.cumsum(cur_histogram, reverse=True)

            # find new bin threshold
            tl.store(hist_ptr + tl.arange(0, 256) * stride_hist, cur_cumsum_histogram, mask=tl.arange(0, 256) < 256)
            cur_offset_histogram = tl.load(position_ptr + (tl.arange(0, 256) + 1) * stride_position, mask=tl.arange(0, 256) < 256, other=0)
            cur_threshold_mask = (cur_cumsum_histogram > l_new_topk) & (cur_offset_histogram <= l_new_topk + 1)
            cur_bin_threshold = tl.max(tl.where(cur_threshold_mask, tl.arange(0, 256), -1))
            tl.store(s_bin_threshold_ptr, cur_bin_threshold)

            # greater than threshold, to output
            cur_gt_mask = (cur_8bit_x > cur_bin_threshold)
            tl.store(mask_ptr + tl.arange(0, N) * stride_mask, cur_gt_mask, mask=tl.arange(0, N) < N)
            tl.store(index_ptr + tl.arange(0, N) * stride_index, x_idx, mask=tl.arange(0, N) < N)
            tl.store(position_ptr + tl.arange(0, 256) * stride_position, cur_cumsum_histogram, mask=tl.arange(0, 256) < 256)
            tl.store(uint16_x_ptr + tl.arange(0, N) * stride_uint16_x, cur_8bit_x, mask=tl.arange(0, N) < N)
            for i in tl.static_range(0, N):
                is_valid = tl.load(mask_ptr + i * stride_mask)
                if is_valid:
                    bit_val = tl.load(uint16_x_ptr + i * stride_uint16_x).to(tl.int32)
                    val = tl.load(x_ptr + i * stride_n)
                    idx = tl.load(index_ptr + i * stride_index)
                    out_ptr_offset = tl.load(position_ptr + (bit_val + 1) * stride_position, mask=(bit_val + 1) < 256, other=0)
                    # skip the previous topk elements
                    out_ptr_offset += l_start_pos
                    tl.store(out_vals_ptr + out_ptr_offset * stride_on, val.to(x_dtype), mask=out_ptr_offset < N)
                    tl.store(out_idxs_ptr + out_ptr_offset * stride_oi, idx, mask=out_ptr_offset < N)
                    tl.store(position_ptr + (bit_val + 1) * stride_position, out_ptr_offset + 1, mask=(bit_val + 1) < 256)

            # equal to threshold, to fine-grained selection
            cur_eq_mask = (cur_8bit_x == cur_bin_threshold)
            l_new_topk = tl.sum(cur_eq_mask)
            if l_new_topk > 0:
                tl.store(mask_ptr + tl.arange(0, N) * stride_mask, cur_eq_mask, mask=tl.arange(0, N) < N)
                tl.store(index_ptr + tl.arange(0, N) * stride_index, x_idx, mask=tl.arange(0, N) < N)
                if round == 3:
                    for i in tl.static_range(0, N):
                        is_valid = tl.load(mask_ptr + i * stride_mask)
                        if is_valid:
                            bit_val = tl.load(uint16_x_ptr + i * stride_uint16_x).to(tl.int32)
                            val = tl.load(x_ptr + i * stride_n)
                            idx = tl.load(index_ptr + i * stride_index)
                            out_ptr_offset = tl.load(position_ptr + (bit_val + 1) * stride_position, mask=(bit_val + 1) < 256, other=0)
                            # skip the previous topk elements
                            out_ptr_offset += l_start_pos
                            if out_ptr_offset < TOPK_THRESHOLD:
                                tl.store(out_vals_ptr + out_ptr_offset * stride_on, val.to(x_dtype), mask=out_ptr_offset < N)
                                tl.store(out_idxs_ptr + out_ptr_offset * stride_oi, idx, mask=out_ptr_offset < N)
                                tl.store(position_ptr + (bit_val + 1) * stride_position, out_ptr_offset + 1, mask=(bit_val + 1) < 256)
                else:
                    sout_ptr_offset = 0
                    for i in tl.static_range(0, N):
                        is_valid = tl.load(mask_ptr + i * stride_mask)
                        if is_valid:
                            val = tl.load(x_ptr + i * stride_n)
                            idx = tl.load(index_ptr + i * stride_index)
                            tl.store(s_out_vals_ptr + sout_ptr_offset * stride_son, val.to(x_dtype), mask=sout_ptr_offset < N)
                            tl.store(s_out_idxs_ptr + sout_ptr_offset * stride_soi, idx, mask=sout_ptr_offset < N)
                            sout_ptr_offset += 1
                    # tl.store(s_num_input_ptr, sout_ptr_offset)


if __name__ == "__main__":
    # test_radix_topk_1d()
    # test_radix_topk_2d()
    # test_radix_topk_large()
    # benchmark_radix_topk()

    # x = torch.arange(32, device='cuda', dtype=torch.int32)
    x = torch.arange(32, device='cuda', dtype=torch.float32) - 16
    hist = torch.zeros(256, device='cuda', dtype=torch.int32)
    uint16_x = torch.zeros(32, device='cuda', dtype=torch.uint16)
    mask_tensor = torch.zeros(32, device='cuda', dtype=torch.bool)
    index_tensor = torch.zeros(32, device='cuda', dtype=torch.int32)
    position_tensor = torch.zeros(256, device='cuda', dtype=torch.int32)
    out_vals = torch.zeros(256, device='cuda', dtype=x.dtype)
    out_idxs = torch.zeros(256, device='cuda', dtype=torch.int32)
    s_out_vals = torch.zeros(256, device='cuda', dtype=x.dtype)
    s_out_idxs = torch.zeros(256, device='cuda', dtype=torch.int32)
    s_bin_threshold = torch.zeros(1, device='cuda', dtype=torch.int32)
    s_num_input = torch.zeros(2, device='cuda', dtype=torch.int32)
    TOPK_THRESHOLD = 9
    test_convert_kernel[(1, )](
        x_ptr=x,
        stride_n=x.stride(0),
        N=x.shape[0],

        out_vals_ptr=out_vals,
        stride_on=out_vals.stride(0),
        ON=out_vals.shape[0],

        out_idxs_ptr=out_idxs,
        stride_oi=out_idxs.stride(0),
        OI=out_idxs.shape[0],

        s_out_vals_ptr=s_out_vals,
        stride_son=s_out_vals.stride(0),
        SON=s_out_vals.shape[0],

        s_out_idxs_ptr=s_out_idxs,
        stride_soi=s_out_idxs.stride(0),
        SOI=s_out_idxs.shape[0],

        s_bin_threshold_ptr=s_bin_threshold,
        stride_s_bin_threshold=s_bin_threshold.stride(0),
        S_BIN_THRESHOLD_N=s_bin_threshold.shape[0],

        hist_ptr=hist,
        stride_hist=hist.stride(0),
        HIST_N=hist.shape[0],

        TOPK_THRESHOLD=TOPK_THRESHOLD,
        
        mask_ptr=mask_tensor,
        stride_mask=mask_tensor.stride(0),
        MASK_N=mask_tensor.shape[0],

        index_ptr=index_tensor,
        stride_index=index_tensor.stride(0),
        INDEX_N=index_tensor.shape[0],

        position_ptr=position_tensor,
        stride_position=position_tensor.stride(0),
        POSITION_N=position_tensor.shape[0],

        uint16_x_ptr=uint16_x,
        stride_uint16_x=uint16_x.stride(0),
        UINT16_X_N=uint16_x.shape[0],

        s_num_input_ptr=s_num_input,
        stride_s_num_input=s_num_input.stride(0),
        S_NUM_INPUT_N=s_num_input.shape[0],
    )
    print(f"{x=}")
    print(f"{out_vals=}")
    print(f"{out_idxs=}")
    print(f"{s_out_vals=}")
    print(f"{s_out_idxs=}")
    print(f"{hist=}")
    print(f"{position_tensor=}")
    print(f"{uint16_x=}")

    # TODO: streaming radix topk
    # 1. for i-th block, previous histogram with prefix sum, topk values as well as indices, and the bin threshold are known.
    # 2. compute the histogram and its prefix sum for current block
    # 3. add the previous histogram to the current histogram
    # 4. find new bin threshold for topk, and determine the coalescing topk indices
    # 5. fine-grained topk selection for the remaining topk indices
    # 6. maintain the histogram and its belonging values above the new bin threshold (space O(SQ * TOPK))



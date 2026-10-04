
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=71, major=7, regs_per_multiprocessor=262144, max_threads_per_multi_processor=8192, multi_processor_count=16), 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': None, 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8738
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 514
    x1 = (xindex // 514)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 257, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (257*x1)), tmp4 & xmask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x2), tmp4 & xmask, other=0.0).to(tl.float32)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp5 * tmp7
    tmp9 = 1.0
    tmp10 = tmp9 - tmp7
    tmp11 = tmp6 * tmp10
    tmp12 = tmp11 + tmp9
    tmp13 = tmp8 * tmp12
    tmp14 = tl.load(in_ptr1 + (257 + x2), tmp4 & xmask, other=0.0).to(tl.float32)
    tmp15 = tmp13 * tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp4, tmp15, tmp16)
    tmp18 = tmp0 >= tmp3
    tmp19 = tl.full([1], 514, tl.int64)
    tmp20 = tmp0 < tmp19
    tmp21 = tl.load(in_ptr0 + ((-257) + x0 + (257*x1)), tmp18 & xmask, other=0.0).to(tl.float32)
    tmp22 = tl.load(in_ptr1 + ((-257) + x2), tmp18 & xmask, other=0.0).to(tl.float32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tl.sigmoid(tmp23)
    tmp25 = tmp23 * tmp24
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp21 * tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp18, tmp27, tmp28)
    tmp30 = tl.where(tmp4, tmp17, tmp29)
    tl.store(out_ptr0 + (x2), tmp30, xmask)

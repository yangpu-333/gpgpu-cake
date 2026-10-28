# BI-V150 scale/mask/softmax前后向融合

实测日期2026-10-28。实现、CPU参考、执行器与实验契约均为实测冻结字节；相应SHA见contract.json及报告附录。结果见[GitHub首页融合附录](../../../README.md#appendix-fusion)。

- 输入：连续CUDA BF16/FP32 `[B,H,S,S]`，S≤4096；目标 `[1,14,S,S]`、S2048/4096。
- FP32缩放/输出，mask有限填充值-10000；支持因果、广播和非连续bool mask。全遮罩行输出均匀分布，输入梯度为0。
- 前向一行一个Triton program；反向保留FP32乘积、规约、mask置零、缩放与输入dtype转换边界。未接入dropout或重计算。
- 严格确定性、原数值及资源保护保持。两次融合数学kernel之外，Torch输出分配产生两次保护性填充，总计4次GPU启动。文件最初“两次启动”描述仅对数学kernel成立。
- `fusion_worker.py`是原冻结执行器。原总启动数2断言失败记录保留；`../softmax_fusion_trace_20261028_v2/`修正观察器对实际4事件的精确检查，随后`../softmax_fusion_trace_20261028_v3/`按用户明确指令将仅补trace的32GiB保守启动线改为16GiB。原实时NUMA16GiB、ownRSS8GiB、host与数值保护未变。不得把原失败目录当作全阶段成功。

这些脚本是当前已准入BI-V150环境的执行记录，包含该环境的路径/UUID/锁及内存保护；不应在其他共享环境直接启动。核心算子可单独导入：

```python
from fused_scale_mask_softmax import fused_scale_mask_softmax
# x: CUDA BF16/FP32 [B,H,S,S]; mask=None表示因果mask
y = fused_scale_mask_softmax(x, 1 / (128 ** 0.5))
dx, = torch.autograd.grad(y, x, upstream_float32)
```

正确性、边界负例与配对计时方法、全部关键误差、逐轮原始计时及trace SHA均折叠保存在首页附录。完整raw trace继续保留在实验原存储；GitHub发布的是事件摘要与SHA。

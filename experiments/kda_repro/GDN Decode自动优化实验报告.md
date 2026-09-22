# GDN Decode 自动优化实验报告

## 实验目的

验证自动优化流程能否从普通 PyTorch 算子实现出发，自动选择 GPU 优化方案，并生成通过正确性
和性能验证的 Triton 候选。

## 实验方法

实验对象为 KDA 公开任务中的 GDN Decode。起点是依据官方 PyTorch reference 整理的普通
eager PyTorch 实现，不包含 Triton 内核、算子融合或厂商融合库。

自动优化流程如下：

1. Agent读取任务约束和普通 PyTorch 基线，分析计算与显存访问；
2. Agent选择单 Triton 内核融合、`ROWS=4`、4 warps 和递推化简；
3. 控制器将结构化设计转换为受限的 V100 Triton 模板；
4. 系统自动完成静态检查、冒烟验证、54组官方 workload 正确性验证和配对计时；
5. 达到正确性与性能门槛后，候选自动晋级。

## 实验结果

测试设备为 NVIDIA Tesla V100-PCIE-32GB。

| 项目 | 结果 |
|---|---:|
| 普通 PyTorch 配对基线 | `0.256053 ms` |
| Agent Triton 候选 | `0.023308 ms` |
| 加速比 | **`10.985×`** |
| 正确 workload | `54/54` |
| 候选胜出 | `54/54` |
| 额外分支验证 | `3/3` |
| 最大 output 绝对误差 | `7.6294e-6` |
| 最大 state 绝对误差 | `5.7220e-6` |

## 结论

本实验确认：把普通 PyTorch 基线交给结构化 Agent 优化流程后，可以通过算子融合、调度选择和
等价计算变换获得约 **10.99倍** 的流程内性能提升，并保持全部测试正确。

该结果中的 Agent负责选择优化策略和硬件调度，控制器负责将设计转换为经过审查的 Triton 模板、
编译、测试和决定是否晋级；并非模型从空白独立写出整份 Triton 内核。若改用已经较强的 Triton
实现作为基线，同一候选的额外提升约为 `0.46%`，两种性能口径应分开报告。

详细结果及原始证据见
[`AGENT_ORDINARY_BASELINE_V100_RESULT.md`](AGENT_ORDINARY_BASELINE_V100_RESULT.md)和
[`evidence/agent-v100/decode-ordinary-structured`](evidence/agent-v100/decode-ordinary-structured)。

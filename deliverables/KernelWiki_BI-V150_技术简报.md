# KernelWiki 天数平台适配技术报告

工作范围：2026年9月30日至10月11日｜整理日期：2026年10月13日

本项目在**天数智芯 BI-V150**上，使用 **Claude Code CLI 加载适配版 KernelWiki skill**，对 Megatron 训练中的**交叉熵、全局梯度范数、RMSNorm、Dense SwiGLU、Attention 和 MoE 带权 SwiGLU**开展优化实验。原有 NVIDIA Hopper／Blackwell 知识保留，天数结论以独立实测为依据。

**主要结果：** 三组已验收的 BF16 模拟训练配置，相对同卡、同模型及精度的原生 Megatron 后端，组合吞吐分别提升 **8.13%、9.34% 和13.96%**。部分算子用 Triton 实现，其余采用 PyTorch、C++/CUDA 或 TorchScript 优化。

项目仓库：[GitHub · gpgpu-cake](https://github.com/yangpu-333/gpgpu-cake)

## 算子与实现方式

| 算子或训练热点 | 已验证的实现方式 |
|---|---|
| 交叉熵 CE | BF16 方案保留原生前向，采用 Triton 反向融合 |
| 全局梯度范数 | 保留 PyTorch 范数计算，合并 GPU 到 CPU 的标量传输 |
| RMSNorm | C++/CUDA 局部点运算融合，保留原生归约和梯度图 |
| Dense SwiGLU | Triton 前后向点运算，复用已编译启动入口 |
| Attention | TorchScript 减少调用开销，复用原生融合 softmax 和 GEMM |
| MoE 带权 SwiGLU | Triton 前后向与路由权重梯度计算，复用已编译入口 |

## 训练性能

下表中的 GPT／MoE 是训练验证模型，各模型分别比较原生后端和对应的组合优化版本。

| 训练配置 | 原生毫秒/步 | 优化毫秒/步 | 吞吐提升 |
|---|---:|---:|---:|
| 4层 MoE，4专家、序列256 | 76.20–77.48 | 70.57–71.09 | **+8.13%** |
| 4层 MoE，8专家、序列512 | 112.99–114.00 | 103.42–103.99 | **+9.34%** |
| 4层 Dense GPT，序列128 | 33.15–34.15 | 29.32–29.93 | **+13.96%** |

8专家模型每步从约113–114毫秒缩短到约104毫秒，同样时间内所测训练区间可处理的数据量增加约9%。MoE 采用 **CE＋梯度范数＋带权 SwiGLU**组合；Dense 采用 **CE＋梯度范数＋SwiGLU＋Attention**组合。不同配置的收益分别成立，不能相加。

耗时为三个种子的范围，吞吐变化按配对比值的几何平均计算。每进程运行120步，排除前20步；原生 GPU 同步后的计时覆盖前向、反向和 SGD 更新，排除启动、checkpoint 保存与评估。

## 验证范围与下一步

三步训练核对 loss、全部参数梯度、BF16 权重和 FP32 主权重，最新两组 MoE 被比较张量逐位一致。384份新轮原件已归档并校验 SHA256；仓库重建、KDA 项目级加载及 CI 通过，包含162项知识库测试和8项新增门禁测试。

结果范围限于**单卡 BI-V150、CoreX4.2、厂商 Torch2.4.1／Triton2.1、模拟数据和 SGD**，尚无生产数据、AdamW、多卡或 SFT/RL 收益。专家 GEMM 保持原生，Attention 复用原生融合 softmax。下一步评估专家 GEMM／GroupedGEMM，并扩展实际训练配置。

## Git 仓库与实验依据

- [GitHub 项目仓库](https://github.com/yangpu-333/gpgpu-cake)
- [详细报告与实验依据](https://github.com/yangpu-333/gpgpu-cake/blob/main/experiments/kernelwiki_iluvatar/PROJECT_REPORT_0930.md)
- [已通过的 Linux CI](https://github.com/yangpu-333/gpgpu-cake/actions/runs/38150991570)

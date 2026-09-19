# KDA 官方 GDN Prefill：V100 适配复现结果

2026-09-19 在 Tesla V100-PCIE-32GB、PyTorch 2.5.1+cu121、Triton 3.1.0 上完成。

## 复现范围

数据固定为 Hugging Face `flashinfer-ai/mlsys26-contest` 修订
`5e832ce88dd1013032b22a0e942979d7d2769cc3`。验证直接执行下载定义中的原始 reference，
覆盖 `gdn_prefill_qk4_v8_d128_k_last` 全部100个官方 workload：总序列长度6–8192，
序列数1–57。另外检查了空序列、`state=None`、已有 state 和 `scale=0` 默认缩放分支。

候选按时间维严格顺序执行 GDN 递推，只在 V 维使用 ROWS=4/8/16 三种并行调度。
外部输入输出保持 BF16，state 保持 FP32；V100 上使用整数位转换读写 BF16、内部使用FP32计算。

## 正确性结果

- 100/100 个官方 workload 全部完成。
- ROWS=4/8/16 共300次候选验证全部正确，没有候选被拒绝。
- 两个额外可选参数/空序列分支均通过。
- 最大 output 绝对误差为 `1.2207e-4`。
- 最大 new_state 绝对误差为 `6.5327e-5`。
- 比较阈值固定为 atol=rtol=0.01，未为候选放宽。

## 候选自身计时

| 调度 | 100组几何平均 | 最小 | 最大 | 单组获胜数 |
|---|---:|---:|---:|---:|
| ROWS=4 | 0.3383 ms | 0.0161 ms | 8.0973 ms | 47 |
| ROWS=8 | 0.3179 ms | 0.0161 ms | 8.3849 ms | 43 |
| ROWS=16 | 0.3380 ms | 0.0167 ms | 9.1591 ms | 10 |

逐 workload 选择最快调度后的几何平均为0.3096 ms，比固定 ROWS=8 再降低约2.67%。
16组长度8192 workload 的获胜调度几何平均为5.2617 ms，其余84组为0.1805 ms。

原始结果保存在 `evidence/official-v100/prefill-full.json`，包含每组输入轴、三种调度的7次
原始计时、误差、峰值分配、设备、数据修订和候选源码哈希。报告源码哈希已与仓库中的
`official_v100/gdn_prefill.py` 核对一致。

## 解释边界

官方 Python reference 含逐时间步循环，本次没有把它作为性能基线，以免产生没有意义的巨大加速比。
表中数字只比较三个自编 Triton 调度，是 V100 上的候选选择证据，不是 B200 竞赛成绩。
性能计数器仍被 `ERR_NVGPUCTRPERM` 拒绝，因此尚无 ncu 指标支撑更细的硬件瓶颈判断。

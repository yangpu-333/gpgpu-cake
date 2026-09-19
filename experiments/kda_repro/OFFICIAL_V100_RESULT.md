# KDA 官方 GDN Decode：V100 适配复现结果

2026-09-18 在 Tesla V100-PCIE-32GB、PyTorch 2.5.1+cu121、Triton 3.1.0 上完成。

## 复现范围

数据固定为 Hugging Face `flashinfer-ai/mlsys26-contest` 修订
`5e832ce88dd1013032b22a0e942979d7d2769cc3`。验证直接执行下载定义中的原始 reference，
覆盖 `gdn_decode_qk4_v8_d128_k_last` 的全部 54 个官方 workload，以及 `state=None`、
`scale=None` 和 `scale=0` 三个额外分支。未读取最终竞赛解答仓库。

候选保留 BF16 外部契约和 FP32 state。由于 V100 没有原生 BF16 算术，自编 Triton kernel
以整数位转换读写 BF16，内部使用 FP32。比较阈值为 atol=rtol=0.01；任何未通过候选均不得计时。

## 结果

- 批量 PyTorch 候选和 ROWS=1/4/8/16 四个 Triton 调度在 54/54 workload 上均通过两个输出的逐元素比较。
- 最终获胜调度中，ROWS=8 赢得 47 组，ROWS=4 赢得 7 组。
- 相对批量 PyTorch 表达式，逐 workload 获胜调度的几何平均加速为 11.68×，范围 8.68×–16.99×。
- 最大 output 绝对误差为 `7.6294e-6`，最大 new_state 绝对误差为 `5.7220e-6`。
- 最大额外显存记录：批量 PyTorch 135,532,544 字节，获胜 Triton 调度 33,685,504 字节。

| batch | workload 数 | 加速比范围 | 组内几何平均 |
|---:|---:|---:|---:|
| 1 | 10 | 16.37×–16.99× | 16.65× |
| 4 | 8 | 15.67×–16.96× | 16.35× |
| 8 | 7 | 11.38×–11.75× | 11.63× |
| 16 | 7 | 10.28×–10.56× | 10.42× |
| 32 | 7 | 9.61×–9.78× | 9.71× |
| 48 | 7 | 8.98×–9.27× | 9.04× |
| 64 | 8 | 8.68×–8.78× | 8.74× |

原始计时、误差、显存、环境、数据修订和代码哈希保存在
`evidence/official-v100/decode-full.json`。计时采用 CUDA Graph，每图 10 次调用、7 次交替顺序采样。

## 解释边界

这是官方任务语义与官方真实输入上的 V100 适配实验，不是官方 B200 evaluator 的验收结果。
基线是批量 PyTorch 表达式，不是 FlashInfer 的 B200 优化实现；因此 11.68× 只能说明本候选在这张
V100 上减少了中间张量和 kernel 启动，不能作为与论文或竞赛成绩直接对比的数字。性能计数器仍受
`ERR_NVGPUCTRPERM` 限制，尚无 ncu 硬件瓶颈数据。

GDN Prefill 后续已在同一张V100上完成全部100组官方输入验证，结果见
`PREFILL_V100_RESULT.md`。DSA 和 MoE 的官方高性能路径分别依赖 FP8 与
DeepGEMM/更新架构，V100 只能做语义适配，不能对齐官方目标硬件性能。

## 一次性复跑

```bash
cd ~/gpgpu-cake
git pull --ff-only origin main
KDA_CA_BUNDLE="$HOME/.local/share/ca-certificates/scholar-git-ca-bundle.pem" \
  bash experiments/kda_repro/scripts/run_official_v100.sh all
```

结果默认写入 `$HOME/kda-repro/runs`，不会覆盖旧证据。

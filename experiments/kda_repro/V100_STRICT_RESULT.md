# V100 KDA 通用流程：严格复核

2026-09-18 在 Tesla V100-PCIE-32GB、PyTorch 2.5.1+cu121、Triton 3.1.0 上完成。

## 已完成

- 发现并修复旧候选权重梯度不匹配。反向重建参考计算图，保留 FP16 舍入边界；只优化前向。
- 六种形状分别校验两个前向输出，以及 normalized-only、residual-only、combined 三种随机上游梯度对应的输入、残差、权重梯度。与参考逐元素比较，atol=rtol=0.002。
- 基线和候选显式选择。基准执行前先验证；计时使用每图 100 次前向、9 组交替测试，保存全部样本。

| 形状 | 基线 μs | 候选 μs | 加速比 |
|---|---:|---:|---:|
| 1×33 | 18.900 | 2.181 | 8.67× |
| 7×127 | 23.676 | 2.336 | 10.13× |
| 64×768 | 25.889 | 3.228 | 8.02× |
| 64×1024 | 26.769 | 3.300 | 8.11× |
| 64×4096 | 40.260 | 4.580 | 8.79× |
| 512×4096 | 134.340 | 23.979 | 5.60× |

原始证据：`evidence/v100-strict/strict-v3.json`。结果针对重复固定输入的 CUDA Graph 微基准，基线为未融合 PyTorch 表达式；不代表训练性能、冷缓存性能或相对已有优化库的优势。

## 复跑

```bash
cd ~/gpgpu-cake
git pull --ff-only origin main
cd experiments/kda_repro/v100_residual_rmsnorm
~/kda-repro/.venv/bin/python reproduce.py --output "$HOME/kda-repro/runs/strict-$(date -u +%Y%m%dT%H%M%SZ).json"
```

## 尚未完成

`nvcc`、`ncu` 均已安装，但普通用户与 sudo 的 ncu 采集都被 `ERR_NVGPUCTRPERM` 拒绝。
日志保存在 `evidence/v100-strict/ncu-attempt.log`。需平台管理员授权 GPU 性能计数器后才能补齐 profiling；尚未形成硬件瓶颈结论。

本次跑通的是 KDA 通用任务闭环（契约、计划、候选、验证、计时、证据），不是官方 B200 FlashInfer 竞赛结果复现。
当前测试只覆盖连续 FP16 输入、固定 epsilon 和随机数据，未覆盖所有输入域、二阶梯度或模型集成。

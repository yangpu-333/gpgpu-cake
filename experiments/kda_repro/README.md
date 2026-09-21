# KDA V100 复现入口

当前计划：[Path4：KDA 复现与国产 GPU 迁移简要计划](PATH4_PLAN.md)。
当前环境、五项任务覆盖、Agent 结果、完成度计算和后续验收条件汇总在
[KDA 复现工作详细进展报告](KDA_REPRODUCTION_PROGRESS_REPORT.md)。

首个模型生成候选已经完成 V100 全闭环，见
[KDA Agent 首轮 V100 闭环结果](AGENT_V100_RESULT.md)。候选通过 54/54 组验证；约 1.10% 的
首次跨运行改善经三次同进程交替复测后修正为约 0.46%，低于1%晋级门槛。
[KDA Agent GDN Prefill V100 结果](AGENT_PREFILL_V100_RESULT.md)完成100/100组验证；模型候选经
三轮同进程配对复测取得1.454%综合改善，超过1%门槛并晋级。启动与复核方式见
[Agent 接入说明](AGENT_SETUP.md)。

官方 GDN Decode 的 V100 适配结果见 [OFFICIAL_V100_RESULT.md](OFFICIAL_V100_RESULT.md)：
固定官方数据版本，54/54 个真实 workload 和 3 个额外分支通过，原始证据已归档。
官方 GDN Prefill 结果见 [PREFILL_V100_RESULT.md](PREFILL_V100_RESULT.md)：100/100 个真实
workload、三种调度和两个额外分支全部通过。
五项官方任务逐项硬件可行性见 [OFFICIAL_TASK_AUDIT.md](OFFICIAL_TASK_AUDIT.md)。
自动优化 API 的配置与启动见 [AGENT_SETUP.md](AGENT_SETUP.md)。不需要额外安装独立 Agent；
仓库内控制器负责候选生成、隔离验证、计时、晋级和证据记录。

最新严格复核见 [V100_STRICT_RESULT.md](V100_STRICT_RESULT.md)：修复旧梯度错误后，六种形状的前向及参考梯度比较已通过，原始重复计时数据已归档。ncu 计数器采集仍需平台授权。

当前目标是复现 Kernel Design Agents（KDA）的公开工作流，而不是继续扩展
BI-V150 上的 CAKE/Megatron 原型。KDA 是一个“定义任务 → 实现候选 → 正确性验证
→ 基准测试 → Nsight Compute 分析 → 记录晋级决定”的智能体工程循环。

本目录不包含 KDA、FlashInfer 或 DeepGEMM 的第三方源码。它们体积大、更新频繁，
且官方要求 KDA 工作流仓库、任务工作区、最终解答快照彼此隔离。`source-lock.json`
记录本次准备时观察到的上游提交；`prepare_kda_sources.sh` 在 V100 容器的持久目录
中按该提交下载源码。

## 硬件边界

- 先用一张 NVIDIA V100 跑通工作流、依赖、数据集、正确性验证和普通 CUDA/Triton
  候选。
- V100 的结果不能同 B200 竞赛成绩直接比较。官方竞赛复现指定 B200 兼容编译路径；
  要对齐该结果，后续仍须在 B200 上重跑。
- BI-V150 可用于 KDA 通用工作流的国产适配；官方竞赛的 NVIDIA CUDA、FlashInfer 和
  DeepGEMM 路径需另行处理，不能直接视为可在天数运行。

## V100 容器首次执行

先拉取本项目并执行主机检查：

```bash
cd /home/huids25/gpgpu-cake
git pull --ff-only origin main
bash experiments/kda_repro/scripts/check_nvidia_host.sh
```

检查输出应确认 NVIDIA GPU、驱动、CUDA 工具链与 `ncu` 是否可见。随后下载公开源代码到
`/home/huids25/kda-repro`：

```bash
bash experiments/kda_repro/scripts/prepare_kda_sources.sh /home/huids25/kda-repro
```

当前容器若经由 Scholar Verify HTTPS 代理访问互联网，先设置 Git 专用 CA 包再运行：

```bash
export GIT_SSL_CAINFO="$HOME/.local/share/ca-certificates/scholar-git-ca-bundle.pem"
bash experiments/kda_repro/scripts/prepare_kda_sources.sh /home/huids25/kda-repro
```

建立 V100 的通用 KDA PyTorch 运行时：

```bash
KDA_CA_BUNDLE="$HOME/.local/share/ca-certificates/scholar-git-ca-bundle.pem" \
  bash experiments/kda_repro/scripts/setup_v100_torch.sh
```

脚本将 PyTorch 2.5.1 CUDA 12.1 wheel 安装到 `~/kda-repro/.venv`，并验证 V100
的 CUDA 可见性。它不会安装 B200 专用的竞赛依赖。

脚本不会下载 `mlsys2026-flashinfer-contest-solution`。官方明确规定：该最终解答仓库
仅用于最终结果验证，不能作为重新运行智能体优化流程的输入。

## 官方 V100 适配复跑

运行一次长脚本即可下载固定版本的 Decode/Prefill 输入，并顺序执行验证：

```bash
KDA_CA_BUNDLE="$HOME/.local/share/ca-certificates/scholar-git-ca-bundle.pem" \
  bash experiments/kda_repro/scripts/run_official_v100.sh all
```

可以把最后一个参数改成 `decode` 或 `prefill` 只跑单项。结果默认保存在
`$HOME/kda-repro/runs`，每次使用 UTC 时间戳命名，不覆盖已有结果。

## B200 后续顺序

1. 在 `release/mlsys2026-flashinfer-contest` 中依照官方文档建立锁定的 Python 3.12、
   PyTorch、Triton、FlashInfer 与 DeepGEMM 环境，并下载 trace 数据。
2. 先运行 packed `solution.json` 的验证链路，确认 evaluator 可以工作。
3. 在 `workspaces/flashinfer-bench-starter-kit` 中建立全新任务工作区；不要把最终
   解答源码复制进去。
4. 先选择 DSA 任务，再依次尝试 GDN 和 FP8 MoE。每个候选必须保存正确性、计时、
   profile 和保留/淘汰原因。

参考：[KDA 工作流](https://github.com/NVlabs/kda)、[官方竞赛复现说明](https://github.com/mit-han-lab/mlsys2026-flashinfer-contest/blob/main/docs/reproduction.md)。

V100 上已完成的通用流程结果见 [V100_INITIAL_RESULT.md](V100_INITIAL_RESULT.md)。

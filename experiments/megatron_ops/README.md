# Megatron-LM 五算子实验入口

本目录对应 [五个训练算子实验计划](../../research/notes/五个训练算子实验计划.md)。目标框架固定为 Megatron-LM `5be9626709af2722333bf54797c954c09edeada3`，目标设备为 BI-V150。

当前第一批代码只完成以下工作：

- 验证 Megatron 源码提交、GPU、PyTorch 和 Triton 环境；
- 运行已有 CPU 单元测试；
- 在 BI-V150 上尝试 Triton 加法和矩阵乘法，验证编译、正确性和计时链路；
- 以 JSON 保存证据，不覆盖旧结果。

这里的加法和矩阵乘法是工具链冒烟测试，不是五个训练算子的完成情况。目前尚未实现五个算子的候选内核、Halide IR 规范化或 Poly 候选生成。

## 1. 更新项目

如果项目仍在私有目录：

```bash
cd /private/gpgpu-cake
git pull --ff-only origin main
```

如果已经移动到共享目录，则使用：

```bash
cd /share/gpgpu-cake
git pull --ff-only origin main
```

## 2. 定位 Megatron-LM

使用实际能够运行训练的 Megatron 源码目录。若不清楚位置，可以先查找：

```bash
find /private /share -maxdepth 5 -type d -name Megatron-LM -print 2>/dev/null
```

设置路径并检查提交：

```bash
export MEGATRON_ROOT=/实际路径/Megatron-LM
git -C "$MEGATRON_ROOT" rev-parse HEAD
```

必须输出：

```text
5be9626709af2722333bf54797c954c09edeada3
```

若提交不一致，先保留现有工作，不执行 `reset --hard`。应另建固定提交的 checkout 或 worktree。

## 3. 运行第一轮单卡验证

在项目根目录执行：

```bash
bash experiments/megatron_ops/scripts/run_first_validation.sh "$MEGATRON_ROOT"
```

脚本通过 `--device 0` 选择当前可见设备中的第一张卡，不会改写平台注入的 `IX_VISIBLE_DEVICES`。在专家环境或 Kubernetes Pod 中不要手工覆盖该变量；它可能承载宿主机设备到容器设备的映射。只有确认平台文档要求时才修改设备可见性。

脚本按顺序执行环境检查、9 项 CPU 单元测试、GPU 环境探测、向量加法和小矩阵乘法。加法失败时会停止，不继续尝试矩阵乘法，以免在失效的设备上下文上产生误导结果。

结果目录形如：

```text
experiments/megatron_ops/results/first-20260915T120000Z/
```

结果目录已被 Git 忽略。不要把运行生成的 JSON、日志或可能包含机器路径的环境数据提交到仓库。

## 4. 查看结果

查找最新一轮：

```bash
LATEST="$(ls -dt experiments/megatron_ops/results/first-* | head -n 1)"
echo "$LATEST"
ls -lh "$LATEST"
cat "$LATEST/megatron-environment.json"
cat "$LATEST/basic-probe.json"
cat "$LATEST/gpu-add.json"
cat "$LATEST/gpu-matmul.json"
```

正常情况下，关键状态应为：

- `megatron-environment.json`：`ready_for_first_validation`
- `basic-probe.json`：`probe_complete`，且 `ready_for_gpu_attempt` 为 `true`
- `gpu-add.json`：`gpu_check_passed`
- `gpu-matmul.json`：`gpu_check_passed`

`speedup_vs_pytorch` 小于 1 不是脚本失败，只表示当前手写候选比 PyTorch 基线慢。第一轮只判断实验设施是否可信，不据此宣称算子优化收益。

如果脚本中途停止，已经生成的 JSON 会保留。查看其中的 `status`、`error_type` 和 `error`，不要安装通用 CUDA、cuDNN 或 PyPI Triton 覆盖厂商环境。

若 GPU 脚本在产生候选记录前失败，更新仓库后运行一次隔离诊断：

```bash
git pull --ff-only origin main
bash experiments/megatron_ops/scripts/run_device_diagnostics.sh \
  > /private/gpgpu-device-diagnostics.log 2>&1
```

诊断会在独立进程中依次测试设备信息、显存分配、CPU/GPU 双向拷贝、PyTorch 加法、GPU event、Triton target，以及 `num_warps=1/4` 的最小 Triton 加法。每项设置 `CUDA_LAUNCH_BLOCKING=1`，因此一项失败不会污染下一项，报告会给出精确阶段和 traceback。它只读取环境和执行小计算，不安装或切换编译器。

## 5. 下一轮需要的输入

第一轮通过后，需要提供：

1. 四个 JSON 和 `unit-tests.log` 的结果；
2. 实际可运行的 Megatron 启动命令或配置文件；
3. Dense GPT 与 MoE GPT 各自使用的模型规模、序列长度、精度及并行参数。

随后接入真实形状捕获，并先实现 SwiGLU、Residual Add RMSNorm 的前向和反向正确性/计时入口。Cross Entropy 的完整验证需要 `torchrun --nproc_per_node=2` 的 TP=2 实验；MoE Grouped GEMM 后续也要补双卡 expert-parallel 路径。

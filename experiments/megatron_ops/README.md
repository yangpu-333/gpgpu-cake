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

## 5. 显存分配失败时的一键重试

若 `torch.empty(..., device="cuda")` 已经报 `operation not supported`，可以运行进一步的运行时对照。一次复制执行以下整段即可，任务在后台运行：

```bash
cd /private/gpgpu-cake &&
git pull --ff-only origin main && {
  LOG="/private/gpgpu-runtime-$(date -u +%Y%m%dT%H%M%SZ).log"
  nohup bash experiments/megatron_ops/scripts/run_runtime_diagnostics.sh \
    > "$LOG" 2>&1 < /dev/null &
  echo "PID: $!"
  echo "日志: $LOG"
}
```

通常几分钟；每个最小测试最多等待 90 秒。日志持续输出 `Trying`、`PASS/FAIL` 和失败阶段；无需等整套结束才能查看。重新登录后查看最新日志：

```bash
LOG="$(ls -t /private/gpgpu-runtime-*.log | head -n 1)"
cat "$LOG"
```

脚本自动比较：

1. 当前配置（增加同步报错和 C++ 调用栈）；
2. 普通分配器：`backend:native,expandable_segments:False`；
3. 上述配置加 `PYTORCH_NO_CUDA_MEMORY_CACHING=1`；
4. 如果存在已配置的 CoreX 目录或 `/usr/local/corex`，临时优先使用其中的运行库，再做以上三种对照。

每次都启动新进程，测试显存分配、填充、加法、拷回 CPU 和数值检查。通过的设置会用另一个新进程复测。若有复测通过的设置，脚本会自动用该设置继续运行 PyTorch/Triton 的完整设备诊断。成功只表示对应测试通过，仍不代表五个训练算子完成或取得性能收益。

所有设置只作用于测试子进程。脚本保留 `IX_VISIBLE_DEVICES`、`CUDA_VISIBLE_DEVICES` 和 `LD_PRELOAD`，不安装包、不写 shell 配置、不修改驱动。默认测试当前可见设备 0；脚本后加 `1` 可选择设备 1。

结果保存在忽略提交的 `results/runtime-*/`，包括逐步更新的 `runtime-diagnostics.json`、简短的 `summary.log`，以及成功后生成的 `runtime-diagnostics-device.json`。JSON 记录分配器参数、真实加载的动态库、`ixsmi` 和 `ldd` 输出、失败阶段及 C++/Python 错误。退出码 0 只表示诊断流程完成；判断 GPU 是否可用要看 `confirmed_variants` 和后续 `diagnostics_passed`。

判断原则：分配器对照通过，说明存在可行的进程配置，仍需确认具体不兼容接口；库路径对照通过，再结合实际加载路径分析。若全部失败，日志用于进一步定位。`ixsmi` 缺失符号说明该进程存在符号解析问题，不能单独证明它与 PyTorch 分配失败同源，也不能证明硬件损坏。环境变量 `COREX_ROOT` 未设置同样不等于 CoreX 未安装。

依据：[PyTorch 2.7.1 分配器实现](https://github.com/pytorch/pytorch/blob/v2.7.1/c10/cuda/CUDACachingAllocator.cpp)中，不同分配模式调用的底层接口有差别，关闭缓存可直接走 `cudaMalloc`；[PyTorch 2.7 调试变量](https://docs.pytorch.org/docs/2.7/debugging_environment_variables.html)提供 C++ 栈开关；[FlagOS CoreX 4.4 基础镜像说明](https://flagos-ai.github.io/release-info/base/iluvatar-corex4.4.0/)提供默认库路径参考。FlagOS 镜像说明不是当前平台镜像的配置证明，具体以本次采集为准。

## 6. 执行流初始化失败时的进一步定位

2026-09-16 新容器的报告显示：包元数据为 `torch 2.7.1+corex.4.4.0`、`triton 3.1.0+corex.4.4.0`；六组运行时对照均在 `torch.empty(1)` 失败。C++ 栈进一步指出 `initGlobalStreamState → getCurrentCUDAStream → NativeCachingAllocator::allocate`。这说明错误暴露在执行流初始化阶段，**不能据此断定 `cudaMalloc` 已经执行并失败**。

[PyTorch 2.7.1 对应源码](https://github.com/pytorch/pytorch/blob/v2.7.1/c10/cuda/CUDAStream.cpp#L162)在该初始化函数中查询 `cudaDeviceGetStreamPriorityRange`。CoreX 分支可能有修改，因此它是待直接验证的接口，而非已经确认的根因。另外，[天数容器工具源码](https://github.com/Deep-Spark/ix-container-toolkit/blob/master/internal/modifier/graphics.go#L195)将 `/dev/itrctl` 列为驱动 4.4.0 所需公共设备；前一轮采集未覆盖该节点，不能以旧报告未列出它来判断缺失。

更新后一次启动新的定向诊断：

```bash
cd /private/gpgpu-cake &&
git pull --ff-only origin main && {
  LOG="/private/gpgpu-stream-api-$(date -u +%Y%m%dT%H%M%SZ).log"
  nohup bash experiments/megatron_ops/scripts/run_stream_api_diagnostics.sh \
    > "$LOG" 2>&1 < /dev/null &
  echo "PID: $!"
  echo "日志: $LOG"
}
```

脚本分别用独立进程测试 PyTorch 当前执行流、直接调用 CoreX 的优先级查询，以及直接显存分配/双向拷贝并校验。裸 API 测试使用已安装的 CoreX `libcudart`，不导入 PyTorch，记录每次调用的原始返回码；默认库路径来自本次真实报告，可以通过 Python 入口 `--library` 指定。接口签名参考 [CUDA Runtime 10.2 设备接口](https://docs.nvidia.com/cuda/archive/10.2/cuda-runtime-api/group__CUDART__DEVICE.html)和[内存接口](https://docs.nvidia.com/cuda/archive/10.2/cuda-runtime-api/group__CUDART__MEMORY.html)，结果反映当前 CoreX 实现。

同时采集 `/dev/itrctl` 和相关挂载信息。若 `/usr/local/iluvatar/lib64` 中存在候选驱动库，则追加该目录优先的子进程对照；这是针对当前平台路径的假设，不代表这个目录一定正确或一定来自宿主机。调整前后的实际库映射保存在 JSON 中。可见性、预加载设置和系统文件保持原样。

查看最新日志：

```bash
cat "$(ls -t /private/gpgpu-stream-api-*.log | head -n 1)"
```

完整结果保存在 `results/stream-api-*/stream-api-diagnostics.json`。这些测试用于定位兼容性问题，不是性能实验，不使用伪造 API 成功返回的补丁。

## 7. 下一轮需要的输入

第一轮通过后，需要提供：

1. 四个 JSON 和 `unit-tests.log` 的结果；
2. 实际可运行的 Megatron 启动命令或配置文件；
3. Dense GPT 与 MoE GPT 各自使用的模型规模、序列长度、精度及并行参数。

随后接入真实形状捕获，并先实现 SwiGLU、Residual Add RMSNorm 的前向和反向正确性/计时入口。Cross Entropy 的完整验证需要 `torchrun --nproc_per_node=2` 的 TP=2 实验；MoE Grouped GEMM 后续也要补双卡 expert-parallel 路径。

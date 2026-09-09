# 基础验证脚本

配套[定性实验计划](../../research/notes/基础验证实验计划.md)。先检查实验设施，再接入项目的 Halide、Poly 和自动优化模块。

## 目前能做什么

- `probe`：记录 Python、PyTorch、Triton 和 GPU 信息；不安装软件、不执行 kernel。
- `cpu`：仅用 Python 标准库检查分块向量加法、矩阵乘法及尾部处理；不测 GPU 性能。
- `gpu`：在 CUDA 兼容环境中运行手工编写的 Triton 候选，检查结果，再计时并保存本轮最佳配置。天数需要厂商适配的工具链与框架。

**尚未实现：** Halide IR 规范化、Poly 依赖分析/候选生成、智能体搜索、训练反向、生产可用的多形状分派。因此这里的分块枚举不能算作 Poly 优化已完成。

文件：`run.py` 是运行入口，`triton_kernels.py` 是 GPU 候选，`test_runner.py` 是标准库自检测试，`cpu_study.py` 是扩展 CPU 正确性实验。

## 环境

- Python 3.9 或更高版本；CPU 模式不需要第三方包，可在 Windows 使用。
- GPU 主实验平台为天数单卡 Linux。使用项目服务器已有的 COREX 和匹配的厂商 PyTorch/Triton 包；不要直接用通用 pip 安装覆盖厂商环境。
- 脚本使用 `torch.cuda` 接口，因为厂商适配框架也可能使用这一接口；接口名称本身不证明设备是 NVIDIA。`probe` 记录实际设备信息，GPU 模式尝试读取 Triton target。
- 没有天数工具链或设备时，不能据本机 CPU 结果判断 GPU 兼容性。未提前假定当前 kernel 已经通过天数验证。

## 运行

以下命令在项目根目录执行。默认生成带时间戳的 JSON，不覆盖旧记录。

```bash
python experiments/basic_validation/run.py --mode probe
python experiments/basic_validation/run.py --mode cpu
python experiments/basic_validation/run.py --mode gpu --operator add --device 0
python experiments/basic_validation/run.py --mode gpu --operator matmul --device 0
```

先跑 `add`，确认编译和执行正常，再跑 `matmul`。也可用 `--operator all` 运行两类实验。

指定新的报告文件：

```bash
python experiments/basic_validation/run.py --mode gpu --operator all --output experiments/basic_validation/results/first-gpu-run.json
```

如果该文件已存在，程序拒绝覆盖。需要更多采样时可以设置 `--samples 15 --launches 50`；`--warmup` 控制每条路径的预热次数，`--seed` 控制输入种子。

## GPU 实验约定

| 项目 | 当前设置 |
|---|---|
| 向量加法 | FP32；长度 1、4099、1048576；分块 128/256/512 |
| 矩阵乘法 | 连续 FP16 输入/输出、Triton FP32 累加；M/N/K 为 64/64/64、127/65/33、256/256/256；三种分块配置 |
| 正确性参考 | 根据实际输入值在 CPU 上使用 PyTorch FP64 计算，再转换成输出类型；PyTorch GPU 基线也必须通过检查 |
| 输入 | 固定种子的正态随机值及全零值；检查输出是否有限、输入是否被意外修改 |
| 容差 | 加法 atol/rtol 均 1e-6，矩阵乘法均 1e-2，选参数过程中不修改；仅是小型 smoke test 的固定标准 |
| 性能基线 | 同输入、输出类型和预分配输出的 PyTorch `add` / `mm` |
| 时间范围 | GPU stream event 包围一批调用，除以调用次数；不计首次编译、数据复制、结果检查和分配 |
| 统计 | 多批测量的中位数、最小值、最大值；每轮轮换路径顺序，减少固定先后顺序影响 |

这是使用同一批缓冲区的基础计时，没有清空 L2 缓存、没有 CUDA Graph，也没有运行内存/同步 sanitizer。event 间隔可能包含发起调用的间隙，不能当成隔离的 kernel 执行时间或论文 CUPTI 测量。小输入差异可能受测量开销影响，先看流程能否工作，再做正式性能分析。

矩阵乘法有维度边界掩码，但尚未检查非连续布局、全部数值分布、大规模形状或梯度。输入/输出采用固定类型，不能据此宣称完整训练精度已验证。

## 如何读结果

JSON 默认在此目录的 `results/` 下，保存源码 SHA256、输入种子、软件/设备信息、每条路径的正确性、配置和计时样本。

- `cpu_check_passed`：Python CPU 自检通过；不会产生 GPU 速度排名。
- `probe_complete`：探测完成；还需看 `ready_for_gpu_attempt`，这个字段也不保证 GPU 编译一定成功。
- `gpu_unavailable`：缺少设备或依赖，退出码 2；不会偷偷改跑 CPU。
- `gpu_check_passed`：本轮所有 GPU 路径通过固定测试并完成计时，不代表提速或生产验收。
- `gpu_completed_with_failures`：存在数值不合格候选，退出码 1；失败候选不计时、不参与选优。
- `run_failed`：执行或计时出现异常，退出码 1；保留已取得的信息，停止后续运行以免在失效设备上下文中继续。

`best_observed_config` 是本轮候选中最快的正确配置。`speedup_vs_pytorch` 小于 1 表示比 PyTorch 慢；程序仍如实报告，不把它包装成优化成功，也不会自动部署替换。当前所有测试形状均参与搜索，尚无独立泛化留出集。

## 本地验证与后续接入

```bash
python -m unittest discover -s experiments/basic_validation -p "test_*.py" -v
```

测试验证 CPU 分块尾部、错误值识别、失败候选不参与选优、报告状态及旧结果保护；不会 import Triton 或验证 GPU kernel。

最新 CPU 实测：9 项单元测试、18 个基础配置、900 个扩展配置和 6 项错误拦截/选优规则检查通过。另一个极端浮点例子展示了累加顺序改变结果的情况，单独报告，不计入配置通过数。详情见 [CPU 基础验证结果](../../research/notes/CPU基础验证结果.md)；早期环境记录见[本机验证记录](LOCAL_VALIDATION.md)。

扩展实验仅需标准库，不测性能：

```bash
python experiments/basic_validation/cpu_study.py
```

覆盖多个尺寸、尾部、小整数与随机浮点分布，并使用固定精度标准。选优规则检查采用虚构耗时标签；它们不是实测速度。报告默认保存到 `results/`，也可通过 `--output` 指定新的文件名。

拿到实际项目模块后，先用其规范化前后结果做语义对照，再将手工候选清单/启动函数替换成 Poly 和后端产生的实现，保留外部正确性参考与计时协议。项目级目标是验证三处真实代码改动，不能仅改变本脚本参数便认定 Halide/Poly 工作已完成。

GPU kernel 的基本 API 与分块思路参照 [Triton 向量加法教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)和[矩阵乘法教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)。这些教程不是天数兼容性证明，厂商后端的实际结果需要现场验证。

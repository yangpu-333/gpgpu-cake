# KernelWiki 在天数 BI-V150 上的知识迁移与验证

**公开技术报告｜更新至 2026 年 10 月 3 日**

## 项目概览

KernelWiki 原本收录 NVIDIA Hopper、Blackwell GPU 的算子优化知识。本项目保留这些原始内容及其适用范围，在同一知识库中增加天数智芯 BI-V150 的工具链说明、可运行案例和独立实验依据。这样，KDA（Kernel Design Agents）既能继续检索 NVIDIA 知识，也能查询“哪些思路已在 BI-V150 上验证、哪些还不能直接迁移”。

目前交付的是**可重建、可检索、可由 KDA 项目目录加载的适配版 skill**。我们在 BI-V150 上完成了多个算子实验，并使用 Claude Code CLI、适配 skill 和固定验收程序持续迭代。最新候选融合残差/RMSNorm 与 TP=1 词表交叉熵，已接入固定版本 Megatron 的正式 `pretrain_gpt.py` 入口。同一四层 FP32 合成 GPT，三组各 120 步的正式训练对照相对原生本地 PyTorch 后端，吞吐提升 **6.97%**；三步输出、loss、全部参数梯度与 SGD 更新通过逐元素复核。此前独立训练 step 基准的 **19.10%** 作为历史结果保留，两个数字使用不同计时范围。**当前仍是单卡、小模型与原生模拟数据验证，尚无生产训练或多卡收益。**

| 交付内容 | 已验证状态 |
|---|---|
| 知识库扩充 | BI-V150 可精确检索 22 条路径；原 SM90/SM100 的 286/379 条路径逐条核对一致 |
| 可复现构建 | 固定上游提交加本项目补丁；整库校验通过，来源与实验数据可追溯 |
| KDA 项目级加载 | 适配版复制到 KDA 项目的 `.claude/skills/`，逐文件校验；不依赖个人 skill 安装，也不改原 NVIDIA 子模块 |
| 自动验收 | 最新构建的 160 项知识库测试在 Linux 全部通过，2 项加载器测试及三架构检索通过；[GitHub Actions](https://github.com/yangpu-333/gpgpu-cake/actions/workflows/kernelwiki-iluvatar-kda.yml)按同一流程从源码重建并验收 |
| 正式训练入口 | 原生模型安装 hook 提供显式优化开关；600 个原生源码文件哈希一致，原生数据、DDP、训练循环、SGD 和 checkpoint 正常运行 |

## 为什么需要单独验证

Hopper 和 Blackwell 的资料既有通用算法思路，也有只属于 NVIDIA 的指令、存储区和同步机制。把“NVIDIA 上有效”直接写成“BI-V150 上也有效”，会误导算子设计。我们采用逐项迁移方法：先识别可复用的算法结构，再在 BI-V150 的 CoreX/Triton 环境中编译、核对数值、计时，并保存失败样本。TMA、TMEM、`tcgen05`、CLC、NVFP4 等仍保留原 NVIDIA 架构标记；目前没有证据把它们认定为 BI-V150 的同名硬件能力。

适配版以 KernelWiki 上游提交 `b6b4301f15e8ce6955a56776690643ce5db369e6` 和本项目[补丁](kernelwiki-iluvatar.patch)构建。KDA 加载器以 KDA 提交 `ef6ce617693ef0782b3ecb9f37e39bbf10226a90` 和独立[补丁](kda-loader.patch)构建。两份补丁都保存在本项目仓库，便于审阅和重建。

## BI-V150 实验结果

下表中的“倍率”均为**基线耗时 ÷ 候选耗时**；大于 1 表示候选更快。除特别注明外，实验使用一张 Iluvatar BI-V150、CoreX/驱动 4.2.0、PyTorch 2.4.1、厂商 Triton 2.1.0。不同基线不能互换，也不能把单个算子的微基准倍率解释为训练吞吐倍率。

| 实验及输入 | 对照与观察 | 证据 |
|---|---|---|
| Residual Add RMSNorm 前向；FP16/BF16，5 个形状，各 3 轮 | 输出及参考梯度检查通过。相对**未融合 PyTorch 前向算子序列**为 **10.12–14.65×**；主要包含减少中间张量和 kernel 启动的收益 | [阶段 3 原始数据](evidence/stage3-20260928/) |
| Residual Add RMSNorm 两核反向；FP16/BF16，6 个形状、2 种梯度模式，3 个独立进程 | **72/72** 组通过分析式 PyTorch 和 autograd 对照。相对**分析式 PyTorch 反向序列**，24 组几何平均为 **10.38–15.41×**；尚未与优化后的反向实现比较 | [阶段 13 数据与哈希](evidence/stage13-20260929/manifest.json) |
| GEMM + bias 融合；FP16/BF16，3 个形状、4 种 tile、各 3 轮 | 数值通过；同一 tile 下，相对**GEMM 后单独执行 PyTorch bias 加法**为 **1.33–3.94×**。这不是与厂商高性能 GEMM 的比较 | [阶段 4 原始数据](evidence/stage4-20260928/) |
| GEMM 流水阶段数；FP16/BF16，两种形状、两种 tile、3 轮 | 72 次候选数值检查通过。BF16 `1024³`、`64×64×32` tile 时，`num_stages=2/3` 相对 1 仅为 **0.915/0.748×**；更多流水阶段不一定更快 | [阶段 11 数据](evidence/stage11-20260928/) |
| 固定 Megatron 的单层 GPT；合成 token，FP32，融合一对 attention 残差加法与 RMSNorm | 当时的前向、反向、loss 和 SGD 后参数检查通过。3 个独立进程、每轮 12 对交替 GPU Event 计时，前向/loss/反向/SGD 区间倍率为 **1.0183–1.0228×**，不含清梯度；17-token 留出序列也通过 | [阶段 21 原始数据](evidence/stage21-20260930/) |
| 同一 GPT step 外加 BF16 autocast；3 个独立进程 | 当时的数值检查通过，探针发现融合点的输入、残差、权重和输出**仍全为 FP32**。同一 GPU Event 区间倍率为 **0.9188–0.9279×**，即候选更慢；这不是 BF16 融合内核的训练结果 | [阶段 22 JSON、stderr 与哈希](evidence/stage22-20260930/manifest.json) |

单层 GPT 的观察形状是 `[8,2,256]`，stride 为 `[512,256,1]`，仅有一对融合点被替换。阶段 21 的约 2% 改善和阶段 22 的退化都来自**合成输入、本地 PyTorch 层规范、小模型**；不能外推到多层生产模型。阶段 14 还在 FP16/BF16 的合成前后向中完成了 **18/18** 组输出、loss、梯度和一次 SGD 更新核对，见[阶段 14 证据](evidence/stage14-20260929/manifest.json)。

负面结果同样进入知识库：当前 CoreX/Triton 组合能分配 FP8 张量，但所测 FP8 `tl.dot` 候选编译失败；分块 scan 虽正确，却比 `torch.cumsum` 慢；部分流水、缓存、持久 kernel 和近似函数候选也没有稳定收益。这些结论只针对上述软件版本和测试形状，不能推断 BI-V150 硬件完全不支持相应能力。

### 正式训练入口：原生训练循环吞吐提升 6.97%（10 月 3 日）

本轮复用经 Claude Code CLI 迭代验收的候选 0006，将它安装到 Megatron 原生 `pre_wrap_hooks` 提供的模型扩展位置。添加 `--iluvatar-kernels` 后，模型实例启用优化；省略该开关即为原生路径。两条路径均由正式 `pretrain_gpt.py` 执行参数解析、数据加载、DDP 包装、前后向调度、SGD、梯度保存和 checkpoint。

实验使用独立的固定提交副本，逐字节核对 **600 个原生源码文件**。旧工具链的导入兼容映射作为显式 `--corex42-compat` 选项保存在本仓库，两条对照共同使用；未修改安装包或该干净副本的源码。当前 Pod 的 Python 3.10.16、Torch 2.4.1 低于该提交声明的 Python 3.12、Torch 2.6 要求，因此这是已测配置的实验兼容方案。Transformer Engine 和 Apex 在进程内禁用，选择上游自带的本地 PyTorch 算子与 Torch SGD；它们不是本轮正式训练成功的前提。未使用的旧 Triton 不兼容 autotuner 只允许导入，若被调用会明确报错。

模型仍为四层、hidden size 512、8 个 head、序列 128、micro-batch/global batch 2、词表 1024、FP32、TP/PP/CP=1、dropout 0。数据改由 **Megatron 原生 MockGPTDataset 与 NullTokenizer** 构建，属于模拟数据。原生与优化路径初始权重和捕获的输入完全相等；逐 token loss、按捕获 mask 重算的 loss、全部 28 个参数的梯度与三次 SGD 更新，以 CPU float64 在原容差 `atol=3e-4, rtol=1e-3` 下核对，**737/737 项通过**。每步检查 13,701,632 个参数元素，三步共检查 41,104,896 个梯度元素及同等数量的更新后参数元素。两条路径也成功保存了第 3 步原生 checkpoint。

| 独立 seed | 原生训练循环均值 | 优化训练循环均值 | 原生 / 优化倍率 | 吞吐变化 |
|---|---:|---:|---:|---:|
| 25002 | 31.59 ms/step | 29.35 ms/step | 1.07632× | +7.63% |
| 25003 | 31.52 ms/step | 29.38 ms/step | 1.07284× | +7.28% |
| 25004 | 30.86 ms/step | 29.11 ms/step | 1.06012× | +6.01% |

三轮倍率几何平均 **1.069736×，即吞吐提升 6.97%**，约从 8.10–8.30 千 token/s 提高到 8.71–8.79 千 token/s。每个进程运行 120 步，原生/优化进程顺序在三组之间交替；原生每 10 步日志中的 GPU 同步 `time.time` 区间计时为主指标，舍弃前 20 步，平均后 100 步的 10 个完整区间。首次单步初始化日志另存，不参与平均；日志单区间精度为 0.1 ms。性能测试关闭张量审计，不保存 checkpoint、不执行评估；启动耗时另记。这个范围包含原生数据与调度、DDP、SGD 和训练运行检查，较此前独立 step 基准更广，不能直接套用历史 19.10% 的倍率。

每个优化性能进程有 480 次残差融合、480 次归一化结果消费、120 次 CE 调用，无原生回退。另一个非计时的正式训练进程观察到残差前向、两种反向及 CE 前后向共五类编译内核的实际 runner 调用；首轮 JIT 的直接启动不经过这一观察接口，因此该审计只证明已记录的调用，不把 API 计数当作 GPU 时间线。

恢复验证使用同一个原生第 3 步 checkpoint，两条路径都加载原生 optimizer/RNG 与学习率调度状态，继续执行第 4、5、6 步并保存第 6 步 checkpoint。实际恢复日志、样本消费进度、共同输入 checkpoint 的字节哈希及后续逐元素比较通过 **776/776 项检查**，见[恢复训练验收](megatron_training_entry/evidence/resume-comparison-01.json)。恢复审计含磁盘与张量保存，仅用于正确性，不参与吞吐统计；三步继续训练也不构成长周期收敛证明。

集成代码、固定契约、每次启动命令、失败诊断、逐张量误差和原始区间保存在 [`megatron_training_entry/`](megatron_training_entry/)。主要结果见[三步数值验收](megatron_training_entry/evidence/audit-comparison-01.json)、[正式循环性能验收](megatron_training_entry/evidence/performance-120-validated.json)和[编译调用观察](megatron_training_entry/evidence/formal-dispatch-01/dispatch.json)。本轮未改变已有算子候选、容差、旧基准程序或 KernelWiki 的 NVIDIA 资料。

### 词表交叉熵扩展：完整 step 吞吐提升 19.10%（阶段 24，10 月 2 日）

本轮沿用阶段 23 的 BI-V150、CoreX 4.2.0、固定 Megatron 原生本地 PyTorch 后端，以及四层 GPT 配置：hidden size 512、8 个 attention head、序列长度 128、micro-batch 2、词表 1024、RMSNorm epsilon `1e-5`、dropout 0、无 linear bias、SGD。原生与候选使用相同输入和初始权重，正确性容差及完整 step 计时也保持一致。

新增目标是 Megatron 的词表交叉熵。原生实现即使在 TP=1 时，仍包含三次单成员 all-reduce 和多次逐元素、归约操作。候选在优化模型实例上替换这一调用，用 Triton 融合前后向，并复用已编译 kernel；原来的 0003 残差/RMSNorm 代码保持不变。优化范围为 **TP=1、零 label smoothing**，不支持的调用交回原生路径。实际模型中的 CE logits 形状为 `[128,2,1024]`。

Claude Code CLI 通过 Paratera 请求 `Claude-Opus-4.8`，原始响应的模型用量元数据也返回该标识。CLI 使用项目级 skill 生成候选，固定程序在 BI-V150 上测试并反馈。0004 在初筛中提速，但独立审计发现较大的 int64 标签被截断，因而淘汰；0005 修正标签处理，FP32 检查通过，仍因 autocast 输出误差超出原容差而淘汰。最终 0006 增加原生等价的 autocast 兼容路径，重新运行全部 14 项验证，通过 16 项汇总门槛。失败候选、原始反馈和修复过程均保留在 [`higher_gain/`](megatron_cc/higher_gain/)。

| 复测 seed | 原生完整 step 中位耗时 | 候选 0006 中位耗时 | 吞吐倍率 |
|---|---:|---:|---:|
| 24002 | 11.682509 ms | 9.713142 ms | 1.202753× |
| 24003 | 11.584879 ms | 9.548354 ms | 1.213285× |
| 24004 | 11.501660 ms | 9.934883 ms | 1.157705× |

三轮倍率的几何平均为 **1.191002×，即 19.10% 的吞吐提升**。指标是原生完整 step 中位耗时除以候选中位耗时，包含清梯度、前向、loss、反向和 SGD。每轮仍为 5 个预热 step、12 组交替样本、每组 10 step；使用同步边界内的 `time.perf_counter`，排除初始编译、数据加载和 checkpoint I/O。诊断 profiler 用于选择热点，其额外开销和嵌套统计不计入这些验收结果。

同期还在三个独立进程中复测旧候选 0003。将每个候选相对各自原生对照的倍率归一化后，0006 相对 0003 的几何平均倍率为 **1.179283×**。这是独立进程的归一化对照，不能当作同一进程内的三臂直接配对结果。

正确性覆盖三轮共 **1512/1512** 个 CE 有限值 case-round，以及 **378/378** 个原生布局错误兼容性 case-round；后者只确认候选回退保持原生报错，不计入有限值正确性。残差/RMSNorm 仍通过 **672/672** 个有限值 case-round 和 **48/48** 个原生非有限值兼容性 case-round。所有比较使用 CPU float64 副本，容差未调整。CE 同时检查逐 token loss、全部 logits 梯度、原生 FP32 输入在前向后变为概率的行为、低精度输入保持不变，以及边界和越界标签。原生 `-100` 不是 ignore-index；候选保留这一语义。额外的 int64 大范围标签探针通过，独立执行审计确认了主 FP32 形状的 CE 前向与反向编译内核调用。模型输出、loss、全部参数梯度、三次 SGD 更新和计时后参数状态均通过。

两层留出模型相对原生的倍率为 **1.219762×**。BF16 autocast 完整 step 倍率为 **1.081632×**，此时 CE logits 实际为 BF16、loss 为 FP32；残差/RMSNorm 使用原生回退，CE 使用与原生一致的 TP=1 Torch autograd 运算，只省略单成员 all-reduce 恒等调用。独立 CE 审计样本中的 loss 和 logits 梯度与原生完全一致，**autocast 下没有 Triton CE 调用**；adapter 的 `optimized` 计数只表示选择了候选 API。因此这个结果不能称为融合 BF16 CE 训练提速。两条路径的临时显存峰值相同，测量范围仍是两个模型同时驻留时的临时峰值，不能解释为单模型总显存。

最终候选只在所测范围晋升。CE 验证的词表大小为 1、7、512、769、1024；这没有证明任意词表大小都有效。最新来源为 `exp-bi-v150-corex42-stage24`，算子页面为 `kernel-cross-entropy-megatron-bi-v150`。所有原始样本、源码哈希和门槛见[阶段 24 验收判定](megatron_cc/higher_gain/evidence/decision-0006.json)与[最终 14 项运行](megatron_cc/higher_gain/evidence/0006-final/)；[独立审计](megatron_cc/higher_gain/evidence/0006-dispatch-audit.json)记录执行路径和额外数值探针。这些结果未覆盖生产数据、多卡、Transformer Engine、微调或 RL 微调。

### 原生 Megatron 与 Claude Code 完整闭环（10 月 2 日）

以下保留阶段 23 的历史结果，优化范围仅为残差/RMSNorm；阶段 24 的最新完整 step 结果见上一节。

本轮优化对象是 attention 残差加法与紧邻的 pre-MLP RMSNorm。基线采用 Megatron 提交 `5be9626709af2722333bf54797c954c09edeada3` 的原生本地 PyTorch 后端。两条路径使用相同权重、合成 token、epsilon、精度和 SGD；没有使用 Transformer Engine 作为基线。

Claude Code CLI 在本机加载 KDA 项目目录中的适配 skill，通过已配置的 Paratera API 请求模型 `Claude-Opus-4.8`。GPU 计算全部经 SSH 在 BI-V150 上执行。固定程序负责验收，实测反馈交回 CLI 生成下一候选。这是 KDA basic-flow 的下游实验，没有运行可选 Humanize 插件。

三个候选保留了完整父子关系及失败证据：第一版数值通过但训练变慢；第二版复用已编译 kernel，FP32 三轮平均倍率达到 1.0124，但 autocast 的严格数值检查失败；第三版保留 FP32 优化，在 autocast 下调用原生算子，最终通过预先设定的全部门槛。

主模型为四层 GPT，hidden size 512、8 个 attention head、序列长度 128、micro-batch 2、词表 1024；dropout 为 0、不含线性层 bias。四层均替换目标调用点，实际融合形状为 `[128,2,512]`。下表是最终候选的三次独立进程结果。

| 复测 | 原生完整 step 中位耗时 | 候选完整 step 中位耗时 | 吞吐倍率 |
|---|---:|---:|---:|
| 1 | 11.2630 ms | 11.1344 ms | 1.0116× |
| 2 | 11.5278 ms | 11.3443 ms | 1.0162× |
| 3 | 11.2027 ms | 11.1487 ms | 1.0048× |

三轮倍率的几何平均为 **1.010846×，即约 1.08% 的吞吐改善**。每轮预热 5 step，保留 12 组交替样本，每组连续运行 10 step。计时使用 GPU 同步边界内的 `time.perf_counter`，包含清梯度、前向、loss、反向和 SGD 更新，排除初始编译、数据读取和 checkpoint I/O。算子微基准仍使用 `torch.cuda.Event`；两个指标不能混用，GPU Event 区间也可能包含主机启动间隙。

正确性按 CPU 拷贝逐元素比较：每轮 224 个原生有限值样本通过，三轮共 **672/672 case-round**；覆盖四种形状、五种输入/残差类型组合、两个 epsilon、三个数值尺度以及残差梯度有无两种模式。另有原生 FP16 小幅值输入产生无穷大梯度的 **48/48** 个异常 case-round，仅核对是否保持原生行为，不计入有限值正确性。模型前向、loss、全部参数梯度、三次 SGD 更新及计时后参数状态均通过。

两层留出模型（hidden size 256、序列长度 17）的倍率为 **1.0119×**。BF16 autocast 倍率为 **1.0116×**，但该场景走的是**原生回退**，只能用于确认兼容性和无明显退化，不能作为融合 BF16 训练提速的证据。回退探针确认前后向未编译任何 Triton kernel，输出和梯度与原生完全一致。两条路径的临时显存峰值相同；这是同时驻留两个模型时的测量，不能解释为单模型总显存。

验收中还发现一个工具链问题：这版 CoreX 的 GPU float64 减法把 `[1,2]` 与 `[1.125,2.25]` 算成零差异，而 GPU float32 和 CPU float64 返回正确差值。因此本轮数值检查改为 CPU 比较，容差保持不变，两版候选均重新验证。较早的错误检查器、原始结果和修复探针全部保留；不能据此推断 BI-V150 硬件不支持 FP64。

最终候选只在已测形状和配置范围内接受。代码的 `rows<=2048`、`hidden<=8192` 是分派边界，尚未逐一验证整个范围。原始数据、源码哈希、CLI 提示与回复、失败记录和验收命令保存在 [`megatron_cc/`](megatron_cc/)，最终判定见[验收 JSON](megatron_cc/evidence/decision-0003.json)。本轮没有生产数据配置、多卡训练、微调或 RL 微调结果。

### 迁移范围一览

原知识库的 **17 项技术条目**已逐项审视，其中 16 项涉及算子或性能机制，1 项是来源检索方法。下表概括覆盖面；完整的逐项状态、来源 ID 和后续探针保存在适配版的 `references/iluvatar-migration-ledger.md`。

| 类别 | 已覆盖条目 | BI-V150 结论 |
|---|---|---|
| 已形成算子级正面证据 | kernel fusion、epilogue fusion | 融合前后向与 GEMM+bias 在所测形状通过；倍率只适用于各自明示的 PyTorch 基线 |
| 可调参但依赖形状 | tile scheduling、register budgeting、vectorized loads | 已记录 tile、寄存器和访问模式；尚无目标端最终向量指令及占用率归因 |
| 已探测但不能认定硬件等价 | cache policy、double buffering、pipeline stages、swizzling、warp specialization、ping-pong scheduling、persistent kernels | 有编译或计时证据，若干候选无收益；Triton 中间表示不足以证明对应 NVIDIA 硬件机制 |
| 低精度与算法候选 | fine-grained quantization、software exp、chunk parallelism、CCCL memory primitives | FP8 `tl.dot` 当前版本编译失败；近似指数与 scan 原型尚无稳定收益 |
| 来源方法 | external source map research | 用于检索与审计，不作为 GPU 硬件迁移项 |

## 从实验到可用知识

适配版不仅增加文字说明，还把每个 BI-V150 结论连接到源码、环境、原始计时和 SHA256 回执。迁移账本按“已验证、仅观察到编译中间表示、尚未验证”区分证据等级。阶段 24 最新构建含 **1063 个页面、1000 个 source ID、37 个资产包、14 个账本**，整库校验通过且无孤立来源文件；新增证据包的 168 个文件均通过 SHA256 核对。最新 **160 项 Linux 知识库测试全部通过**，原始结果见[测试日志](megatron_cc/higher_gain/evidence/stage24-linux-tests.stderr)和[整库校验日志](megatron_cc/higher_gain/evidence/stage24-linux-validation.stdout)。

KDA 的 2 项加载器测试通过；项目级加载逐文件校验了 **2041 个文件**，本轮 CLI 使用的 checkout 也完成同样核对。BI-V150 检索仅增加本轮 CE 来源和算子页，合计 22 条路径；SM90/SM100 的 286/379 条路径与上一版本逐条一致，见[阶段 24 回归与证据核对](megatron_cc/higher_gain/evidence/stage24-skill-provenance.json)。[阶段 23 Linux 测试](megatron_cc/evidence/linux-skill-unittest.log)与[整库校验](megatron_cc/evidence/linux-skill-validate.log)作为历史记录保留。Windows 初次测试因默认 GBK 编码和系统 `python3` 别名报错，记录保留，回归采用与 CI 一致的 Linux 环境。

KDA 可以从项目目录读取这套知识。一次只读 Claude Code 验证实际读取了新增的[阶段 21 来源页](evidence/kda-repo-integration-20260930/claude-stage21-read.json)，正确返回观察形状、三轮完整 step 倍率及其非生产训练的限制。加载器、测试和 CI 均作为本仓库补丁交付；本项目没有向 NVlabs/kda 上游提交 PR。

## 当前边界与下一步

1. **生产训练配置。** 正式 `pretrain_gpt.py` 已在本地 PyTorch 后端运行并接入优化，原生模拟数据的四层 FP32 对照取得 6.97% 吞吐提升。尚无代表性的生产模型、数据、训练命令和精度配置；取得配置后，需要重新采集实际形状并验证 loss 曲线、显存与吞吐。当前运行时兼容方案也只在上述范围验证。
2. **模型级融合。** 已在四层合成 GPT 中接入逐层残差/RMSNorm和 TP=1 词表交叉熵。下一步按真实训练配置重新分析热点，覆盖实际形状、bias、dropout、精度和词表大小；TP>1 的交叉熵还需要正确的跨卡归约。当前 autocast 采用原生或原生等价计算，低精度融合需要单独设计并验证。
3. **硬件归因。** 目前有 Triton 编译中间表示和计时，没有足够的最终机器指令、计算/搬运重叠或共享存储冲突证据。平台开放 tracefs/debugfs 后，才能用 ixSYS 采集时间线并进一步分析。
4. **低精度与版本升级。** 在新版 CoreX/Triton 上重新探测 FP8/FP4、编译和数值行为；不同软件版本的性能数据分别记录。

## 复现与证据入口

从本仓库根目录重建适配版 skill；构建脚本拉取固定的上游版本并应用本项目补丁，目标目录应事先不存在：

```bash
python experiments/kernelwiki_iluvatar/build_skill.py --output /path/to/kernelwiki-iluvatar
python /path/to/kernelwiki-iluvatar/scripts/validate.py
python -m unittest discover -s /path/to/kernelwiki-iluvatar/tests
```

KDA 项目级加载使用[构建脚本](build_kda_integration.py)和[验收脚本](verify_kda_integration.py)。[CI 工作流](../../.github/workflows/kernelwiki-iluvatar-kda.yml)从固定的 KDA 与 KernelWiki 提交重新构建，执行完整测试并核查 BI-V150、SM90、SM100 三架构查询。实验脚本、逐轮 JSON、失败记录和校验清单位于本目录的 [`evidence/`](evidence/) 及相邻源码文件；性能结论以这些原始记录为准。

正式入口复测使用 `megatron_training_entry/contract.json` 和同目录的 `source_manifest.json`，候选仍为 `megatron_cc/higher_gain/candidates/0006/candidate.py`。准备本仓库的独立 Linux 工作副本及 Megatron 的干净固定提交，保留两目录的相邻关系；将复制后的原 `evidence/` 归档到另一个目录，保留原文件，重新建立空的 `evidence/` 用于本次运行。下面命令中新 case 目录与 output 应尚不存在。入口显式使用 `--corex42-compat`；直接启动 `launch_pretrain.py` 时加 `--iluvatar-kernels` 启用优化。该运行时选项针对本报告的软件组合，不能作为新版环境的通用兼容层。

```bash
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/openmpi/lib
cd /path/to/independent-copy/experiments/kernelwiki_iluvatar/megatron_training_entry
export MEGATRON_CHECKOUT=/path/to/clean-pinned-megatron
python run_case.py native-check --mode native --megatron-root "$MEGATRON_CHECKOUT" --steps 3 --seed 25001 --audit
python run_case.py optimized-check --mode optimized --megatron-root "$MEGATRON_CHECKOUT" --steps 3 --seed 25001 --audit
python compare_cases.py evidence/native-check evidence/optimized-check --output evidence/audit-rerun.json
python run_performance.py "$MEGATRON_CHECKOUT"
python summarize_performance.py evidence evidence/performance-rerun.json
python run_dispatch.py "$MEGATRON_CHECKOUT"
python run_resume.py native-resume-check optimized-resume-check --megatron-root "$MEGATRON_CHECKOUT" --checkpoint-case evidence/native-check
python compare_resume.py evidence/native-resume-check evidence/optimized-resume-check --checkpoint-case evidence/native-check --output evidence/resume-rerun.json
```

大张量快照与原生 checkpoint 单独保留，不提交 Git；远端60个原始文件合计约2.19 GB，其大小、位置和逐文件SHA256见[原始数据清单](megatron_training_entry/evidence/raw-snapshots-manifest.json)。公开仓库保存日志、契约、误差与哈希回执，远端原始根目录在清单中记录；最终数值验收与恢复的快照另备份到本机同一case的`snapshots/`。CPU CI增加了21项安装器测试和10项计时日志测试；实机训练与恢复验证在BI-V150上单独执行，GitHub CI不运行GPU训练。

阶段 24 使用 `megatron_cc/higher_gain/contract.json`，同时固定继承阶段 23 的 `benchmark.py`、原契约和 0003 残差/RMSNorm 源码。在具有上述 CoreX 环境及固定 Megatron 源码的 Linux 机器上，准备独立工作副本，保留 `megatron_cc/` 的目录结构。复测副本中的 `higher_gain/evidence/0006-final/` 应尚不存在，已归档的原始运行目录完整保留在证据副本中。

```bash
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/openmpi/lib
cd /path/to/independent-copy/megatron_cc/higher_gain
python run_extended.py 0006 final
python audit_ce_dispatch.py candidates/0006/candidate.py evidence/0006-dispatch-rerun.json
python summarize_extended.py 0006 evidence/0006-final evidence/decision-rerun.json \
  --contract contract.json --candidate candidates/0006/candidate.py \
  --best ../candidates/0003/candidate.py --base ../benchmark.py \
  --extension driver.py --parent-contract ../contract.json \
  --dispatch-audit evidence/0006-dispatch-rerun.json
```

阶段 23 的历史复测仍使用 `megatron_cc/contract.json`。在另一份独立工作副本中执行以下命令，其 `evidence/0003-final/` 也应尚不存在：

```bash
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/openmpi/lib
cd /path/to/independent-copy/megatron_cc
python run_remote.py 0003 final
python summarize.py 0003 evidence/decision-rerun.json
```

`run_remote.py`、`higher_gain/run_extended.py` 和独立审计中的 Megatron 路径默认是 `/private/atrex-megatron/src/megatron-lm`；其他部署需改为本机固定源码的位置。源码、契约和验收程序的 SHA256 同时进入逐轮结果，以便识别复测是否使用相同内容。

**ixSYS 可视化：** 项目已准备[带 NVTX 标记的短工作负载](stage4_trace_case.py)。10 月 2 日重启后的 Pod 仍缺少 tracefs/debugfs 控制节点，尚未生成可上传的 `.ptrace`；本轮网页访问也超时。JSON、stdout 和 stderr 不能直接当 trace 上传。平台开放挂载并可正常访问网站后，先用 ixSYS CLI 采集 `.ptrace`，将文件下载到本机，再在 [ixSYS 页面](https://ui.ixsys.iluvatar.com)选择 **Open trace file**。当前诊断见[tracefs 日志](megatron_cc/evidence/tracefs-20261002.txt)，原始采集失败记录见[平台诊断日志](evidence/stage12-20260929/tracefs-probe.log)。

---

本报告记录的是截至 2026 年 10 月 3 日、固定环境下的可复核结果。NVIDIA 原知识、BI-V150 实测和待验证推断在适配版中分别标注，避免跨设备借用性能数字。

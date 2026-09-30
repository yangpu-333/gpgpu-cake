# KernelWiki 在天数 BI-V150 上的知识迁移与验证

**公开技术报告｜2026 年 9 月**

## 项目概览

KernelWiki 原本收录 NVIDIA Hopper、Blackwell GPU 的算子优化知识。本项目保留这些原始内容及其适用范围，在同一知识库中增加天数智芯 BI-V150 的工具链说明、可运行案例和独立实验依据。这样，KDA（Kernel Design Agents）既能继续检索 NVIDIA 知识，也能查询“哪些思路已在 BI-V150 上验证、哪些还不能直接迁移”。

目前交付的是**可重建、可检索、可由 KDA 项目目录加载的适配版 skill**。我们在 BI-V150 上完成了多个算子实验，并把一次局部融合接入固定版本 Megatron 的单层 GPT 训练 step。结果显示：部分算子在所测微基准中明显受益，但完整小模型的收益很小，换到 BF16 autocast 上下文后还出现退化。因此，**本项目证明了知识迁移和局部训练接入的可行性，尚未证明生产训练提速**。

| 交付内容 | 已验证状态 |
|---|---|
| 知识库扩充 | BI-V150 可精确检索 18 条路径；原 SM90/SM100 的 286/379 条路径保持不变 |
| 可复现构建 | 固定上游提交加本项目补丁；整库校验通过，来源与实验数据可追溯 |
| KDA 项目级加载 | 适配版复制到 KDA 项目的 `.claude/skills/`，逐文件校验；不依赖个人 skill 安装，也不改原 NVIDIA 子模块 |
| 自动验收 | 160 项知识库测试、加载器测试和三架构检索在 [GitHub Actions](https://github.com/yangpu-333/gpgpu-cake/actions/runs/36659128027) 通过；同一提交已同步至本项目 GitHub 和 GitLab |

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
| 固定 Megatron 的单层 GPT；合成 token，FP32，融合一对 attention 残差加法与 RMSNorm | 前向、反向、loss 和 SGD 后参数检查通过。3 个独立进程、每轮 12 对交替计时，**完整 step** 倍率为 **1.0183–1.0228×**；17-token 留出序列也通过 | [阶段 21 原始数据](evidence/stage21-20260930/) |
| 同一 GPT step 外加 BF16 autocast；3 个独立进程 | 数值检查通过，但探针发现融合点的输入、残差、权重和输出**仍全为 FP32**。完整 step 倍率为 **0.9188–0.9279×**，即候选更慢；这不是 BF16 融合内核的训练结果 | [阶段 22 JSON、stderr 与哈希](evidence/stage22-20260930/manifest.json) |

单层 GPT 的观察形状是 `[8,2,256]`，stride 为 `[512,256,1]`，仅有一对融合点被替换。阶段 21 的约 2% 改善和阶段 22 的退化都来自**合成输入、本地 PyTorch 层规范、小模型**；不能外推到多层生产模型。阶段 14 还在 FP16/BF16 的合成前后向中完成了 **18/18** 组输出、loss、梯度和一次 SGD 更新核对，见[阶段 14 证据](evidence/stage14-20260929/manifest.json)。

负面结果同样进入知识库：当前 CoreX/Triton 组合能分配 FP8 张量，但所测 FP8 `tl.dot` 候选编译失败；分块 scan 虽正确，却比 `torch.cumsum` 慢；部分流水、缓存、持久 kernel 和近似函数候选也没有稳定收益。这些结论只针对上述软件版本和测试形状，不能推断 BI-V150 硬件完全不支持相应能力。

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

适配版不仅增加文字说明，还把每个 BI-V150 结论连接到源码、环境、原始计时和 SHA256 回执。迁移账本按“已验证、仅观察到编译中间表示、尚未验证”区分证据等级。最新构建含 **1059 个页面、998 个 source ID、37 个资产包、14 个账本**，无孤立来源文件；KDA 项目级加载并校验 **1783 个文件**。原 SM90/SM100 的精确检索路径集合保持不变。

KDA 可以从项目目录读取这套知识。一次只读 Claude Code 验证实际读取了新增的[阶段 21 来源页](evidence/kda-repo-integration-20260930/claude-stage21-read.json)，正确返回观察形状、三轮完整 step 倍率及其非生产训练的限制。加载器、测试和 CI 均作为本仓库补丁交付；本项目没有向 NVlabs/kda 上游提交 PR。

## 当前边界与下一步

1. **真实训练。** 当前 Pod 的 Transformer Engine 与固定 Megatron 提交不兼容，因此模型实验采用 Megatron 自带的本地 PyTorch 层规范。尚无生产训练配置；需要兼容运行时和代表性配置后，才能采集实际形状、精度、loss 曲线、显存及多层吞吐。
2. **模型级融合。** 先按真实 dtype 与调用形状筛选可融合位置，再逐步扩到多个位置，和固定基线比较完整 step。阶段 22 说明必须检查融合点实际 dtype，不能仅凭外层 autocast 开关判断。
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

**ixSYS 可视化：** 项目已准备[带 NVTX 标记的短工作负载](stage4_trace_case.py)，但当前容器无法访问 tracefs/debugfs 控制节点，尚未生成可上传的 `.ptrace`。JSON 和日志不能直接当 trace 上传。平台开放挂载后，先用 ixSYS CLI 采集 `.ptrace`，再在 [ixSYS 页面](https://ui.ixsys.iluvatar.com)选择 **Open trace file**；网站能否正常访问还需在采集时复查。[平台诊断日志](evidence/stage12-20260929/tracefs-probe.log)保留了当前阻断证据。

---

本报告记录的是截至 2026 年 9 月 30 日、固定环境下的可复核结果。NVIDIA 原知识、BI-V150 实测和待验证推断在适配版中分别标注，避免跨设备借用性能数字。

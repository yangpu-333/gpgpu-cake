# KDA KernelWiki 天数 BI-V150 适配：工作总报告与下一步计划

实验日期：2026-09-28 至 2026-09-30；汇总更新：2026-09-30。范围为本项目中 KernelWiki skill 的 BI-V150 知识扩充与实机验证。**当前成果是可从固定上游快照重建、由 KDA 项目级加载器读取的适配版 skill；固定 Megatron 的单层 GPT 本地后备路径已完成一步训练与局部融合对照，生产模型和兼容 Transformer Engine 路径仍待验证。** 本报告合并了全部阶段结论、17 项迁移账本和 ixSYS 操作说明；原始记录与失败样本保留在 `evidence/`。所有性能数字均限定在所列设备、软件版本、形状和基线。

## 1. 目标与当前状态

目标是在**保留 Hopper/Blackwell 原页面、架构标记和性能数据**的前提下，给 KernelWiki 增加 Iluvatar BI-V150 的工具链、算子、迁移边界及实验来源。下文逐项记录 17 项技术的目标端证据与下一步验证；skill 内也有可查询的 `references/iluvatar-migration-ledger.md`。`bi-v150` 已能精确检索，SM90/SM100 的原查询路径集合未变化。

当前交付以固定上游提交 `b6b4301f15e8ce6955a56776690643ce5db369e6`、[适配补丁](kernelwiki-iluvatar.patch)和[构建脚本](build_skill.py)重建。KDA 项目级加载器以独立[补丁](kda-loader.patch)和[构建脚本](build_kda_integration.py)保存，不依赖个人安装目录；原 NVIDIA 子模块不变。没有将本地结果写成 NVIDIA 指令在 BI-V150 上的同名实现。

## 2. 访问、环境与证据约定

用户提供的入口为 `root@ic.h3i.buaa.edu.cn:41686`，Pod 内 SSH 监听 22。本机 `bi-v150` 别名已使用专用 Ed25519 公钥完成免密登录；Pod 重建时曾因 `sshd` 未启动、公钥未保留及 host key 文件权限错误导致拒绝连接，逐项修正后可运行 `ssh -o BatchMode=yes bi-v150`。报告和仓库不保存密码或私钥。

主要实验固定为 **Iluvatar BI-V150 device 0**（Pod 有两张 32 GiB 卡），CoreX/驱动 4.2.0、PyTorch 2.4.1、厂商 Triton 2.1.0、Python 3.10.16。非交互 SSH 运行需显式注入 CoreX 的 `PYTHONPATH` 和 `LD_LIBRARY_PATH`。曾遇到新 Pod 的最小 GPU 分配在两张卡上均无响应；用户重启后，最小分配、同步与能力复测通过，故未把该故障计入 kernel 失败。较早项目报告使用过 PyTorch 2.7.1/Triton 3.1.0，不能与此批结果合并比较。最小分配及恢复记录在 [`evidence/stage3-20260928/`](evidence/stage3-20260928/)。

原始源码、逐轮 JSON、日志、失败候选和 SHA256 回执保存在本目录的 `evidence/`；远端实验工作目录为 `/private/kda-biv150-stage3/`。短 kernel 使用预热后的 GPU event；正式性能结论采用同进程交替对照及至少三次独立进程复测。正确性检查、编译和参考实现开销不计入 event 时间。

## 3. 已完成工作与结果

| 阶段 | 完成内容 | 能支持的结论与边界 |
|---|---|---|
| 0 | SSH/GPU 工具链；向量加法、FP16 GEMM smoke；修复旧版 Triton 对 `triton.cdiv(K,BK)` 的 JIT 不兼容 | BI-V150 可执行最小 Triton 路径；首次单轮短核比值仅用于验证计时链路 |
| 1 | 固定上游版本；增加 `bi-v150` 架构词表、别名、查询、索引、来源证据契约与迁移页；保留原 NVIDIA 页面 | 初版 SM100 379、SM90 286 条查询路径与上游一致；阶段 1 上游测试 160 项通过 |
| 2 | BF16 `tl.dot` 三形状、FP16/BF16 前向 RMSNorm 各三形状 | 9/9 修正后正确；该 Triton 构建缺 `tl.rsqrt`，用 `1.0/tl.sqrt` 才编译通过；未验证训练反向性能 |
| 3 | FP16/BF16 Residual Add RMSNorm 融合，五个形状、三轮独立进程；检查前向、residual 输出及分析式 PyTorch 后向梯度 | 所选配置相对**未融合 PyTorch 前向序列**为 10.119–14.647×；`num_warps=1` 在部分形状数值失败；尚非完整训练吞吐或融合后向收益 |
| 4：A/B | GEMM+bias epilogue，FP16/BF16、三形状、四 tile、三轮；连续/stride-2 访问与资源观测 | 同 tile 融合 bias 对独立 PyTorch bias launch 为 1.334–3.936×；不是与厂商高性能 GEMM 比较。访问核可观测 `n_regs/n_spills/shared`，不据此臆断具体向量指令或 occupancy |
| 4：C/D/E | exp 近似、两段式 scan、缓存修饰、阶段数、FP8、固定网格循环 | `tl.exp2` 缺失；有限区间 Taylor 正确但无稳定收益；scan 正确却比 `torch.cumsum` 慢；FP8 张量可分配而四个 FP8 `tl.dot` 候选编译失败；持久加法原型较慢，均为当前软件栈和测试范围的结论 |
| 9 | 保存 `num_stages` 的 TTGIR/LLVM IR/二进制；程序 tile 重排、双累加器与 warp-specialization 开关探测 | stage 2/3 的 TTGIR 有 async slice/commit/wait 与多层 shared；`tl.swizzle2d` 是程序 tile 重排；warp-specialization 开关未改变该 GEMM 的 IR。均不构成 TMA、共享内存 bank swizzle、TMEM ping-pong 或 warp 角色分工的硬件等价证据 |
| 10 | 两个更大 FP16 GEMM 形状上的 stages 1/2/3 三轮配对复测 | 32×32×32 tile 中，stage 2 为 1.0154×/1.0018×，stage 3 为 0.7724×/0.9166×（均相对 stage 1） |
| 11 | 扩到 BF16 和 64×64×32 tile；两形状、两 dtype、两 tile、三阶段数、三轮 | 72 次候选数值检查全部通过；BF16 长 K、32 tile 的 stage 2 为 1.0162×，BF16 1024³、64 tile 的 stage 2/3 仅为 0.9152×/0.7484×；阶段数须与 dtype、shape、tile 联合选择 |
| 13 | 依据 skill 的融合页和迁移账本，实现 Residual Add RMSNorm 两核反向；FP16/BF16、两种 residual 输出梯度模式、六形状、三次独立进程 | 72/72 case-round 同时通过分析式 PyTorch 与独立 autograd 梯度检查；后向微基准相对分析式 PyTorch 序列的 24 组几何平均为 10.379–15.405×，包含基线的多次 launch 与临时张量开销，不是完整训练吞吐 |
| 14 | 将阶段 3 前向与阶段 13 反向接入自定义 `torch.autograd.Function`；合成 loss 使用两个输出并执行一次 SGD 权重更新；探测固定提交的 Megatron 路径 | FP16/BF16、三种 3D 形状、三轮独立进程共 18/18 case-round 通过输出、loss、三组梯度和更新后权重对照；真实 Megatron 导入受包解析冲突及当前 Transformer Engine stub 缺 `__version__` 阻断，尚无真实训练结果 |
| 15 | 在独立进程中临时指定固定 Megatron 命名空间，并把厂商 `te_version()` 映射到该 Megatron 期望的 `__version__`；未改动安装包和 checkout | 导入继续推进后仍因当前 CoreX Transformer Engine 缺 `transformer_engine.pytorch.float8_tensor` 停止；仅补版本字段不能形成兼容运行时，未进入真实模型 step |
| 18–19 | 在进程内屏蔽不兼容的 Transformer Engine stub，映射 CoreX PyTorch 2.4 的旧 DTensor 导出路径，调用固定 Megatron 的 local BDA/RMSNorm 路径，并用融合前后反向与一次 SGD 更新对照 | FP16/BF16 合成 `[8,2,1024]` 算子路径及 FP32 `[8,2,256]` 模型观测形状通过；仅为算子路径，不是完整 GPT step |
| 20 | 固定 Megatron 提交的单层 GPT 以合成 token 完成前向、反向、SGD 更新，并通过 hook 捕获输入 | 真实模型调用观测到三次连续 FP32 RMSNorm 输入 `[8,2,256]`、stride `[512,256,1]`；首步 504.85 ms 含初始化与编译，不用于吞吐结论 |
| 21 | 将单层 GPT 中的一对 attention BDA→pre-MLP RMSNorm 接成 BI-V150 前向加两核反向；与同权重本地 Megatron 基线交替计时 | 三次独立进程均通过模型输出、loss 和更新后参数检查；12 对稳态 step 的进程内中位耗时比为 1.0183–1.0228，17-token 留出序列也通过。仅限 FP32、单层、合成 token、本地后备路径，不代表生产训练收益 |

阶段 3、4 等较早性能数字属于**单卡预热算子微基准**；阶段 13 单独测反向，阶段 21 单独测一个小 GPT 的完整 step。阶段 3 的大幅比值包含消除 PyTorch 多次 launch 的收益，阶段 4 的 epilogue 比值包含消除独立 bias launch 的收益；两者不能直接解释为模型吞吐、训练反向或对其它优化内核的加速。

阶段 13 是独立的**仅反向微基准**，不属于上段正向数字。两核实现先按行计算 `grad_x`、`grad_residual` 和 FP32 权重梯度部分和，再跨行归约；保留 residual 输出的可选梯度。三轮原始事件样本、六形状（含未见过的 17×1537 尾块）、误差统计和 SHA256 见[阶段 13 manifest](evidence/stage13-20260929/manifest.json)，可运行源码为 [`stage13_backward.py`](stage13_backward.py)。例如 BI-V150、BF16、64×1024、residual 输出梯度存在时，三轮后向几何平均相对分析式 PyTorch 序列为 **15.405392×**，来源为 `exp-bi-v150-corex42-stage13` 的 `manifest.json#summary[10]`。该基线包含多次 PyTorch kernel launch 和临时张量，尚未与另一个优化反向实现或真实训练 step 比较。

阶段 14 将两个已验证内核连成合成前后向路径，并检查一次 SGD 更新；三轮共 18/18 用例通过。最大绝对误差分别为输出 0.000977、输入梯度 3.05e-5、权重梯度 5.96e-8、更新后权重 0；原始逐例结果及哈希见[阶段 14 manifest](evidence/stage14-20260929/manifest.json)，源码见[`stage14_autograd_integration.py`](stage14_autograd_integration.py)。这证明所测形状的接口组合与梯度传播可行，不含真实模型 step 或吞吐计时。固定 Megatron 提交 `5be9626709af2722333bf54797c954c09edeada3` 的导入探针先受到 CoreX 自带同名 `megatron` 包覆盖；只在临时 overlay 中指定固定 checkout 后，又因当前 `transformer_engine` stub 缺 `__version__` 而停止。两份失败证据分别保留在[探针 JSON](evidence/stage14-20260929/megatron-route-probe-final.json)和[overlay 日志](evidence/stage14-20260929/megatron-import-overlay.log)。

阶段 15 为定位阻断深度，只在探针进程内隔离同名包并映射厂商 `te_version()`（1.6.0）到 `__version__`。固定 checkout 的 detached HEAD 与预期提交一致；越过版本检查后，导入又停在缺少 `transformer_engine.pytorch.float8_tensor`。此模块被固定 Megatron 的若干路径引用，不能据此宣称当前厂商运行时与该 Megatron 提交兼容；也未通过伪造 FP8 类来推进训练。探针源码、两轮原始 JSON 与哈希见[阶段 15 证据目录](evidence/stage15-20260929/manifest.json)。

阶段 18–21 改用固定 Megatron 自带的 **local PyTorch 层规范**：只在探针进程内屏蔽不兼容的 Transformer Engine stub，并把 PyTorch 2.4 已存在的 DTensor 类映射到固定提交期待的导入路径；没有修改厂商包或 Megatron checkout。阶段 19 的真实选中算子路径通过 FP16/BF16 `[8,2,1024]` 及 FP32 `[8,2,256]` 的输出、loss、梯度和权重更新对照。阶段 20 的单层 GPT 以合成 token 完成完整训练 step，hook 捕获到三次连续 FP32 RMSNorm 输入 `[8,2,256]`，stride `[512,256,1]`。这是真实模型代码在**小测试配置**里的调用形状，不是生产工作负载形状；原始 JSON 见[阶段 19](evidence/stage19-20260930/)和[阶段 20](evidence/stage20-20260930/)。

阶段 21 在该单层模型中仅融合 **attention BDA→pre-MLP RMSNorm 一对**；其它层保持固定 Megatron 本地实现。种子 21005–21007 各自独立进程运行一次正确性 step、3 次预热和 12 对交替顺序的 GPU event 计时。三轮均通过输出、loss、更新后参数检查；三个基线/候选中位数分别为 3.7784/3.6944、3.7747/3.6905、3.7748/3.7071 ms，倍率为 **1.0227、1.0228、1.0183×**。17-token 留出形状亦通过，3.8171/3.7328 ms。详细误差、逐次样本和早期探索失败均在[阶段 21 原始数据](evidence/stage21-20260930/)；固定版脚本与 SHA256 回执已进入 skill 的 `exp-bi-v150-corex42-stage21` 来源页。约 2% 的改善很小，仅证明此 FP32 小模型、本地后备路径、单个融合点可工作；不推断生产模型、FP16/BF16 全模型或 Transformer Engine 性能。

阶段 11 的完整留出集如下，数值为三轮配对中位数比值的几何平均；`>1` 才表示比相同 tile 的 stage 1 更快。每轮 8 个 case × 3 个阶段数均通过 CPU FP64 参考后的 dtype 舍入检查，FP16 使用 `atol=rtol=0.03`，BF16 使用 `0.05`；逐轮样本与哈希在 [`evidence/stage11-20260928/`](evidence/stage11-20260928/)。

| dtype | M×N×K | tile | stage 2 / stage 1 | stage 3 / stage 1 |
|---|---|---|---:|---:|
| FP16 | 512×512×2048 | 32×32×32 | 1.0171× | 0.7814× |
| FP16 | 512×512×2048 | 64×64×32 | 1.0188× | 1.0032× |
| FP16 | 1024×1024×1024 | 32×32×32 | 1.0007× | 0.9150× |
| FP16 | 1024×1024×1024 | 64×64×32 | 0.9153× | 0.7451× |
| BF16 | 512×512×2048 | 32×32×32 | 1.0162× | 0.7748× |
| BF16 | 512×512×2048 | 64×64×32 | 1.0060× | 0.9932× |
| BF16 | 1024×1024×1024 | 32×32×32 | 0.9776× | 0.8975× |
| BF16 | 1024×1024×1024 | 64×64×32 | 0.9152× | 0.7484× |

## 4. Skill 与重建验收

最新调度审计来源页包含 **50 件逐项标注 SHA256 的证据文件，安装版复核 50/50 匹配**。从固定上游归档加补丁在 BI-V150 Pod 新目录干净重建后，Linux 全库验证通过：**1055 页、995 个 source ID、37 个 asset bundle、14 个 ledger、0 个孤立源文件**；本机 Windows 安装版同样通过。最新重建版的**完整上游测试 160 项通过**，另行运行的 BI-V150 专项测试 2 项、架构索引测试 6 项也通过。BI-V150/SM90/SM100 精确查询分别为 **14/286/379** 条；最新三份路径集合与前一阶段逐项一致。原始验收日志、查询清单及哈希见[阶段 11 证据目录](evidence/stage11-20260928/)。

Windows 验证器最初用字符串 `/` 判断资产目录归属，误报 411 个孤立文件；改为 `Path.parents` 后全库复验通过。阶段 1 与阶段 11 重建版的完整上游 160 项测试均通过。**2026-09-29 报告收口后的补丁又在 BI-V150 Pod 的全新 Linux 目录从固定上游归档重建，验证器通过、160/160 项测试通过；本机重新查询的 BI-V150/SM90/SM100 完整路径集合与阶段 11 原始清单逐项相同（14/286/379）。** 新增实验不更改原 SM90/SM100 页面及性能数字的适用范围。

阶段 13 将上述反向证据回写到独立来源页与算子页。新补丁从固定上游提交重新构建后，**1756 个文件与已测试版逐项哈希一致**；整库验证为 **1057 页、996 个 source ID、37 个 asset bundle、14 个 ledger、0 个孤立源文件**，BI-V150/SM90/SM100 精确查询为 **16/286/379** 条，SM90/SM100 路径集合不变。BI-V150 Pod Linux 的完整 unittest **160/160 通过**，原始日志见[阶段 13 测试记录](evidence/stage13-20260929/kernelwiki-unittest.log)。Windows 直接运行 unittest 会受本机 GBK 默认编码和 `python3` 执行别名影响，故以同一重建版的 Linux 全量测试为验收。

阶段 14 的新来源和合成前后向结论已加入同一补丁与算子页。从固定上游 SHA 再重建后，**1766 个非缓存文件逐项哈希一致**；整库验证为 **1058 页、997 个 source ID、37 个 asset bundle、14 个 ledger、0 个孤立源文件**。BI-V150 精确查询增至 **17** 条；SM90/SM100 仍为 **286/379** 条，完整路径集合不变。Pod Linux 完整 unittest **160/160 通过**，见[阶段 14 测试日志](evidence/stage14-20260929/kernelwiki-unittest.log)。

阶段 21 将完整模型证据回写为新来源页，固定上游与补丁在本机和 BI-V150 Pod 的 Linux 环境中干净重建成功：**1059 页、998 个 source ID、37 个 asset bundle、14 个 ledger、0 个孤立源文件**，KDA 加载器复制并校验 **1779 个文件**；BI-V150/SM90/SM100 查询路径为 **18/286/379**。Windows 重建版整库校验通过，KDA 加载器 2 项测试通过，项目级安装与三架构查询通过。Linux Pod 的整库校验与完整 unittest **160/160 通过**，原始[校验和测试日志](evidence/stage21-20260930/linux-rebuild/)已取回。

### KDA 的 skill 加载入口

KDA 原仓库的加载方式是把固定的 `skills/KernelWiki` 子模块链接到 `~/.claude/skills/KernelWiki`。阶段 14 曾另用 [`link_kda_skills.ps1`](link_kda_skills.ps1)把适配版链接到本机 `~/.claude/skills/kernelwiki-iluvatar`，直读检索分别返回原 NVIDIA 的 `kernel-flash-attention-4` 和目标端的 `kernel-residual-rmsnorm-bi-v150`；当时适配版 BI-V150 检索为 17 页。当前 KDA checkout 已改为下述仓库级加载器复制的托管副本，BI-V150 检索为 18 页，原第三方子模块没有改动。

2026-09-29 经用户授权，使用 [`claude_paratera_smoke.ps1`](claude_paratera_smoke.ps1) 在 KDA checkout 的项目级 `.claude/skills/` 入口运行一次只读 Claude Code 验证。服务地址为 `https://llmapi.paratera.com`，模型 ID 为 `Claude-Opus-4.8`；Key 由用户在本机配置，未写入仓库或输出日志。会话 2 轮、退出码 0，Claude 的 `Read` 工具实际读取适配版 `wiki/hardware/bi-v150-stack.md`，返回 `hw-bi-v150-stack`、`BI-V150 observed kernel toolchain`、`experimental`、首个来源 `doc-flagtree-iluvatar`，与页面 frontmatter 一致。原始响应见[Claude Code smoke JSON](evidence/claude-paratera-smoke-20260929-153202.json)，工具调用路径和本机会话记录哈希见[精简 trace](evidence/claude-paratera-smoke-20260929-153202-trace.json)。这证明当前机器上的适配 skill 可由 Claude Code 调用并读页；KDA 仓库正式集成、真实算子选择与训练收益仍按第 5–6 节验收。

阶段 13 安装版同步后，Claude Code 又从同一入口实读新的 `kernel-residual-rmsnorm-backward-bi-v150` 页面，返回来源 `exp-bi-v150-corex42-stage13`、BF16 64×1024 的 **15.405392×** 后向微基准值和“非完整训练吞吐”的边界；原始响应见[反向页实读记录](evidence/claude-paratera-backward-page-20260929.json)，本机会话的[精简 trace](evidence/claude-paratera-backward-page-20260929-trace.json)记录了实际 `Read` 路径和 transcript 哈希。模型规划阶段的只读输出见[规划响应](evidence/claude-paratera-backward-plan-20260929.json)，其中公式与来源经本次 GPU 实验独立检验；这仍是个人及项目级 skill 入口的闭环，尚非 KDA 上游仓库的发布或 CI 集成。

阶段 14 安装版同步后，Claude Code 从 KDA 项目级入口实读新增来源页与反向算子页，准确返回 `exp-bi-v150-corex42-stage14`、**18 个合成 case-round**、当时的 Megatron/Transformer Engine 导入阻断和阶段 13 微基准的适用边界；会话退出码 0、3 轮，原始响应见[阶段 14 实读记录](evidence/stage14-20260929/claude-stage14-read.json)，[精简 trace](evidence/stage14-20260929/claude-stage14-read-trace.json)保留了两个实际 `Read` 路径和会话哈希。此后阶段 21 已补充小模型完整 step 的实验，见上节。

Claude Code 用户配置现已指向 Paratera，旧配置备份留在用户目录；API Key 只在本机用户配置与环境变量中。清除当前进程旧代理变量后，普通 `claude -p` 对 `Claude-Opus-4.8` 的最小调用返回 `OK`，见[默认配置验证 JSON](evidence/claude-paratera-global-config-20260929.json)。密钥不在本仓库及这些响应文件中。

2026-09-30 已准备 KDA 仓库级通用加载器：[`kda-loader.patch`](kda-loader.patch) 固定在 KDA 提交 `ef6ce617693ef0782b3ecb9f37e39bbf10226a90` 上，可由 [`build_kda_integration.py`](build_kda_integration.py)重建。加载器把下游已准备的 skill 复制到 KDA checkout 的 `.claude/skills/kernelwiki-iluvatar`，逐文件 SHA256 核验、拒绝覆盖非托管目录或链接；KDA 原第三方 `skills/KernelWiki` 子模块保持不变。KDA checkout 的 Windows 2 项加载器测试通过，独立重建 checkout 的项目级加载与 18/286/379 架构检索通过。GPGPU 的 [CI 工作流](../../.github/workflows/kernelwiki-iluvatar-kda.yml)会从固定 KDA 与 KernelWiki 提交重建、跑加载器测试、skill 整库验证、160 项上游测试及项目级加载查询；KDA 补丁内也有 Ubuntu/Windows 加载器 CI。Claude Code 已从此**仓库托管副本**实际 `Read` 阶段 14 来源页并回答其边界，成功响应与 trace 见[项目级实读证据](evidence/kda-repo-integration-20260929/)。这套方案不依赖本机个人 skill 安装。KDA 上游 PR/合并状态与 CI 线上运行结果仍须单独核实。

最终阶段 21 版本已更新到同一 KDA checkout 的项目技能目录。Claude Code 使用该入口读取新来源页后返回准确的来源 ID、`[8,2,256]`/stride `[512,256,1]`、三轮 1.0227/1.0228/1.0183 的比值及非生产训练的限制；[只读响应](evidence/kda-repo-integration-20260930/claude-stage21-read.json)退出码 0、3 轮。新加载器还将生成的 `.claude/skills/` 忽略于 Git，避免把外部 skill 资产误加入 KDA 提交。KDA 代码差异只包含通用加载器、其测试/CI、说明和忽略规则。

## 5. 未完成项与外部依赖

1. **生产配置与兼容运行时。** 固定 Megatron 的一层本地 GPT 后备路径已实测，但输入是合成 token、仅 FP32，且只融合一对 BDA→RMSNorm；捕获的 `[8,2,256]` 属于测试配置。当前 Pod 的 Transformer Engine stub 仍与固定提交不兼容。需要生产训练配置或其可公开的最小代表配置、兼容运行时，再采集真实形状分布，比较多层、实际精度、loss 曲线、吞吐和显存。
2. **ixSYS 可视化。** CLI V4.2.0 与 NVTX 短脚本可用，但 Pod 缺少 tracefs/debugfs 的 `tracing_on` 控制节点，容器内挂载因只读限制失败。2026-09-29 再测确认内核列出 tracefs/debugfs，而容器无 `CAP_SYS_ADMIN`、`/sys` 为只读；`mount` 返回 32，ixSYS 最窄的 `cuda_kernel` 模式仍在采集前退出，未生成 `.ptrace`。原始诊断见 [`evidence/stage12-20260929/tracefs-probe.log`](evidence/stage12-20260929/tracefs-probe.log)。JSON/日志不能代替 trace。平台需在宿主机或 Pod 创建配置中提供可访问的 tracefs/debugfs 节点，再按第 8 节采集、拷回和导入。
3. **硬件机制对应关系。** 尚未取得可读最终机器指令、计算/搬运重叠 trace、shared bank 冲突或 warp 角色拆分证据。原 NVIDIA TMA、mbarrier、TMEM、`tcgen05`、CLC、NVFP4 等仍限定原架构；当前目标端相似算法只按所测证据等级描述。
4. **FP8/软件版本。** 当前 CoreX 4.2 的 FP8 `tl.dot` 编译失败；不能推出 BI-V150 硬件不支持。新 CoreX/Triton 构建须作为独立环境版本重新探测，不与现有性能样本合并。
5. **KDA 上游发布。** KDA checkout 的通用加载器、Ubuntu/Windows CI 定义和 GPGPU 的重建/全库 CI 定义已准备并本地验收；KDA 上游尚未有 PR/合并。KDA 旧子模块提交 `76d27b56f804e7e7295d4c570e1e5d7eef4b0a75` 保持原样，BI-V150 扩充由外部固定提交加补丁生成。正式上游提交需符合 KDA 的 DCO 签署与贡献流程。

## 6. 下一步执行计划

| 优先级 | 任务与依赖 | 具体执行 | 完成门槛与产物 |
|---|---|---|---|
| P0 | KDA 上游发布与 CI 绿灯 | 审阅已准备的通用加载器补丁和 DCO 要求；发布可审阅分支/PR，并观察 GPGPU 与 KDA 的 CI 真正跑通 | 上游接受或明确反馈；两端 CI 日志可追溯，旧第三方子模块不变 |
| P1 | 生产形状与兼容 Transformer Engine | 取得实际训练配置及目标端兼容运行时；在多层训练中捕获每个 Residual Add RMSNorm 的 rows/hidden、dtype、stride、调用次数、前/反向耗时与显存；固定调参/留出形状 | 原始 capture、环境指纹和哈希；明确阶段 21 小模型形状的覆盖范围 |
| P1 | 扩展端到端验收 | 将阶段 21 的单对融合扩到所有合法调用点，比较实际精度和多层模型的 loss、吞吐、显存；对回退和尾块复测 | 三次独立复测、留出形状正确；短训练 loss 通过；明确净收益与退化尾部 |
| P2 | 目标端流水机制；依赖平台 tracefs/debugfs 挂载或厂商最终指令工具 | 用已准备的 ixSYS NVTX 脚本生成 `.ptrace`，在网站检查 kernel/stream 时间线；对 `num_stages` 的最终代码、屏障与搬运重叠做证据对照 | trace 可打开且来源、版本和 workload 可追溯；只在观察到对应机制后提升迁移账本的硬件映射等级 |
| P2 | 低精度与其他热点；依赖新工具链或真实模型热点排序 | 在新 CoreX 构建重新测 FP8/FP4 表示、缩放与 `tl.dot`；按真实热点顺序扩至 SwiGLU、Attention、Cross Entropy 或 MoE Grouped GEMM | 每项独立环境指纹、数值与性能闭环；FP8 编译失败/成功均保留日志；不跨版本借用旧结论 |

**执行顺序：** 最新补丁验收已完成，先推进 P0 的 KDA 仓库集成和 P1 的真实形状采集，再做训练接入。P2 的 ixSYS 和低精度任务可在相应平台条件具备后并行启动；条件未具备时继续保持“未验证”，不补写推测。所有新结果必须回写下列迁移账本、skill 来源页及原始证据目录，并复查 SM90/SM100 查询路径集合。

## 7. Hopper/Blackwell → BI-V150 逐项迁移账本

| 上游技术页 | 可迁移的思路 | BI-V150 当前证据 | 下一项验证 |
|---|---|---|---|
| `technique-kernel-fusion` | 合并相邻算子以减少中间张量和 launch | Residual Add RMSNorm 前向在 FP16/BF16、5 个形状上通过三次独立进程复测；阶段 13 两核反向 72 个 case-round 通过；阶段 14 合成前后向及更新 18/18 正确；阶段 21 固定 Megatron 的单层 FP32 GPT 完整 step 融合一对 BDA→RMSNorm，三次独立模型对照通过且中位耗时比 1.0183–1.0228，仅限合成 token 小模型 | 捕获生产形状和精度；与兼容优化运行时、多层完整训练及其它目标端优化反向比较 |
| `technique-epilogue-fusion` | 在结果写回前合并缩放、加法或激活 | FP16/BF16 GEMM+bias 三形状、四配置、三轮均正确；相同 tile 相对单独 bias launch 加速 1.33–3.94× | 对比厂商高性能 GEMM 与完整模型调用路径 |
| `technique-tile-scheduling` | 按形状调 tile 和 launch 布局 | GEMM+bias 四配置三轮复测：127/256 方阵偏向 32×64×32/8，512 方阵偏向 64×64×32/4；仅限所测形状 | 扩大 K/N 和布局，再查编译资源与性能稳定性 |
| `technique-register-budgeting` | 控制每程序资源占用 | CoreX 编译对象可报告 n_regs/n_spills/shared；访问核 block 256/1024/4096 分别为 3/8/26 个寄存器、零 spill，耗时非单调 | 取得 occupancy 和生成代码后再推导资源阈值 |
| `technique-vectorized-loads` | 连续访存和向量化 | FP16 仿射核连续/stride 2、两规模、四配置三轮均正确且已计时；尚无目标指令向量宽度证据 | 留存生成代码并分析实际 load 指令及带宽 |
| `technique-cache-policy` | 基于复用模式调整缓存行为 | tl.load 的默认、.ca、.cg 在当前构建均编译且正确；最小拷贝未见稳定收益 | 对照生成代码、缓存语义和真实复用负载 |
| `technique-double-buffering` | 计算与数据搬运重叠 | stages 2/3 的目标编译 TTGIR 有 insert_slice_async/commit/wait 和 2/3 层 shared；FP16/BF16 与两种 tile 留出三轮均正确，但性能随形状和 tile 变，未证实实际计算/搬运重叠 | 查最终机器指令与重叠 profiler，测试真实模型负载 |
| `technique-pipeline-stages` | 分阶段预取、计算、回收缓冲 | GEMM stages 1/2/3 均正确；32 tile 的 shared 4/8/12 KiB，64 tile 为 8/16/24 KiB；stage 2 在 BF16 长 K、32 tile 为 1.0162×，在 BF16 1024³、64 tile 为 0.9152×；stage 3 后一 case 为 0.7484× | 按 dtype/shape/tile 联合选参，取得最终指令和真实负载证据 |
| `technique-swizzling` | 降低局部存储冲突 | tl.swizzle2d 的程序 tile 重排三形状三轮正确，收益不稳定；它并非共享内存 bank swizzle | 另行取目标端共享存储布局、bank 冲突与指令证据 |
| `technique-warp-specialization` | 分配不同线程组角色 | enable_warp_specialization=True 在 256³ GEMM 可编译，但 TTGIR/LLVM IR 与默认逐字相同；当前算例未见角色拆分 | 查厂商专用入口与实际角色分工/同步生成代码 |
| `technique-ping-pong-scheduling` | 交替缓冲或工作组 | 双累加器交替 K 块的 GEMM 三形状三轮正确；512³ 约 0.937×，非 TMEM ping-pong | 若厂商有对应同步/局部存储机制，再验证目标实现 |
| `technique-persistent-kernels` | 复用驻留工作块 | 固定较小网格循环遍历 tile 的 FP16 加法两形状三轮正确，但比普通网格慢；无 CLC 使用 | 在计算密集 GEMM 上验证，仍须区分 SM100 CLC |
| `technique-fine-grained-quantization` | 分块缩放与低精度计算 | FP8 张量可分配，但当前 Triton 的 e4m3/e5m2 tl.dot 四个小 GEMM 均编译失败；FP4/分块缩放未验证 | 核对厂商低精度编程接口，分离格式转换、缩放、计算路径 |
| `technique-software-exp` | 用近似函数换吞吐 | tl.exp2 在此版本缺失；[-1,0] 六阶 Taylor 通过误差门槛但 1M 较慢、4M 无稳定收益 | 若需 softmax，按完整输入域和归一化误差重新验收 |
| `technique-chunk-parallelism` | 沿序列分块并行 | 两段式分块 scan 对 4099/65537 元素三轮数值通过，但仅为 torch.cumsum 耗时的 0.30–0.36× 加速比 | 减少第二遍块前缀代价，测试长序列与 GDN 语义 |
| `technique-cccl-memory-primitives` | 扫描等算法可重实现 | tl.cumsum/tl.sum 的目标端 scan 原型正确但较框架实现慢；未使用 CCCL | 优化目标端 scan，保留 CCCL 页面原 CUDA 范围 |
| `technique-external-source-map-research` | 检索方法，无 kernel 实现 | 不适用硬件迁移 | 继续用于来源审计 |

### NVIDIA 硬件机制

原 skill 的 `hw-tma`、`hw-tmem`、`hw-tcgen05-mma`、`hw-clc`、`hw-mbarrier`、`hw-pdl-gdc`、`hw-2sm-cooperative`、`hw-nvfp4` 页面保留 NVIDIA 适用范围。当前本地证据没有确认 BI-V150 对应指令、存储区、同步协议或精度格式。不得将这些名称直接改标 `bi-v150`。若厂商资料描述相近能力，先记录软件版本和具体 API，再用编译、正确性及生成代码证据建立关系。

## 8. ixSYS trace 采集与上传

依据用户提供的厂商文档 `ixSYS.pdf`（本地参考附件）第 4、14–18 页。图形界面入口为 <https://ui.ixsys.iluvatar.com>。ixSYS **读取其 CLI 采集的 `.ptrace` trace**；本项目的 `round*.json`、`*.log` 和 `manifest.json` 是可复核的数值/计时证据，不能直接当作 trace 上传。

### 当前 Pod 状态

`/usr/local/corex/bin/ixsys --version` 为 V4.2.0，短时工作负载 [`stage4_trace_case.py`](stage4_trace_case.py) 可直接运行。但 `ixsys -t cuda,nvtx` 和更窄的 `-t cuda_kernel` 均提示无法访问 ftrace 控制节点，`/sys/kernel/tracing/tracing_on` 与 `/sys/kernel/debug/tracing/tracing_on` 均不存在；容器无 `CAP_SYS_ADMIN`，`/sys` 只读，实际挂载尝试失败。故 **目前没有 `.ptrace` 文件**。需在平台创建/配置 Pod 时提供可访问的 tracefs/debugfs 挂载，或由平台管理员按文档完成宿主机挂载与容器映射；仅在容器内重试 ixSYS 无法解决。

### 平台挂载恢复后采集

先在 Pod 里确认 `test -r /sys/kernel/tracing/tracing_on` 或 `test -r /sys/kernel/debug/tracing/tracing_on` 成功，然后用当前已验证的 CoreX 环境执行：

```bash
cd /private/kda-biv150-stage3
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/openmpi/lib
export PATH=/usr/local/corex/bin:/root/miniconda3/bin:$PATH
ixsys -t cuda,nvtx -o stage4-gemm-epilogue.ptrace \
  /root/miniconda3/bin/python -u stage4_trace_case.py --iterations 20
ls -lh stage4-gemm-epilogue.ptrace
```

短脚本对 256³ FP16 的分离 GEMM+bias 和融合 epilogue 各执行 20 次，并用 NVTX 范围标记两段。采集开销会扰动耗时，因此性能数字仍以阶段 4 JSON/event 复测为准；trace 用于查看 launch、kernel 时间线与内存活动。

### 在 Windows 浏览器查看

若 trace **小于 400 MB**，在本机 PowerShell 将文件取回：

```powershell
scp bi-v150:/private/kda-biv150-stage3/stage4-gemm-epilogue.ptrace E:\code\GPGPU\experiments\kernelwiki_iluvatar\evidence\stage4-20260928\
```

然后访问 <https://ui.ixsys.iluvatar.com>，点击 **Open trace file**，选择 `E:\code\GPGPU\experiments\kernelwiki_iluvatar\evidence\stage4-20260928\stage4-gemm-epilogue.ptrace`。在时间线中查找 `unfused_gemm_plus_bias` 和 `fused_gemm_bias_epilogue` 两个 NVTX 范围，展开 CUDA kernels/stream，圈选区域查看统计。

该网址来自项目所附 2024 年版厂商 PDF；本次环境访问站点超时，无法确认网站当前在线状态。若页面显示离线模式或版本不匹配，按 PDF 第 25–26 页切换 **SW On** 离线模式并重新加载。先确认 `.ptrace` 已由 ixSYS CLI 生成，再排查网页加载。

若 trace **大于 400 MB**，按 PDF 第 16–18 页在界面下载 Windows `trace_processor_shell`，在本机对已拷回的 trace 运行：

```powershell
.\trace_processor_shell.exe --httpd E:\code\GPGPU\experiments\kernelwiki_iluvatar\evidence\stage4-20260928\stage4-gemm-epilogue.ptrace
```

打开或刷新网站根地址，选择 **Yes, use loaded trace**。文档注明超过 2 GB 加载会较慢，超过 3 GB 不受支持。若文件留在远端，可用 `ssh -N -L 19001:127.0.0.1:19001 bi-v150` 建立端口转发，并在远端运行 Linux 版 `trace_processor_shell --httpd /private/kda-biv150-stage3/stage4-gemm-epilogue.ptrace`；当前连接可 SSH 直达，无需文档中的 Nginx 跳板方案。

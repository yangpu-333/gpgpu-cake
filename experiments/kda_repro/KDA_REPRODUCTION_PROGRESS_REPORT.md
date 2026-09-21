# KDA 复现工作详细进展报告

日期：2026-09-21  
当前代码提交：`c09298ac27da044a562ab164233c65f04f97d336`  
主要实验设备：Tesla V100-PCIE-32GB

## 1. 执行结论

项目已经在 V100 上完成一个可审计的 KDA 适配闭环：模型提出局部 Triton 修改，控制器生成候选，
依次执行静态检查、冒烟验证、官方 workload 全量正确性验证、同进程配对计时、独立重复复测，
最后根据固定门槛自动晋级或淘汰候选，并保存源码、API 元数据、原始计时和决策账本。

目前最重要的两个结果是：

1. **GDN Decode：** 54/54 个官方 workload 通过。Agent 候选正确且方向上略快，但三轮综合改善
   只有 `0.46%`，低于1%门槛，最终未晋级。
2. **GDN Prefill：** 100/100 个官方 workload 和2个边界分支通过。Agent 候选在三轮、300组
   配对测量中的综合改善为 `1.454%`，超过1%门槛并正式晋级。

若范围限定为“在现有 V100 上证明 KDA 方法闭环可运行”，定性完成度约为 **85%**。若范围扩大为
“五项官方任务、官方 B200 运行环境、ncu 硬件分析和国产 GPU 迁移”，总体完成度约为
**40%–50%**。这两个数字对应不同的验收范围，不能混用。

## 2. 当前目标和复现边界

当前阶段目标已经从早期 CAKE/Megatron 原型转为 Path4：先在 NVIDIA V100 上复现 KDA 的核心
工程循环，再选择适合的算子迁移到 BI-V150。当前成果属于本项目参照 KDA 思路实现的受控 Agent
控制器，不等同于 NVLabs 官方 Agent 工具链，也不等同于 B200 竞赛成绩。

本阶段遵守以下边界：

- 使用公开任务定义、公开 workload 和公开 reference；
- 不读取最终竞赛解答仓库作为 Agent 输入；
- V100 没有原生 BF16/FP8 高性能路径时，明确标为 V100 语义适配；
- 不用放宽误差、削弱基线或跨时间单次计时制造加速；
- 只有正确候选才能计时，只有重复配对结果达到1%门槛才能晋级。

## 3. 固定的软件、源码和数据

### 3.1 运行环境

| 项目 | 当前环境 |
|---|---|
| GPU | Tesla V100-PCIE-32GB，SM70 |
| Python | 3.10.12 |
| PyTorch | 2.5.1+cu121 |
| Triton | 3.1.0 |
| CUDA 编译工具 | CUDA 12.1，`nvcc` 已安装 |
| 性能分析工具 | `ncu` 已安装，但计数器权限被平台拒绝 |
| 模型服务 | 学校 OpenAI Chat Completions 兼容 API |
| 已用模型标识 | `Claude-Opus-4.8` |

### 3.2 上游源码锁定

| 仓库 | 固定提交 |
|---|---|
| NVLabs/kda | `ef6ce617693ef0782b3ecb9f37e39bbf10226a90` |
| mlsys2026-flashinfer-contest | `d61573ffb2008b6e68fb32fe8e17703e4b2448c9` |
| flashinfer-bench | `40e6ca7844b514eb4b1c7edba6d6a7377df57870` |
| flashinfer-bench-starter-kit | `75ccd05cafceb0fd1f86be4cd0f2117249463c66` |
| DeepGEMM | `78b69000794d0937b47ae3387eff7663410264d1` |

实验数据固定为 `flashinfer-ai/mlsys26-contest` 修订
`5e832ce88dd1013032b22a0e942979d7d2769cc3`。

## 4. 已实现的自动优化闭环

当前控制器按以下顺序执行：

1. 读取任务契约、实现计划、当前最佳候选和验证边界；
2. 请求模型返回少量精确源码替换，不允许直接覆盖验证器或 reference；
3. 将替换应用到已审核源码，重新检查完整候选；
4. 执行 AST 静态限制，拒绝文件、网络、进程等越界调用；
5. 在独立子进程中执行4组冒烟验证；
6. 对全部官方 workload 做逐元素正确性检查；
7. 在同一进程中交替测量当前最佳与新候选，每组各取21个样本；
8. 对初次达到门槛的候选再做两轮独立完整复测；
9. 汇总三轮数据，按至少1%改善决定 `promoted` 或 `demoted`；
10. 保存不可变候选源码、源码哈希、API token 元数据、原始报告和候选账本。

控制器已经处理 API 空内容、JSON mode 不兼容、输出 token 截断、错误候选、中断续跑和重复复核。
验证子进程会清除模型 API 密钥环境变量。API 密钥只保存在项目目录之外、权限为600的环境文件中。

## 5. 逐项实验结果

### 5.1 Residual Add RMSNorm 通用闭环

这是早期用于验证 KDA 通用流程的训练算子。严格复核覆盖6种形状、两个前向输出以及三种随机
上游梯度组合，比较输入、残差和权重梯度，容差固定为 `atol=rtol=0.002`。

候选相对未融合 PyTorch 表达式的加速范围为 `5.60x–10.13x`。该结果证明融合 kernel、严格
梯度检查和证据记录流程可用；它不是相对厂商 RMSNorm 库的性能结论，也没有完成 Megatron
真实训练集成。

### 5.2 GDN Decode 官方输入适配

| 项目 | 结果 |
|---|---:|
| 官方 workload | 54/54通过 |
| 额外边界分支 | `state=None`、`scale=None`、`scale=0` 全部通过 |
| Triton 调度 | ROWS=1/4/8/16均正确 |
| 最终静态调度 | ROWS=8赢47组，ROWS=4赢7组 |
| 相对批量 PyTorch 几何平均加速 | `11.68x` |
| 最大 output 绝对误差 | `7.6294e-6` |
| 最大 state 绝对误差 | `5.7220e-6` |

V100 不提供原生 BF16 算术，因此候选保持 BF16 外部契约，用整数位转换读写 BF16，内部使用
FP32。`11.68x` 的基线是批量 PyTorch 表达式，不是 B200 FlashInfer 优化实现。

### 5.3 GDN Prefill 官方输入适配

| 项目 | 结果 |
|---|---:|
| 官方 workload | 100/100通过 |
| 输入范围 | 总序列长度6–8192，序列数1–57 |
| 调度验证 | ROWS=4/8/16共300次验证全部正确 |
| 固定 ROWS=8 几何平均 | `0.3179 ms` |
| 逐 workload 选择最快调度 | `0.3096 ms` |
| 最大 output 绝对误差 | `1.2207e-4` |
| 最大 state 绝对误差 | `6.5327e-5` |

Prefill 按时间维严格顺序执行递推，只在 V 维切分并行，避免改变算法语义。

### 5.4 Decode Agent 候选

模型把状态更新进行代数化简，减少循环内逐元素运算。候选通过54/54 workload 和3个额外分支。
首次跨运行比较显示约 `1.10%` 改善，但严格的三轮同进程配对改善只有 `0.55%`、`0.27%`、
`0.56%`，162组聚合改善为 `0.46%`。控制器因此将它降级，继续保留种子候选为最佳基线。

该结果验证了自动淘汰机制：模型生成的正确代码不因单次计时更快而自动视为优化成功。

### 5.5 Prefill Agent 候选

模型将：

```python
new_v = beta * v + (1. - beta) * old_v
state = old - k * old_v[:, None] + k * new_v[:, None]
```

化简为：

```python
delta = beta * (v - old_v)
state = old + k * delta[:, None]
```

三轮完整配对结果如下：

| 轮次 | 基线几何平均 | 候选几何平均 | 改善 | 候选胜出 |
|---|---:|---:|---:|---:|
| 首轮 | 0.312800 ms | 0.308359 ms | 1.420% | 80/100 |
| 复测1 | 0.313259 ms | 0.308657 ms | 1.469% | 84/100 |
| 复测2 | 0.312590 ms | 0.307989 ms | 1.472% | 84/100 |
| 三轮合并 | 0.312883 ms | 0.308335 ms | **1.454%** | **248/300** |

候选源码 SHA256 为
`5491375fc7a3bc331bd16910f8ae15ba0249ea3dd38cf276a631f31fba890e9b`。三轮方向一致，综合结果
超过1%门槛，因此 `0001-agent` 已正式晋级。

## 6. 五项官方任务覆盖情况

| 任务 | workload | 当前状态 | 能否形成 V100 性能结论 |
|---|---:|---|---|
| GDN Decode | 54 | 全量正确性、计时、Agent候选完成 | 可以，但只与V100适配基线比较 |
| GDN Prefill | 100 | 全量正确性、计时、Agent晋级完成 | 可以，但不是B200官方成绩 |
| DSA TopK Indexer | 128 | 已审计定义；尚未实现FP8软件解码验证 | 不能，V100无原生FP8路径 |
| DSA Sparse Attention | 23 | 已审计定义；尚未实现候选 | 可做可信BF16/FP32适配实验 |
| FP8 MoE | 19 | 已审计定义；尚未做小规模语义验证 | 不能对齐DeepGEMM性能 |

按任务数量计算，目前完整执行2/5项，即 **40%**。这也是完整项目进度低于 V100 核心闭环进度的
主要原因。

## 7. “V100闭环完成85%”的构成

该比例是工程估计，不是官方指标。已完成的约85%包括任务契约、候选生成、静态限制、冒烟验证、
全量正确性、配对计时、重复复测、晋级账本、失败恢复和证据归档。

剩余约15%如下：

| 缺口 | 估计占比 | 当前情况 | 完成条件 |
|---|---:|---|---|
| ncu硬件性能反馈 | 5% | `ERR_NVGPUCTRPERM` | 管理员开放计数器；基线和候选均取得profile |
| 多轮持续自主搜索 | 5% | 已有一次Prefill晋级，尚未形成长链搜索 | 固定预算运行多轮，Agent利用历史结果继续提案 |
| 全新容器一键验收 | 3% | 各步骤已分别复跑 | 新V100容器从零执行安装、下载、Agent和三轮复测 |
| 故障恢复覆盖 | 2% | 已覆盖主要API和候选错误 | 补充网络中断、GPU退出、损坏报告和自动恢复测试 |

## 8. 证据与可复现性

仓库保存了以下证据：

- `evidence/official-v100/decode-full.json`：Decode全量输入、误差、显存和计时；
- `evidence/official-v100/prefill-full.json`：Prefill三种调度的全量结果；
- `evidence/agent-v100/0003-agent/`：Decode Agent候选和三轮复测；
- `evidence/agent-v100/prefill-0001-agent/`：Prefill Agent候选、API元数据和三轮复测；
- `evidence/v100-strict/strict-v3.json`：Residual Add RMSNorm严格前向和梯度复核；
- `evidence/v100-strict/ncu-attempt.log`：性能计数器权限失败证据。

所有 Agent 候选均保存源码哈希。模型 API 元数据中保存模型标识、token 用量和结束原因，不保存
API 密钥。配对复核脚本是幂等的，重复使用同一批报告不会重复追加候选决策。

## 9. 尚未解决的问题

1. `ncu` 计数器权限缺失，尚不能说明 Prefill 改善具体来自指令数、寄存器、带宽还是调度变化。
2. 当前只有一个 Agent 候选达到稳定晋级，尚不足以统计 Agent 搜索成功率和平均调用成本。
3. 没有 B200，无法运行官方目标架构上的完整依赖和 evaluator。
4. DSA TopK 和 FP8 MoE 在 V100 上只能做语义模拟，不能形成原生 FP8 性能结论。
5. 尚未将 KDA 自动优化闭环迁移到 BI-V150；FlagGems 的 Iluvatar 后端还未成为本轮实测基线。
6. 当前结果是算子微基准，尚未证明端到端模型训练或推理吞吐提升。

## 10. 下一阶段建议与验收标准

### 第一优先级：DSA Sparse Attention

- 下载并固定23个官方 workload 的实际输入；
- 先完成 reference 和最小候选逐元素比较；
- 实现一个 V100 Triton 融合候选；
- 保存逐 workload 误差、显存和交替计时；
- 接入现有 Agent 控制器和1%晋级规则。

验收条件：23/23 workload 正确，边界分支明确，至少三轮配对复测，任何性能结论都注明基线范围。

### 第二优先级：补齐 V100 闭环15%

- 向平台管理员申请 NVIDIA 性能计数器权限；
- 对 Prefill 种子和晋级候选各采集一份相同参数的 ncu profile；
- 在全新 V100 容器运行一次完整的一键复现；
- 固定 API 调用预算，连续运行多轮候选搜索并统计有效率。

### 第三优先级：目标硬件与国产迁移

- 获得 B200/Hopper 后按官方锁定环境运行 evaluator；
- 在 BI-V150 上先选择 Residual Add RMSNorm 或 FlagGems 已覆盖算子验证迁移流程；
- NVIDIA 专用候选不直接复制到天数，按其编译器、运行时和 Triton 后端重新生成和评测。

## 11. 复跑入口

官方 GDN Decode 和 Prefill 的 V100 适配复跑：

```bash
cd "$HOME/gpgpu-cake"
git pull --ff-only origin main
KDA_CA_BUNDLE="$HOME/.local/share/ca-certificates/scholar-git-ca-bundle.pem" \
  bash experiments/kda_repro/scripts/run_official_v100.sh all
```

运行 Prefill Agent：

```bash
nohup bash experiments/kda_repro/scripts/run_gdn_prefill_agent.sh 1 \
  > "$HOME/kda-repro/agent-prefill.log" 2>&1 < /dev/null &
```

对达到初步门槛的候选执行三轮复核：

```bash
nohup bash experiments/kda_repro/scripts/run_paired_review.sh \
  prefill 0001-agent 0000-seed 2 \
  > "$HOME/kda-repro/prefill-paired-review.log" 2>&1 < /dev/null &
```

## 12. 总结

当前已经完成了最关键的可行性证明：LLM Agent 能在受控范围内修改真实 Triton 内核，验证器能
拒绝错误或不稳定候选，并在 V100 官方 workload 上产生一个三轮稳定、达到门槛的 Prefill 优化。
后续工作的重点已经从“闭环是否可行”转为“扩大任务覆盖、加入硬件 profile、在全新环境验收，
并迁移到 B200 和国产 GPU”。

# 普通 PyTorch 基线到 Triton 的结构化 Agent 实验

日期：2026-09-22。设备为 Tesla V100-PCIE-32GB；数据固定为
`flashinfer-ai/mlsys26-contest` 修订 `5e832ce88dd1013032b22a0e942979d7d2769cc3`。

## 结果

本实验把 GDN Decode 的普通 eager PyTorch 表达式作为候选 `0000-seed`。模型先分析内存流量和
Volta 调度风险，再选择规范化设计：单 Triton 内核融合、`ROWS=4`、`num_warps=4`，并使用等价
的递推化简。控制器将该设计降为经过审查的 sm70 Triton 模板，随后执行静态检查、4 组冒烟、
54 组官方 workload 和 3 个额外分支验证。

| 项目 | 结果 |
|---|---:|
| 模型 | `Claude-Opus-4.8`（学校兼容 API 返回标识） |
| 普通 PyTorch 独立基线 | `0.264630 ms` |
| 同轮配对 PyTorch 基线 | `0.256053 ms` |
| Agent Triton 候选 | `0.023308 ms` |
| 配对加速比 | **`10.985×`** |
| 正确 workload | `54/54` |
| 候选胜出 | `54/54` |
| 额外分支 | `3/3` |
| 最大 output 绝对误差 | `7.6294e-6` |
| 最大 state 绝对误差 | `5.7220e-6` |
| 候选源码 SHA256 | `be5fb6683d9404d53a9ed7378d557d06ef18e4a1ac8a8e3ddd310d687d52a028` |

结构化设计请求首次采用 JSON mode 时触及 1200 completion-token 上限；控制器自动使用紧凑重试，
成功响应使用 1526 个输入 token 和 1093 个输出 token。成功设计随后通过全部验证并自动晋级。

## 为什么结构化接口有效

直接要求模型返回整份 Triton 源码的第一次尝试失败：两次请求各达到 6000 completion tokens，
服务均返回空内容。结构化接口让模型只选择可解释的实现策略和硬件调度，确定性控制器负责模板
降级、编译和验证，避免长代码被接口截断。这与 CAKE 的核心思路一致：把生成空间约束为可验证、
可执行的优化动作。

模型给出的主要理由是：普通实现会物化多个完整 FP32 state 中间张量；融合内核可将门控、归约、
秩一更新和输出投影合并，显著减少状态矩阵的显存往返。`ROWS=4` 用较多程序块覆盖 V100 的 SM，
同时控制寄存器压力。

## 证据与边界

完整证据位于
[`evidence/agent-v100/decode-ordinary-structured`](evidence/agent-v100/decode-ordinary-structured)：

- `candidates/0000-seed/candidate.py`：普通 PyTorch 起点；
- `docs/draft.md`：模型生成的优化分析和候选排序；
- `api/0001-agent.json`：自由整核生成被长度限制拒绝的记录；
- `api/0002-agent.json`：结构化设计与 token 元数据；
- `candidates/0002-agent/candidate.py`：控制器降级后的候选；
- `runs/0002-agent-smoke.json` 与 `runs/0002-agent-full.json`：原始验证和配对计时；
- `candidates.jsonl`：不可变晋级账本。

`10.985×` 是相对普通 eager PyTorch 的流程内总提升。候选的 Triton 骨架来自项目内经过审查的
受限模板，Agent负责选择融合策略、调度参数和递推变换；这不是模型从空白自主写出整份内核。
该候选源码哈希与此前强 Triton 基线上的正确候选相同，因此相对强 Triton 基线的额外收益仍应
引用三轮配对复核所得的 `0.46%`，不能把 `10.985×`解释为 Triton 对 Triton 的提升。

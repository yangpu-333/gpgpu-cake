# KDA Agent GDN Prefill V100 结果

日期：2026-09-21。设备为 Tesla V100-PCIE-32GB；数据固定为
`flashinfer-ai/mlsys26-contest` 修订 `5e832ce88dd1013032b22a0e942979d7d2769cc3`。

## 结果

学校模型 API 生成了一处受限的 Triton 源码替换。控制器完成静态检查、4组冒烟验证、100组完整
workload 验证和两项边界分支验证。随后用三轮同进程、交替顺序的配对计时复核候选；候选在三轮
合并后达到预设的1%晋级门槛，正式晋级为 Prefill 当前最佳候选。

| 项目 | 结果 |
|---|---:|
| 模型 | `Claude-Opus-4.8`（学校兼容 API 返回标识） |
| API 请求 | 首次成功，1406 completion tokens，`finish_reason=stop` |
| 每轮正确 workload | 100/100 |
| 每轮边界分支 | 2/2 |
| 三轮配对测量 | 300组，每个候选每组21次交替样本 |
| 基线几何平均 | `0.312883 ms` |
| Agent 候选几何平均 | `0.308335 ms` |
| 综合提升 | `1.454%`（`1.01475x`） |
| 候选胜出 | 248/300组 |
| 最大绝对误差 | `1.2207e-4` |
| 候选源码 SHA256 | `5491375fc7a3bc331bd16910f8ae15ba0249ea3dd38cf276a631f31fba890e9b` |

三轮独立改善分别为 `1.420%`、`1.469%` 和 `1.472%`。结果方向一致，且综合结果超过1%门槛。

## 模型修改

模型把循环内状态更新：

```python
new_v = beta * v + (1. - beta) * old_v
state = old - k * old_v[:, None] + k * new_v[:, None]
```

代数化简为：

```python
delta = beta * (v - old_v)
state = old + k * delta[:, None]
```

两者数学等价；新形式减少循环内中间结果和逐元素运算。它体现了 KDA 的核心闭环：模型提出可解释
的内核变换，确定性验证器负责正确性和真实性能，只有达到门槛的候选才能进入最佳候选账本。

## 证据与边界

原始证据位于 [`evidence/agent-v100/prefill-0001-agent`](evidence/agent-v100/prefill-0001-agent)：

- `candidate.py`：最终候选源码；
- `api.json`：不含密钥的模型响应元数据；
- `0001-agent-smoke.json`：4组冒烟验证；
- `0001-agent-full.json`、`0001-agent-paired-1.json`、`0001-agent-paired-2.json`：三轮完整配对数据；
- `0001-agent-paired-review.json`：三轮聚合与晋级决定；
- `candidates.jsonl`：候选父子关系和不可变决策记录。

这是本项目在 V100 上对 KDA 思路的受控适配，使用官方数据和 reference，但不是 NVLabs 官方
Agent 工具链、官方评测器结果或 B200 成绩。

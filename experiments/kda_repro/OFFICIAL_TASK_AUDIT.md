# 五个官方 KDA 任务的 V100 可行性审计

审计基于 `source-lock.json` 固定的 KDA、竞赛提示、FlashInfer Bench、DeepGEMM 和 starter-kit
提交，以及数据集修订 `5e832ce88dd1013032b22a0e942979d7d2769cc3`。只读取任务定义、workload
元数据和 evaluator；未读取最终解答仓库。

| 任务 | workload | 官方输入范围 | V100 当前结论 |
|---|---:|---|---|
| GDN Decode | 54 | batch 1–64 | 已完整通过官方输入和 reference；有可重复性能证据 |
| GDN Prefill | 100 | 总长度 6–8192，序列数 1–57 | 自编顺序递推 Triton 候选与完整 runner 已就绪，等待 V100 恢复后跑完 |
| DSA TopK Indexer | 128 | batch 1–31，固定 11923 页 | FP8 E4M3 和 DeepGEMM 格式；V100 可做软件解码语义验证，不能代表原生 FP8 性能 |
| DSA Sparse Attention | 23 | token 1–8，固定 8462 页，top-k 2048 | BF16 存储、FP32 参考可适配；约 624 MB KV 输入可放入 32 GB V100 |
| FP8 MoE | 19 | token 1–14107，32 个本地 expert | 权重约 1.41 GB FP8，参考展开 FP32 约 5.64 GB；能做小规模语义检查，但官方 DeepGEMM 高性能路径不支持 V100 |

## 为什么先做 GDN

GDN Decode/Prefill 的核心是 128×128 FP32 state 递推，V100 虽无原生 BF16 算术，仍可保持
BF16 外部输入输出并用 FP32 完成内部计算。它不要求 SM90 Tensor Core 或原生 FP8，所以最适合
用现有卡验证“读取契约—生成候选—正确性淘汰—调度选择—保存证据”的 KDA 闭环。

Decode 已覆盖全部官方数据。Prefill 候选按时间维严格顺序更新，不把递推错误地并行化；ROWS
只切分 V 维。完整验证还会覆盖空序列、无初始 state 和默认 scale。

## DSA 与 MoE 的边界

DSA TopK 的 `float8_e4m3fn` 查询和页内 FP8+scale 打包可在 V100 上按位解码为 FP32，因此可以
检查索引语义。但该结果没有原生 FP8 指令意义。DSA Sparse Attention 可以在 FP32 中做 reference
和候选比较，适合作为 Prefill 之后的下一项 V100 语义实验。

MoE 固定几何为 H=7168、I=2048、256 个全局 expert、32 个本地 expert、top-k=8。官方高性能
实现依赖 DeepGEMM 和更新架构；V100 上运行 PyTorch FP32 展开只说明路由与数值语义，不能验证
竞赛性能。因此当前不会为了“看起来完成”而把 V100 的软件模拟写成官方复现成功。

## 后续执行顺序

1. V100 恢复后先运行 `run_official_v100.sh prefill`，完成全部 100 组 Prefill。
2. 再实现 DSA Sparse Attention 的 V100 融合候选；它不需要 FP8，是第二个可信性能实验。
3. DSA TopK 仅做按位 FP8 解码和索引一致性验证。
4. MoE 在 V100 上只跑路由和小规模语义检查；完整性能复现转到 B200/Hopper 及以上环境。

以上顺序最大化现有 V100 的有效产出，同时把硬件不支持与代码失败分开记录。

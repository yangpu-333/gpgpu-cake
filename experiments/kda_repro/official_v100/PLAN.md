# 官方任务的 V100 适配计划（实现前草案与执行契约）

来源：固定 KDA/MLSys 提示仓库和 Hugging Face `flashinfer-ai/mlsys26-contest` 官方定义及输入。
不读取最终提交仓库，不用其答案初始化候选。

首选 GDN Decode：保留 BF16 q/k/v/a/b、FP32 state、4→8 head 映射、可选 state、scale=0/None 默认值和输出格式。
参考使用下载定义内原始 `reference` 函数；候选一为批量 PyTorch 表达式，候选二为自行编写的一次 Triton 融合。
V100 无原生 BF16 算术，因此候选以整数位转换读写 BF16，中间用 FP32 运算；禁止把外部 BF16 契约改为 FP16。

验证先覆盖全部 54 个官方 Decode workload，再增加 state=None 和 scale 默认分支。
使用官方配置默认 atol=rtol=0.01，所有元素通过，另存最大误差；不宣称替代官方 FlashInfer evaluator。
检查输入未修改。候选未通过就拒绝，不为过关放宽误差。

计时用独立 CUDA Graph，先预热；每图多次调用，交替测量，保存原始值、峰值分配、环境与代码哈希。
对照批量 PyTorch 候选，避免只与带 Python 双循环的官方参考比较造成夸大。
候选父子关系与失败原因写入 JSON 报告。性能采集受平台计数器权限限制，沿用已保存拒绝日志。

其余四个官方任务读取定义、输入与依赖，逐项记录能运行的语义验证以及不能原样跑官方栈的证据。
先完成 Decode 的全部真实输入，再决定额外任务的具体实现；不安装不支持 sm70 的官方 cu132 栈覆盖已验证环境。

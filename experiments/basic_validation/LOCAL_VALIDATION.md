# 本机验证记录

日期：2026-09-09。环境：Windows 11、Python 3.12.14（Codex bundled Python）。

| 检查 | 结果 |
|---|---|
| Python 语法检查 | `run.py`、`triton_kernels.py`、`test_runner.py` 均通过；语法检查不验证 Triton 编译 |
| 标准库单元测试 | 7 项通过，覆盖分块尾部、数值检查、错误候选排除及旧报告保护 |
| CPU 实际运行 | 18 组配置通过；没有生成 GPU 性能排名 |
| 环境探测 | 当前 Python 缺少 PyTorch 和 Triton，无法建立 GPU 测试环境 |
| GPU 模式实际尝试 | 正确记录 `gpu_unavailable`，未运行 kernel，未回退为 CPU |

原始结果仅在执行本次验证的本机保存：`results/local-cpu-check.json`、`results/local-probe.json`、`results/local-gpu-attempt.json`。报告包含本次入口和 kernel 源码 SHA256。`results/` 不纳入 Git；远程 clone 后应按 README 运行，在目标机器生成自己的报告。

**尚未完成：** GPU JIT 编译、天数正确性与性能实测、内存/同步检查、Halide/Poly 接入、训练前后向验证。以上 GPU 脚本属于待目标环境验证的实验起点。

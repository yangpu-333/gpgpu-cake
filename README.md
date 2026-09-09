# GPGPU-CAKE

围绕 **自动化算子优化、Poly 模块优化、Halide IR 规范化**，准备在天数 GPU 上开展基础验证。CAKE 是方法参考；这里包含调研、实验计划和起步脚本。

目前已完成 CPU 自检，**尚未完成天数 GPU 实测，也未接入实际 Halide/Poly 模块**。现有分块候选用于检查实验流程，不能视为 Poly 优化成果。

## 远程机器快速开始

在已经配置 GitLab SSH 公钥的机器上执行：

```bash
git clone ssh://git@gitlab.ci-lab.net:5997/yangpu/gpgpu-cake.git
cd gpgpu-cake
python3 -m unittest discover -s experiments/basic_validation -p 'test_*.py' -v
python3 experiments/basic_validation/run.py --mode cpu
python3 experiments/basic_validation/run.py --mode probe
```

CPU 自检只需要 Python 3.9+。GPU 实验使用天数服务器已有的 Linux、COREX 及配套 PyTorch/Triton 环境。先激活厂商环境，再检查 `probe` 生成的报告；这里不提供可能覆盖厂商组件的通用安装命令。

确认环境后，先运行向量加法，再运行矩阵乘法：

```bash
python3 experiments/basic_validation/run.py --mode gpu --operator add --device 0
python3 experiments/basic_validation/run.py --mode gpu --operator matmul --device 0
```

结果自动保存到 `experiments/basic_validation/results/`，每次生成新 JSON。只有正确候选才参与计时和选优；具体容差、计时口径和退出状态见[实验代码说明](experiments/basic_validation/README.md)。

## 先读哪些内容

| 内容 | 入口 |
|---|---|
| 最直接、通俗的建议 | [三个关键词的实施建议](research/notes/围绕三个关键词的直接建议.md) |
| 接下来怎么做实验 | [基础验证实验计划](research/notes/基础验证实验计划.md) |
| CAKE 的核心依据及限制 | [论文精读](research/notes/CAKE论文精读与核心要点.md) |
| 项目目标与相关工作 | [初步调研](research/notes/项目与CAKE初步调研.md) |
| 已实际完成的验证 | [本机验证记录](experiments/basic_validation/LOCAL_VALIDATION.md) |
| CPU 实测结论与复现 | [CPU 基础验证结果](research/notes/CPU基础验证结果.md) |
| 参考资料来源 | [资料索引](research/README.md) |

## 按需下载论文和参考源码

运行基础实验不需要下载这些资料。需要阅读 CAKE 论文和第三方 `cake-ir` 源码时：

```bash
python3 scripts/fetch_references.py --group cake
python3 -m zipfile -e research/code/cake-ir.zip research/code/
```

还可用 `--group atrex`、`--group flashinfer` 或 `--group all`。脚本按照[下载清单](research/sources/download-manifest.json)核对大小和 SHA256，已有且校验一致的文件直接复用；发现不一致会报错，保留原文件。`--verify-only` 仅检查本地文件，不联网。历史链接内容若发生变化，校验会失败，需要先复核来源，不能直接跳过校验。

源码 ZIP 固定到具体提交，解压后保留上游许可证。`cake-ir` 是第三方实现；FlashInfer diff 只是阅读材料，不能单独编译。

## 文件整理约定

- Git 中保存实验源码、原创文档、下载工具及来源清单。
- `research/papers/`、`research/code/` 保存可重新下载的参考资料；`research/sources/` 中除清单外的提取文本和元数据仅在本地保留。
- `research/local/` 保存论文检查截图和临时验证日志；`experiments/basic_validation/results/` 保存本机运行结果。这些目录不上传 Git。
- 原始项目说明及厂商软件栈附件仍保留在本地 `自动化大模型训练算子编译优化/`，不随本次仓库发布。理解目标可先读仓库内的初步调研。

不要提交账号凭据或厂商安装包。后续可将适合分享的实验结论写成文档提交；原始运行记录留在执行实验的机器上。


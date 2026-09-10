# 项目与 CAKE 初步调研

总览：[阶段工作精要报告](notes/阶段工作精要报告.md)，汇总项目目标、论文与代码调研、CPU 实测、仓库整理及后续工作。

调研日期：2026-09-08。首先阅读 [整体报告](notes/项目与CAKE初步调研.md)。

2026-09-09 补充：[CAKE 论文精读与核心要点](notes/CAKE论文精读与核心要点.md)，逐项标注论文依据、工程推论与实验边界。

面向实施的短版：[围绕自动化、Poly、Halide IR 规范化的直接建议](notes/围绕三个关键词的直接建议.md)。

后续工作：[基础验证实验计划](notes/基础验证实验计划.md)及[配套实验代码](../experiments/basic_validation/README.md)。

已完成的 CPU 实测：[CPU 基础验证结果](notes/CPU基础验证结果.md)，含边界输入、错误拦截和浮点累加反例。

## 资料目录

Git 仓库保存本索引、`notes/` 中的原创 Markdown 文档及 `sources/download-manifest.json`。下表中的论文、源码及提取文本是本地参考缓存，clone 不会自动带上。按需执行下面的命令下载公开资料并核对 SHA256：

```bash
python3 scripts/fetch_references.py --group cake
python3 -m zipfile -e research/code/cake-ir.zip research/code/
```

命令从项目根目录运行。用 `--group all` 下载清单中的全部公开资料；ATREX ZIP 可同样解压到 `research/code/`。脚本不下载厂商软件栈附件，不重建本地提取文本、GitHub 元数据或检查截图。历史调研中指向这类材料的链接，在新机器上可能没有本地目标；论文和仓库的公开来源链接仍可使用。

| 位置 | 内容 |
|---|---|
| `papers/CAKE-2608.12629.pdf` | CAKE 论文 v1，21 页 |
| `papers/ATREX-2607.14541.pdf` | ATREX 论文 v1，22 页 |
| `code/cake-ir-b0fc98aeff065e464e958c475fa607bdd8e3c686/` | 第三方 cake-ir 完整源码快照，保留许可证 |
| `code/atrex-kernel-agent-d643fb5bc004ed021d1a60aa99bf9513204d3025/` | ATREX 完整源码快照，保留许可证 |
| `code/*.zip` | 对应源码原始压缩包 |
| `code/flashinfer-pr-4262.diff` | KDA prefill 代码、绑定、测试与 benchmark 变更 |
| `code/flashinfer-pr-4274.diff` | TinyGEMM2 代码、dispatch 与测试变更 |
| `code/flashinfer-pr-4638.diff` | 训练相关 SwiGLU + MXFP8 前向、反向变更 |
| `sources/*head.json` | 下载源码时的 GitHub 提交元数据 |
| `sources/flashinfer-*.json` | 官方 tracker、PR 状态及部分文件列表快照 |
| `sources/*.txt` | 本地 PDF 文本提取，含 PDF 页码分隔，便于检索 |
| `sources/download-manifest.json` | 下载来源、提交、文件大小和 SHA256 |
| `notes/local-validation.md` | 本机尝试验证的真实结果与限制 |
| `local/validation/` | 本地检查截图、第三方源码验证日志；不上传 Git |

源码通过固定提交的 HTTPS ZIP 下载；不是完整 Git 历史。FlashInfer 保存的是 PR 差异，不是可独立编译的完整仓库。论文下载地址、PR 和仓库链接见报告与 manifest。没有修改原始项目说明和原有附件，没有启动 GPU 调优或付费模型调用。

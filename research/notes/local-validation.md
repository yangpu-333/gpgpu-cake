# 本机验证记录

日期：2026-09-08；主机环境：Windows / PowerShell。

- CAKE 与 ATREX PDF 均能经 pypdf 解析，分别 21、22 页；软件栈 PDF 1277 页。抽查渲染了 CAKE p2 与软件栈 p487，图文可读。
- cake-ir 完整源码 ZIP 已解压；版本固定为 `b0fc98aeff065e464e958c475fa607bdd8e3c686`。
- 尝试 `python -m pytest <snapshot>/tests -q -p no:cacheprovider`：当前 bundled Python 没有 pytest，未执行测试。错误保存在本地 `research/local/validation/cake-ir-pytest.txt`，不纳入 Git。
- 设置源码 `PYTHONPATH` 后尝试 `python <snapshot>/scripts/corpus_gate.py`：导入 `cake_ir/cache.py` 第 7 行的 `fcntl` 失败，Windows Python 不提供该 Unix 模块。没有使用假模块绕过验证。
- WSL 发行版枚举返回 `E_ACCESSDENIED`，未取得可用 Linux 环境；不据此推断机器未安装 WSL。
- 因而没有宣称 corpus gate、GPU 编译、正确性、性能或论文复现通过。没有修改第三方代码。

## 后续 Linux 环境验证建议

在源码目录和独立 Python 环境中安装仓库 dev 依赖后，先执行仓库的 pytest 与 corpus gate，再按 README 在匹配 NVIDIA target 上执行 GPU smoke。此步骤只能验证第三方实现；天数需要先实现对应 target/toolchain/runtime，再独立验证。

FlashInfer PR diff 是阅读资料，编译需要对应完整仓库及依赖，不能单独执行 diff。搜索预算和 GPU 服务器尚未配置，不应直接启动长时间 agent campaign。

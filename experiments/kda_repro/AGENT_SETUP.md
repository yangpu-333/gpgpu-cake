# GDN Decode 自动优化 Agent 接入

不需要安装 Claude Code、Codex CLI 或 LangChain。本项目自带受控 Agent 控制器，并只使用 Python
标准库调用 OpenAI Chat Completions 兼容 API。GPU 环境继续使用现有 PyTorch 2.5.1、Triton 3.1.0
和 safetensors 0.5.3。

## 远程机器上唯一需要配置的内容

API 密钥保存在项目目录之外：

```bash
install -d -m 700 "$HOME/.config/gpgpu-cake"
nano "$HOME/.config/gpgpu-cake/agent.env"
chmod 600 "$HOME/.config/gpgpu-cake/agent.env"
```

文件内容：

```bash
export KDA_LLM_BASE_URL="https://服务地址/v1"
export KDA_LLM_MODEL="模型名称"
export KDA_LLM_API_KEY="密钥"
export SSL_CERT_FILE="$HOME/.local/share/ca-certificates/scholar-git-ca-bundle.pem"
```

候选请求默认最多使用 4000 个输出 token。模型只返回少量精确代码替换，控制器将其应用到
已审核源码，并重新检查完整源码。若兼容服务返回空内容，控制器会自动关闭 JSON mode
重试一次，并在 `api/` 中保存两次请求的结束原因和 token 用量。
可通过 `KDA_LLM_CANDIDATE_MAX_TOKENS` 调整上限。

如果服务提供的是完整 Chat Completions 地址，可改用
`KDA_LLM_ENDPOINT=https://.../chat/completions`。若服务不支持 JSON mode，再加
`export KDA_LLM_JSON_MODE=0`。密钥文件不得提交到 Git，也不要把密钥写到命令行参数或日志中。
使用 Azure 风格的 `api-key` 请求头时，增加：

```bash
export KDA_LLM_AUTH_HEADER="api-key"
export KDA_LLM_AUTH_SCHEME=""
```

## 启动

先做配置检查；该命令不会发出 API 请求：

```bash
cd "$HOME/gpgpu-cake"
git pull --ff-only origin main
source "$HOME/.config/gpgpu-cake/agent.env"
"$HOME/kda-repro/.venv/bin/python" experiments/kda_repro/official_v100/agent_controller.py \
  --data "$HOME/kda-repro/data/official" \
  --workspace "$HOME/kda-repro/agent-gdn-decode" \
  --iterations 1 --check-config
```

确认配置后可做一次最小 API 请求（不运行 GPU）：

```bash
"$HOME/kda-repro/.venv/bin/python" experiments/kda_repro/official_v100/agent_controller.py \
  --data "$HOME/kda-repro/data/official" \
  --workspace "$HOME/kda-repro/agent-gdn-decode" \
  --iterations 1 --check-api
```

正式运行十轮：

```bash
LOG="$HOME/kda-repro/agent-gdn-$(date -u +%Y%m%dT%H%M%SZ).log"
nohup bash experiments/kda_repro/scripts/run_gdn_decode_agent.sh 10 \
  > "$LOG" 2>&1 < /dev/null &
echo "PID: $!"
echo "日志: $LOG"
```

第一次运行先完整测量种子候选，再让 Agent 写 `docs/draft.md`，然后开始十轮优化。每轮依次通过静态
限制、4组冒烟验证和54组完整验证。完整验证在同一进程中对当前最佳与新候选各取21次交替样本，
新候选相对同轮基线至少改善1%才晋级。中断后用同一命令可继续，已有候选与证据不会覆盖。

## 保存位置

默认工作区是 `$HOME/kda-repro/agent-gdn-decode`：

- `candidates/`：种子及每轮不可变候选源码；
- `runs/`：冒烟与完整原始报告，以及子进程输出；
- `candidates.jsonl`：父子关系、正确性、速度和晋级/淘汰原因；
- `best.json`：当前最佳候选；
- `api/`：不含密钥的 API 返回内容和 token 用量；
- `docs/draft.md`：Agent 的初始优化草案。

该静态限制是工程护栏，不是强安全沙箱。若 API 属于不受信任的第三方，应在单独容器或独立普通
用户下运行 Agent 工作区。验证器、官方 reference 和数据均不由模型修改。

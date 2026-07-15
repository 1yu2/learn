# 环境配置

本仓库默认按 AgentScope 2.x 和 DashScope 学习。配置模板不包含任何真实密钥，模型调用之外的测试不需要网络或 API key。

## 初始化

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env
```

在 `.env` 中填写 `DASHSCOPE_API_KEY`。不要把 `.env`、API key 或运行时生成的 `.workspace/` 提交到 Git。

配置会从当前目录开始向父目录查找最近的 `.env`，因此在仓库根目录或
`examples/` 子目录启动都可以复用根目录配置。推荐使用 `uv run`，它会自动使用
项目环境：

```bash
# 在仓库根目录
uv run python examples/01_quickstart/hello_agent.py

# 在 examples/ 目录
uv run 01_quickstart/hello_agent.py
```

直接使用系统 `python` 时，需要先安装本仓库的 editable package；否则 `src` 布局下
系统解释器不会自动发现 `agentscope_learn`：

```bash
uv pip install -e ".[dev]"
python examples/01_quickstart/hello_agent.py
```

## 配置字段

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `MODEL_PROVIDER` | `dashscope` | `dashscope`、`openai` 或 `ollama` |
| `DASHSCOPE_API_KEY` | 空 | DashScope 凭据 |
| `OPENAI_API_KEY` | 空 | OpenAI-compatible provider 凭据 |
| `MODEL_NAME` | `qwen-plus` | 模型名称 |
| `MODEL_BASE_URL` | 空 | OpenAI-compatible 或 Ollama 地址 |
| `AGENTSCOPE_LOG_LEVEL` | `INFO` | `DEBUG`、`INFO`、`WARNING` 或 `ERROR` |
| `AGENTSCOPE_WORKSPACE` | `./.workspace` | Agent 工作区和临时状态目录 |
| `RAG_DATA_DIR` | `./data/knowledge` | 本地知识库输入目录 |
| `RAG_COLLECTION` | `agentscope_learning` | RAG 集合名称 |
| `ENABLE_TRACING` | `false` | 是否启用追踪开关 |

程序代码使用 `Settings` 读取配置，系统环境变量优先于 `.env`：

```python
from agentscope_learn import Settings

settings = Settings.from_env()
print(settings.redacted_summary())
settings.require_model_credentials()
```

`redacted_summary()` 只显示 `***configured***`，不会打印 key 原文。`require_model_credentials()` 应在真正发起模型调用前执行。

## 切换 provider

### OpenAI-compatible

```dotenv
MODEL_PROVIDER=openai
OPENAI_API_KEY=replace-with-a-local-secret
MODEL_NAME=your-model-name
MODEL_BASE_URL=https://your-compatible-endpoint/v1
```

### Ollama

```dotenv
MODEL_PROVIDER=ollama
MODEL_NAME=your-local-model
MODEL_BASE_URL=http://localhost:11434/v1
```

Ollama 不需要云端 API key，但本地服务必须已经启动。

## 离线验证

不配置 key 也可以运行以下检查：

```bash
python3 -m pytest -q
python3 -m compileall src examples tests
git diff --check
```

涉及真实模型的示例才需要填写 key。若出现 `DASHSCOPE_API_KEY is required`，说明配置读取正常，只是当前运行没有提供凭据。

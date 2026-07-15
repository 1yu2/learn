# 01 First Agent

本目录从最小 Agno Agent 开始，逐步加入 DeepSeek、MCP、金融工具、表格输出和 AgentOS 服务。

## 学习目标

- 理解 `Agent`、`model`、`description`、`instructions` 和 `markdown`。
- 使用 `print_response(stream=True)` 观察流式模型响应。
- 通过 Tool 和 MCP 让 Agent 获取外部信息。
- 理解普通文本输出、结构化输出提示和 Pydantic 校验之间的区别。
- 将 Agent 包装成 AgentOS 服务，并连接 Agno Control Plane。

## 环境准备

在仓库根目录安装基础依赖：

```bash
uv sync
```

运行 AgentOS 示例还需要安装可选依赖：

```bash
uv sync --extra agentos
```

项目要求 Python 3.12 或更高版本。

## DeepSeek 配置

复制环境变量模板：

```bash
cp .env.example .env
```

在 `.env` 中填写 DeepSeek 配置：

```dotenv
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_MODEL_ID=deepseek-v4-flash
```

`DEEPSEEK_MODEL_ID` 可省略，示例默认使用 `deepseek-v4-flash`。

当前脚本为了兼容旧的本地配置，也会回退读取 `OPENAI_API_KEY`。这个变量中必须保存 DeepSeek 密钥；真正的 OpenAI 密钥不能用于 DeepSeek，并且不应发送到 DeepSeek 接口。新配置应始终使用 `DEEPSEEK_API_KEY`。

运行示例会访问外部模型或数据服务，可能产生 API 费用。

## 示例一览

| 文件 | 内容 | 外部服务 |
| --- | --- | --- |
| [`01_agent.py`](01_agent.py) | 最小 DeepSeek Agent，流式输出自我介绍 | DeepSeek API |
| [`03_mcp.py`](03_mcp.py) | 通过 Streamable HTTP 查询 Agno 官方文档 MCP | DeepSeek API、Agno Docs MCP |
| [`04_finacial.py`](04_finacial.py) | 使用 Yahoo Finance 工具分析 NVDA | DeepSeek API、Yahoo Finance |
| [`05_weather.py`](05_weather.py) | 生成纽约各月天气 Markdown 表格 | DeepSeek API |
| [`05_agentos.py`](05_agentos.py) | 将网页搜索 Agent 作为 AgentOS 服务运行 | DeepSeek API、网页搜索、AgentOS |

目录编号目前不连续，并且包含两个 `05_*` 文件。请按实际文件名执行命令。`04_finacial.py` 的 `finacial` 是当前文件名中的拼写，运行时不要改成其他名称。

## 01_agent.py

最小示例展示：

- 使用 Agno 原生 `DeepSeek` 模型。
- 使用 `description` 描述 Agent。
- 开启 Markdown 输出。
- 使用 `stream=True` 显示流式响应。

运行：

```bash
uv run python examples/01_first_agent/01_agent.py
```

该文件在模块顶层调用模型，因此导入文件也会立即发起请求。

## 03_mcp.py

该示例通过 `MCPTools` 连接：

```text
https://docs.agno.com/mcp
```

代码使用 `async with MCPTools(...)` 管理 MCP 连接，并通过 `await agent.aprint_response(...)` 消费异步流式响应。Agent 会先调用 `search_agno` 或 `query_docs_filesystem_agno` 查询官方文档，再生成回答。

运行：

```bash
uv run python examples/01_first_agent/03_mcp.py
```

运行时必须同时能够访问 DeepSeek API 和 Agno 文档 MCP 服务。

## 04_finacial.py

该示例使用 `YFinanceTools` 查询 NVIDIA（`NVDA`）并生成金融分析。

运行：

```bash
uv run python examples/01_first_agent/04_finacial.py
```

当前代码使用默认的 `YFinanceTools()`。在当前 Agno 版本中，默认只启用当前股价工具 `get_current_stock_price`，并没有启用历史价格、基本面、公司新闻等完整金融函数。若需要完整分析，应在代码中使用 `YFinanceTools(all=True)` 或显式启用所需函数。

该文件在模块顶层运行，导入文件也会立即请求 DeepSeek 和 Yahoo Finance。

## 05_weather.py

该示例要求模型生成纽约每个月的季节和平均温度表格。

运行：

```bash
uv run python examples/01_first_agent/05_weather.py
```

注意：

- 示例没有接入实时天气 API，内容来自模型知识，不代表实时气象数据。
- 文件定义了 `WeatherData`，但目前没有将它传给 Agent 的 `output_schema`。
- `expected_output` 只是提示模型输出格式，结果不会经过 Pydantic 结构校验。
- 该文件在模块顶层运行，导入文件也会立即调用模型。

## 05_agentos.py

该示例复用了 DeepSeek 网页搜索 Agent，并通过 `AgentOS` 暴露为本地服务。

安装 AgentOS 依赖：

```bash
uv sync --extra agentos
```

启动：

```bash
uv run --extra agentos python examples/01_first_agent/05_agentos.py
```

启动后保持终端运行，可访问：

| 地址 | 用途 |
| --- | --- |
| <http://localhost:7777> | AgentOS API |
| <http://localhost:7777/docs> | Swagger API 文档 |
| <http://localhost:7777/config> | AgentOS 配置 |
| <https://os.agno.com> | AgentOS Control Plane |

在 Control Plane 中点击 **Connect OS**，连接地址填写：

```text
http://localhost:7777
```

连接后选择 `web-search-agent` 即可对话和查看工具调用。浏览器如果询问是否允许访问本地网络，需要允许该权限。

按 `Ctrl+C` 停止 AgentOS。端口 `7777` 必须空闲。当前示例没有配置持久化数据库，重启服务后不要依赖历史会话持久化。

## 常见问题

### 提示缺少 API Key

确认项目根目录存在 `.env`，并包含：

```dotenv
DEEPSEEK_API_KEY=your-deepseek-api-key
```

### 日志显示 OpenAI API

Agno 的 DeepSeek 适配器复用了 OpenAI-compatible 客户端，因此部分底层错误日志写作 `OpenAI API`。使用 `DeepSeek` 模型时，请求地址仍是 DeepSeek 官方接口。

### 出现 Request timed out

这通常是 DeepSeek 连接或网络抖动。AgentOS 网页搜索示例已经将 TLS 连接超时提高到 15 秒，并保留 SDK 重试。其他示例仍使用 SDK 默认连接配置。

### 网页搜索显示 No results found

搜索服务可能因查询过窄、限流或上游反爬而没有结果。`05_agentos.py` 使用自动搜索后端，并将 `DDGSException` 转换为可重试提示。

### MCP 连接失败

确认可以访问 `https://docs.agno.com/mcp`，并检查本地是否安装了 `mcp` 依赖。MCP 示例使用异步连接，不能改回同步 `print_response`。

### AgentOS 无法启动

检查：

```bash
lsof -nP -iTCP:7777 -sTCP:LISTEN
```

如果端口已被占用，先停止已有进程，或修改 `05_agentos.py` 中的端口。

## 当前代码说明

`04_finacial.py` 和 `05_weather.py` 仍包含部分未使用导入。这些导入不影响示例主要逻辑，但会增加运行依赖，并会被 Ruff 的 `F401` 规则报告。README 记录当前代码行为，不在这里隐含假设这些脚本已经完成清理。

## 参考

- [Build Your First Agent](https://docs.agno.com/first-agent)
- [MCP Tools](https://docs.agno.com/tools/mcp/overview)
- [AgentOS Control Plane](https://docs.agno.com/agent-os/control-plane)

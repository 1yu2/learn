# 02 Tools

本目录通过 6 个递进示例演示 Agno 的工具调用：从单个 Python 函数、多个函数协作，逐步扩展到内置 Toolkit、文件操作和 AgentOS 控制面板。

## 学习目标

- 理解 Agent 如何选择工具、生成参数并根据工具结果回答。
- 使用类型注解和 docstring 定义清晰的工具 schema。
- 注册单个或多个自定义工具，并观察连续工具调用。
- 使用 `DuckDuckGoTools`、`YFinanceTools` 等内置 Toolkit。
- 将带工具的 Agent 发布为 AgentOS 服务，并连接 Agno 控制面板。

## 工具调用流程

```text
用户输入问题
    ↓
Agent 分析问题，判断是否需要工具
    ↓
从已注册工具中选择工具，并生成调用参数
    ↓
Agno 校验参数并执行 Python 函数或 Toolkit
    ↓
工具返回结果
    ↓
Agent 判断是否需要继续调用其他工具
    ↓
Agent 基于工具结果生成最终回答
```

工具是否被调用由模型决定。函数名、参数类型、docstring 和 Agent 的 `instructions` 越明确，模型越容易正确选择工具。

## 运行准备

在仓库根目录执行：

```bash
uv sync
```

项目要求 Python 3.12 或更高版本。然后在根目录创建或修改 `.env`：

```dotenv
DEEPSEEK_API_KEY=你的_DeepSeek_API_Key
# 可选；不设置时示例默认使用 deepseek-v4-flash
DEEPSEEK_MODEL_ID=deepseek-v4-flash
```

这些示例实际创建的是 `agno.models.deepseek.DeepSeek`。代码为兼容已有环境会回退读取 `OPENAI_API_KEY`，但 OpenAI 的真实 API Key 不能用于调用 DeepSeek，推荐始终配置 `DEEPSEEK_API_KEY`。

> Agno 的 DeepSeek 适配器复用了 OpenAI-compatible 客户端，因此异常日志中可能出现 `OpenAI API`。这不代表代码切换到了 OpenAI 模型；模型类型、API 地址和 Key 仍由 `DeepSeek(...)` 配置决定。

## 示例索引

| 文件 | 主要内容 | 注册的工具 | 外部服务 |
| --- | --- | --- | --- |
| `01_tool_agent.py` | 搜索最新网页信息 | `DuckDuckGoTools` | DeepSeek、DDGS 搜索后端 |
| `02_learn_tool.py` | 第一个自定义工具 | `get_current_time` | DeepSeek |
| `03_tool_example.py` | 带参数的数值工具 | `calculate_square` | DeepSeek |
| `04_more_tool.py` | 多工具连续调用 | `add`、`multiply`、`calculate_square` | DeepSeek |
| `05_file_agent.py` | 文件读取、写入、枚举和元数据查询 | `read_file`、`write_file`、`list_files`、`get_file_info` | DeepSeek、本地文件系统 |
| `06_internel_tool_agentos.py` | 将网页搜索和股票查询 Agent 发布到 AgentOS | `DuckDuckGoTools`、`YFinanceTools` | DeepSeek、DDGS 搜索后端、Yahoo Finance；可选 Agno Control Plane |

## 01：内置网页搜索工具

`01_tool_agent.py` 使用 `DuckDuckGoTools` 搜索 2026 年人工智能领域的最新信息，并要求回答附带来源链接。该 Toolkit 注册的实际工具名是 `web_search`，使用自动搜索后端且固定最多返回 5 条结果。当搜索抛出 `DDGSException` 时，tool hook 会返回“缩短关键词后再次调用”的提示，但是否再次调用仍由模型决定。

```bash
uv run python examples/02_tools/01_tool_agent.py
```

该示例依赖网络搜索。工具返回的是搜索结果和摘要，不会读取每个网页的完整正文；模型总结和来源内容仍需人工核对，不能只根据生成文本判断事实是否可靠。

## 02：无参数自定义工具

`02_learn_tool.py` 把普通 Python 函数 `get_current_time()` 注册为工具。函数返回本机当前时间，不包含时区信息。这个问题适合调用工具，但代码没有强制工具调用，最终决策仍由模型完成。

```bash
uv run python examples/02_tools/02_learn_tool.py
```

## 03：带参数自定义工具

`03_tool_example.py` 演示模型如何根据函数签名生成 `number` 参数，再调用 `calculate_square(number)`。示例问题是“15 的平方是多少？”，工具结果为浮点数 `225.0`。模型也能直接计算这个简单问题，因此每次运行不一定都会调用工具。

```bash
uv run python examples/02_tools/03_tool_example.py
```

## 04：多个工具协作

`04_more_tool.py` 同时注册加法、乘法和平方工具。示例提示中的 `3—6` 使用长横线表示减法，但没有注册减法工具；模型可以将它转换为 `add(3, -6)`，再进行乘法和平方，最终结果应为 `225`。

```bash
uv run python examples/02_tools/04_more_tool.py
```

这个示例用于观察模型的工具规划能力；调用顺序由模型生成，不保证每次运行的中间步骤完全一致。

## 05：文件操作 Agent

`05_file_agent.py` 使用 `pathlib` 实现 4 个本地工具：

- `read_file`：以 UTF-8 读取文件。
- `write_file`：以 UTF-8 写入文件。
- `list_files`：按名称排序并列出指定目录中的文件，不包含子目录。
- `get_file_info`：返回文件大小和修改时间。

```bash
uv run python examples/02_tools/05_file_agent.py
```

示例提示词会列出当前工作目录的文件，并尝试选择第三个文件查询和读取。由于命令从仓库根目录运行，“当前目录”就是仓库根目录。如果目录中不足 3 个普通文件，模型无法完成全部步骤；提示中的“关闭文件”也没有对应工具，因为 `Path.read_text()` 读取完成后已经自动关闭文件。

> 这些工具没有把路径限制在某个沙箱目录内，可以读写当前进程权限允许访问的任意路径。它们适合可信的本地学习环境，不应未经路径校验就暴露给不可信用户。写入不存在的父目录时也会返回错误，不会自动创建目录。

## 06：AgentOS 和控制面板

`06_internel_tool_agentos.py` 将一个多功能 Agent 发布为 AgentOS 服务。它组合了网页搜索和股票查询工具，并监听本机 `7777` 端口。

先安装 AgentOS 可选依赖：

```bash
uv sync --extra agentos
```

启动服务：

```bash
uv run --extra agentos python examples/02_tools/06_internel_tool_agentos.py
```

启动后保持终端运行，可以访问：

- AgentOS 服务：<http://localhost:7777>
- OpenAPI 文档：<http://localhost:7777/docs>
- AgentOS 配置：<http://localhost:7777/config>
- Agno 控制面板：<https://os.agno.com>

在控制面板中添加 `http://localhost:7777` 作为 AgentOS 地址，然后选择 `multi-tool` Agent 创建会话。`localhost:7777` 提供的是 AgentOS API，不是独立的聊天网页；聊天 UI 位于 Agno 控制面板。按 `Ctrl+C` 可停止本地服务。

服务默认绑定 `localhost`，只能从当前电脑访问。示例未配置服务认证；如需改为局域网或公网地址，必须先补充认证、访问控制和网络安全配置。

当前代码使用 `YFinanceTools()` 的默认配置。在当前 Agno 版本中，它只注册实时股价查询；如需历史行情、公司基本面或新闻，需要在代码中显式启用对应函数。

文件名中的 `internel` 是现有文件名，运行命令必须保持这个拼写。

## 如何定义一个合适的工具

```python
def calculate_square(number: float) -> float:
    """计算数字的平方。

    Args:
        number: 要计算平方的数字。

    Returns:
        计算结果。
    """
    return number**2
```

建议遵循以下规则：

1. 使用明确的函数名，避免 `run`、`do_it` 等含义模糊的名称。
2. 为每个参数添加准确的类型注解。
3. 在 docstring 中说明用途、参数含义和返回值。
4. 返回字符串、数字、字典等容易序列化的结果。
5. 对可预期的 I/O 或网络异常返回清晰信息，让 Agent 能决定是否重试。

## 常见问题

### `Agent.__init__()` 不支持 `show_tool_calls`

当前项目使用的 Agno 版本不接受 `show_tool_calls` 构造参数。本目录示例已经移除或注释该参数，不要重新启用。使用 `print_response(...)` 时，终端会按当前 Agno 的输出格式展示工具调用过程。

### 请求超时或出现 `OpenAI API: Request timed out`

先检查 `DEEPSEEK_API_KEY`、`DEEPSEEK_MODEL_ID` 和到 DeepSeek API 的网络连接。涉及搜索或股票的示例还依赖第三方服务，任意一段网络请求超时都可能使本次运行失败。代码已设置较长读取超时和 2 次重试，但无法消除服务端限流或网络不可达问题。

### DuckDuckGo 没有搜索结果

缩短关键词、减少限定条件后重试。连续请求过快也可能触发 DuckDuckGo 的限制；稍后再运行通常比反复立即重试更有效。

### `7777` 端口被占用

先确认占用进程：

```bash
lsof -i :7777
```

停止已有服务，或同步修改 `agent_os.serve(..., port=7777)` 和控制面板连接地址。

### 导入脚本时意外发起模型请求

所有示例都会在模块加载阶段读取并校验 API Key。除此之外，`01_tool_agent.py` 到 `04_more_tool.py` 的示例调用位于模块顶层，因此导入文件也会立即请求模型。`05_file_agent.py` 和 `06_internel_tool_agentos.py` 只保护了实际对话和服务启动，导入时仍要求环境中存在 API Key。

## 参考资料

- [Agno：What are Tools?](https://docs.agno.com/tools/overview.md)
- [Agno：AgentOS](https://docs.agno.com/agent-os/introduction)
- [DeepSeek API 文档](https://api-docs.deepseek.com/)

# AgentScope 学习计划

本仓库用于按官方文档逐步学习 AgentScope。学习基线为 AgentScope 2.x，当前官方文档入口会指向 `2.0.3 English`。AgentScope 2.0 相比 1.0 是一次 breaking change，因此本仓库默认只跟随 2.x 文档和代码示例。

## 官方入口

- 官方文档: https://docs.agentscope.io/
- 2.0.3 文档首页: https://docs.agentscope.io/versions/2.0.3/en
- Quickstart: https://docs.agentscope.io/versions/2.0.3/en/quickstart
- GitHub: https://github.com/agentscope-ai/agentscope
- PyPI: https://pypi.org/project/agentscope/
- 文档索引: https://docs.agentscope.io/llms.txt

## 学习目标

完成本计划后，应能独立完成以下任务：

- 安装并运行一个最小 AgentScope Agent。
- 理解 `Message`、`Event`、`Agent`、`Model`、`Toolkit`、`Context` 的职责边界。
- 给 Agent 接入内置工具、自定义工具、权限控制和流式事件处理。
- 使用 Plan、Middleware、RAG、Long-Term Memory 等能力构建更完整的 Agent 应用。
- 了解 Agent as Service、Agent Team、RAG Service 的部署方向。
- 在本仓库沉淀可运行示例、学习笔记和阶段复盘。

## 建议仓库结构

本仓库按“学习阶段 + 可运行示例 + 笔记 + 最终项目”的方式组织：

```text
.
├── README.md
├── pyproject.toml
├── examples/
│   ├── 01_quickstart/
│   ├── 02_message_event/
│   ├── 03_agent_tool/
│   ├── 04_permission_plan/
│   ├── 05_middleware_context/
│   ├── 06_rag_memory/
│   └── 07_service_team/
├── notes/
│   ├── concepts.md
│   ├── api-differences.md
│   └── troubleshooting.md
├── projects/
│   └── personal_research_agent/
├── src/
│   └── agentscope_learn/
│       └── learning_plan.py
└── tests/
    ├── test_custom_tool.py
    └── test_learning_plan.py
```

## 环境准备

AgentScope 2.x 要求 Python 3.11+。官方推荐使用 `uv` 安装。

```bash
uv venv
source .venv/bin/activate
uv pip install agentscope
```

如果要安装本仓库的学习包和测试依赖：

```bash
uv pip install -e ".[dev]"
```

如果需要更多模型、工具和 RAG 相关依赖，可安装额外依赖：

```bash
uv pip install agentscope\[full\]
uv pip install agentscope\[rag\]
```

本仓库当前的离线验证命令：

```bash
python3 -m pytest
python3 -m compileall src examples tests
git diff --check
```

安装后验证：

```python
import agentscope

print(agentscope.__version__)
```

## 阶段 1: 快速运行第一个 Agent

官方文档：

- Quickstart: https://docs.agentscope.io/versions/2.0.3/en/quickstart
- AgentScope 2.0 介绍: https://docs.agentscope.io/versions/2.0.3/en

学习重点：

- AgentScope 2.0 的定位：安全、效率、灵活性、完整工程链路。
- `Agent` 的最小构造：`name`、`system_prompt`、`model`、`toolkit`。
- `reply` 和 `reply_stream` 的区别。
- 使用 `DASHSCOPE_API_KEY` 或替换为其他模型提供商。

练习任务：

- 创建 `examples/01_quickstart/hello_agent.py`。
- 运行一个最小 Agent，让它回答一句自我介绍。
- 分别尝试 `reply` 和 `reply_stream`，观察最终消息和流式事件。

阶段产出：

- 一份可运行的 quickstart 脚本。
- 在 `notes/concepts.md` 记录 `Agent`、`Model`、`Toolkit` 三者关系。

## 阶段 2: Message 与 Event

官方文档：

- Message & Event: https://docs.agentscope.io/versions/2.0.3/en/building-blocks/message-and-event

学习重点：

- `Msg` 是完整对话轮次，用于 Agent 间通信、上下文存储和最终持久化。
- `Event` 是流式执行过程中的增量信号，用于前端渲染、工具调用进度、人类确认等。
- 一次 `reply_stream` 会产生一串事件，这些事件最终可以重建一个完整的 Assistant Message。
- 内容块包括 `TextBlock`、`DataBlock`、`ThinkingBlock`、`ToolCallBlock`、`ToolResultBlock`、`HintBlock`。

练习任务：

- 创建 `examples/02_message_event/inspect_events.py`。
- 打印每个 event 的类型、`reply_id`、`block_id` 或 `tool_call_id`。
- 用 `append_event()` 将流式事件重建为最终消息。

阶段产出：

- 一份事件追踪脚本。
- 在 `notes/concepts.md` 画出 `reply_stream -> events -> Msg` 的数据流。

## 阶段 3: Model 层与结构化输出

官方文档：

- Model: https://docs.agentscope.io/versions/2.0.3/en/building-blocks/model

学习重点：

- 模型层由 `Credential` 和模型族组成，包括 Chat Model、TTS、Embedding、Realtime Model。
- Chat Model 支持多家提供商，例如 OpenAI、DashScope、DeepSeek、Gemini、Anthropic、Ollama 等。
- 模型调用可返回一次性 `ChatResponse`，也可返回流式 `AsyncGenerator`。
- `generate_structured_output` 可基于 Pydantic schema 生成结构化输出。
- Formatter 负责把 AgentScope 的 `Msg` 转换为不同模型 API 需要的 payload。

练习任务：

- 创建 `examples/03_agent_tool/model_structured_output.py`。
- 定义一个 Pydantic schema，例如 `BookSummary` 或 `WeatherInfo`。
- 调用 `generate_structured_output`，检查返回数据是否符合 schema。

阶段产出：

- 一个结构化输出示例。
- 在 `notes/api-differences.md` 记录不同模型提供商的 credential 和 model class。

## 阶段 4: Agent、Tool 与 Toolkit

官方文档：

- Agent: https://docs.agentscope.io/versions/2.0.3/en/building-blocks/agent
- Tool: https://docs.agentscope.io/versions/2.0.3/en/building-blocks/tool

学习重点：

- `Agent` 是 AgentScope 的核心抽象，负责 ReAct 推理-行动循环。
- `Toolkit` 管理工具、MCP、技能和工具组，并把 JSON Schema 暴露给模型。
- 内置工具包括 `Bash`、`Read`、`Write`、`Edit`、`Glob`、`Grep` 和 Plan tools。
- 可通过 `ToolBase` 编写强控制的自定义工具，也可用 `FunctionTool` 包装普通 Python 函数。
- 工具可以声明只读、并发安全、外部执行、状态注入等属性。

练习任务：

- 创建 `examples/03_agent_tool/custom_tool.py`。
- 用 `FunctionTool` 包装一个只读函数，例如本地时间、简单搜索或计算器。
- 再用 `ToolBase` 写一个带权限判断的自定义工具。
- 让 Agent 在一次任务中调用工具并返回结果。

阶段产出：

- 一个 `FunctionTool` 示例。
- 一个 `ToolBase` 示例。
- 在 `notes/concepts.md` 总结 `Agent -> Toolkit -> Tool -> ToolChunk` 的调用链路。

## 阶段 5: 权限系统与人类确认

官方文档：

- Permission System: https://docs.agentscope.io/versions/2.0.3/en/building-blocks/permission-system
- Agent Human-in-the-Loop: https://docs.agentscope.io/versions/2.0.3/en/building-blocks/agent

学习重点：

- AgentScope 的权限系统会拦截每次工具调用，并产生 `ALLOW`、`DENY` 或 `ASK`。
- 权限决策由规则、模式和工具内置检查共同决定。
- 常见模式包括 `DEFAULT`、`EXPLORE`、`ACCEPT_EDITS`、`BYPASS`、`DONT_ASK`。
- 当需要用户确认时，Agent 会暂停并发出 `RequireUserConfirmEvent`。
- 外部执行工具会通过 `RequireExternalExecutionEvent` 暂停，等待外部系统回填结果。

练习任务：

- 创建 `examples/04_permission_plan/permission_modes.py`。
- 对比 `DEFAULT`、`EXPLORE`、`ACCEPT_EDITS` 下文件读写和 Bash 调用的行为。
- 捕获 `RequireUserConfirmEvent`，手动构造 `UserConfirmResultEvent` 继续执行。

阶段产出：

- 一个权限模式对比脚本。
- 在 `notes/troubleshooting.md` 记录常见权限暂停和恢复方式。

## 阶段 6: Plan 工具与复杂任务拆解

官方文档：

- Plan: https://docs.agentscope.io/versions/2.0.3/en/building-blocks/plan

学习重点：

- Plan tools 用结构化任务列表替代纯文本推理中的临时计划。
- 内置工具包括 `TaskCreate`、`TaskGet`、`TaskList`、`TaskUpdate`。
- 任务状态通常按 `pending -> in_progress -> completed` 流转。
- 任务可通过 `blocks` 和 `blocked_by` 表达依赖关系。
- 任务状态存储在 `agent.state.tasks_context`，可随 Agent state 持久化。

练习任务：

- 创建 `examples/04_permission_plan/planning_agent.py`。
- 给 Agent 一个三步以上的复杂任务，例如“阅读 README、总结学习路线、生成下一步 todo”。
- 观察 Agent 是否调用 Plan tools 创建、查看和更新任务。

阶段产出：

- 一个带计划能力的 Agent。
- 在 `notes/concepts.md` 总结何时需要 Plan tools，何时不需要。

## 阶段 7: Context、Workspace 与 Middleware

官方文档：

- Context: https://docs.agentscope.io/versions/2.0.3/en/building-blocks/context
- Workspace: https://docs.agentscope.io/versions/2.0.3/en/building-blocks/workspace
- Middleware: https://docs.agentscope.io/versions/2.0.3/en/building-blocks/middleware

学习重点：

- Context 是 Agent 的工作记忆，由 system prompt、压缩摘要和最近消息组成。
- Context 过长时，AgentScope 可压缩旧消息、截断大型工具结果、把内容 offload 到外部存储。
- `LocalWorkspace` 可作为本地工作区和 offloader。
- Middleware 可在 reply、reasoning、acting、model call、context compression、system prompt 等位置插入逻辑。
- 内置 Middleware 包括 tracing、预算控制、TTS、长期记忆等方向。

练习任务：

- 创建 `examples/05_middleware_context/context_compression.py`。
- 配置 `ContextConfig`，观察上下文压缩触发条件。
- 创建一个简单 Middleware，记录每次 model call 的耗时。

阶段产出：

- 一个 context compression 示例。
- 一个自定义 Middleware 示例。
- 在 `notes/troubleshooting.md` 记录 token 过长、工具输出过长时的处理策略。

## 阶段 8: RAG 与 Long-Term Memory

官方文档：

- RAG: https://docs.agentscope.io/versions/2.0.3/en/building-blocks/rag
- Long-Term Memory: https://docs.agentscope.io/versions/2.0.3/en/building-blocks/long-term-memory
- RAG Service: https://docs.agentscope.io/versions/2.0.3/en/deploy/rag

学习重点：

- RAG 由 Parser、Chunker、Embedding Model、Vector Store、KnowledgeBase 组成。
- 内置 Parser 覆盖文本、Markdown、CSV、HTML、JSON、XML、YAML、PDF、PPT、图片等类型。
- 默认向量存储可使用 `QdrantStore`，支持内存、本地和远程服务形态。
- RAG 可以在非服务场景中直接索引和检索，也可通过 RAG Service 做多租户和分布式索引。
- Long-Term Memory 通过 Middleware 持久化跨 session 的事实和偏好。

练习任务：

- 创建 `examples/06_rag_memory/local_rag.py`。
- 准备一个小型 Markdown 或 TXT 文件作为知识库。
- 完成文件索引、向量检索，并把检索结果接入 Agent。
- 记录“普通 RAG”和“长期记忆”的差别。

阶段产出：

- 一个本地 RAG 示例。
- 一个关于长期记忆适用场景的笔记。

## 阶段 9: Agent as Service、Agent Team 与部署

官方文档：

- Agent Service Architecture: https://docs.agentscope.io/versions/2.0.3/en/deploy/agent-service
- Agent Team: https://docs.agentscope.io/versions/2.0.3/en/deploy/agent-team
- API Reference: https://docs.agentscope.io/api-reference

学习重点：

- Agent as Service 面向多租户、多会话和 HTTP 服务化。
- Agent Team 支持 leader agent 创建并协调 worker agent。
- RAG Service 提供一键式多租户、分布式 RAG 服务。
- 服务化阶段需要关注 credential 管理、session 隔离、workspace 隔离、权限策略和事件流。

练习任务：

- 阅读 `examples/agent_service` 和 `examples/web_ui` 官方示例。
- 梳理一个最小服务化 Agent 的请求链路：创建 agent、创建 session、触发 chat、订阅事件流。
- 设计一个 `projects/personal_research_agent/` 项目骨架。

阶段产出：

- 一份服务化架构笔记。
- 一个个人研究 Agent 的项目设计草稿。

## 阶段 10: 复盘与个人项目

目标项目：`projects/personal_research_agent/`

建议功能：

- 接收一个研究主题。
- 自动创建计划。
- 使用工具读取本地资料。
- 使用 RAG 检索资料片段。
- 输出阶段性研究摘要。
- 使用权限系统控制文件写入和命令执行。
- 通过事件流展示执行过程。

验收标准：

- 至少包含一个自定义工具。
- 至少使用一次 Plan tools。
- 至少接入一个模型 provider。
- 至少完成一次 RAG 检索。
- README 或 notes 中记录遇到的问题和解决方式。

## 学习记录模板

每完成一个阶段，在 `notes/` 中追加如下记录：

```markdown
## YYYY-MM-DD 阶段 N: 标题

- 学习文档:
- 今日目标:
- 完成代码:
- 关键概念:
- 遇到的问题:
- 解决方式:
- 下一步:
```

## 常见注意事项

- 优先使用 `https://docs.agentscope.io/` 下的 2.x 文档。
- 搜索结果中可能出现旧版 `doc.agentscope.io` 文档；除非专门比较版本差异，否则不要混用旧 API。
- AgentScope 2.0 是 breaking change，遇到 1.x 教程时要先核对 import、类名和参数。
- 对涉及文件写入、命令执行、外部服务调用的示例，先从 `EXPLORE` 或明确权限规则开始。
- API key 不要写入代码仓库；使用环境变量或本地 `.env`，并确保 `.gitignore` 覆盖敏感文件。

## 当前进度

- [ ] 阶段 1: 快速运行第一个 Agent
- [ ] 阶段 2: Message 与 Event
- [ ] 阶段 3: Model 层与结构化输出
- [ ] 阶段 4: Agent、Tool 与 Toolkit
- [ ] 阶段 5: 权限系统与人类确认
- [ ] 阶段 6: Plan 工具与复杂任务拆解
- [ ] 阶段 7: Context、Workspace 与 Middleware
- [ ] 阶段 8: RAG 与 Long-Term Memory
- [ ] 阶段 9: Agent as Service、Agent Team 与部署
- [ ] 阶段 10: 复盘与个人项目

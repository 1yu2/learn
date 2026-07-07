# Agno 学习计划

这个仓库用于按阶段学习 [Agno](https://docs.agno.com/introduction.md)。Agno 是一个用于构建、运行和管理 Agent 平台的 SDK 与运行时：先用 SDK 构建 Agents、Teams、Workflows，再用 AgentOS 把它们作为服务运行，并获得会话、记忆、知识库、追踪、权限和审计等生产能力。

资料以 Agno 官方文档为主，建议边读边在仓库里沉淀可运行示例。每个阶段都要留下代码、运行记录和复盘笔记。

## 学习目标

- 理解 Agno 的核心抽象：Agent、Tool、Model、Memory、Knowledge、Team、Workflow、AgentOS。
- 从单个脚本开始，逐步演进到可服务化、可追踪、可持久化的 Agent 应用。
- 建立自己的示例库：每个主题至少有一个可运行 Python 文件。
- 学会判断何时用单 Agent、何时拆成 Team，何时用 Workflow 固化流程。
- 最后完成一个小型综合项目，例如“文档问答 + 工具调用 + 记忆 + AgentOS 服务”。

## 环境准备

建议使用 Python 3.12 和 `uv`。

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -U agno openai
```

如果使用 OpenAI 模型：

```bash
export OPENAI_API_KEY=sk-***
```

官方入门示例还会用到 AgentOS：

```bash
uv pip install -U "agno[os]"
```

参考：

- [Build Your First Agent](https://docs.agno.com/first-agent.md)
- [Welcome to Agno](https://docs.agno.com/introduction.md)
- [Documentation Index](https://docs.agno.com/llms.txt)

## 项目代码规划

整体代码规划见 [docs/architecture.md](docs/architecture.md)，阶段交付计划见 [docs/milestones.md](docs/milestones.md)。

仓库采用“阶段示例 + 共享包 + 最终服务化”的方式推进：

- `examples/`：按学习阶段存放可独立运行的最小示例。
- `src/agno_learn/`：沉淀跨阶段复用的配置、路径、工具、知识库、Agent 和 Workflow 代码。
- `docs/`：记录架构边界、阶段计划和后续实施顺序。
- `notes/`：记录概念理解、错误排查和复盘。
- `tmp/`：本地数据库、向量库、缓存和运行产物，不提交到 Git。

## 仓库结构

```text
.
├── README.md
├── pyproject.toml
├── .env.example
├── docs/
│   ├── architecture.md
│   └── milestones.md
├── examples/
│   ├── README.md
│   ├── 01_first_agent/
│   ├── 02_tools/
│   ├── 03_storage_memory/
│   ├── 04_knowledge/
│   ├── 05_teams/
│   ├── 06_workflows/
│   ├── 07_agentos/
│   └── 08_evals_observability/
├── notes/
│   ├── concepts.md
│   └── troubleshooting.md
├── src/
│   └── agno_learn/
│       ├── config.py
│       ├── paths.py
│       ├── agents/
│       ├── tools/
│       ├── knowledge/
│       └── workflows/
├── tests/
│   └── README.md
└── tmp/
```

当前先提交规划和目录骨架，后续每个阶段再补 `main.py`、运行说明、测试和复盘。

## 阶段 0：理解 Agno 全貌

目标：先建立地图，不急着写复杂代码。

阅读：

- [Welcome to Agno](https://docs.agno.com/introduction.md)
- [SDK Introduction](https://docs.agno.com/sdk/introduction.md)
- [What is AgentOS?](https://docs.agno.com/agent-os/introduction.md)

重点理解：

- Agno SDK 负责构建 Agents、Teams、Workflows。
- AgentOS 是 FastAPI 运行时，用来把 Agent 系统变成服务。
- AgentOS 可以管理 API、会话、数据库、记忆、知识、追踪、权限和人工审批。
- 数据默认运行在自己的基础设施和数据库里。

产出：

- 在 `notes/concepts.md` 记录核心概念。
- 画出一张简单关系图：Model -> Agent -> Tool/Memory/Knowledge -> Team/Workflow -> AgentOS。

## 阶段 1：第一个 Agent

目标：跑通最小 Agent，理解模型、指令、响应流。

阅读：

- [Build Your First Agent](https://docs.agno.com/first-agent.md)
- [What are Agents?](https://docs.agno.com/agents/overview.md)
- [Building Agents](https://docs.agno.com/agents/building-agents.md)
- [Running Agents](https://docs.agno.com/agents/running-agents.md)
- [Debugging Agents](https://docs.agno.com/agents/debugging-agents.md)

练习：

- 在 `examples/01_first_agent/` 创建一个最小 Agent。
- 尝试不同 `instructions`，观察输出变化。
- 使用 `stream=True` 打印流式响应。
- 记录一次失败或不稳定输出，并写下如何调试。

检查点：

- 能解释 Agent 是“围绕无状态模型的有状态控制循环”。
- 能说清楚 `instructions`、`model`、`markdown`、`print_response` 的作用。

## 阶段 2：工具调用 Tools

目标：让 Agent 从“会回答”变成“会行动”。

阅读：

- [What are Tools?](https://docs.agno.com/tools/overview.md)
- [Agent Tools](https://docs.agno.com/tools/agent.md)
- [Creating Tools](https://docs.agno.com/tools/creating-tools/overview.md)
- [Toolkits](https://docs.agno.com/tools/toolkits/overview.md)
- [MCP Tools](https://docs.agno.com/tools/mcp/overview.md)

重点理解：

- Tool 本质上是 Agent 可调用的 Python 函数或 Toolkit。
- Agno 会根据函数签名和 docstring 生成模型可理解的工具定义。
- 工具可以访问运行上下文、会话状态、媒体文件，也可以返回结构化结果。
- 异步运行时，多个工具调用可以并发执行，前提是模型支持并行 function calling。

练习：

- 在 `examples/02_tools/` 写一个自定义工具，例如天气、计算器、文件摘要或网页搜索模拟工具。
- 给工具补完整 docstring，观察工具描述如何影响调用。
- 尝试一个官方 Toolkit，例如 HackerNews、Workspace 或 Web Search 相关工具。
- 做一个“必须调用工具才能回答”的问题集。

检查点：

- 能解释工具 schema 是怎么从 Python 函数生成的。
- 能判断什么时候该写自定义工具，什么时候使用 Toolkit。

## 阶段 3：模型、输入输出与结构化结果

目标：掌握模型选择、响应格式和结构化输出。

阅读：

- [Models](https://docs.agno.com/models/overview.md)
- [Input and Output](https://docs.agno.com/input-output/overview.md)
- [Agent with Structured Output](https://docs.agno.com/agents/usage/agent-with-structured-output.md)
- [Multimodal](https://docs.agno.com/multimodal/overview.md)

练习：

- 在 `examples/01_first_agent/` 中切换至少两种模型配置。
- 用 Pydantic 定义结构化输出，例如 `TaskPlan`、`ResearchSummary` 或 `BugReport`。
- 对同一个输入比较普通文本输出和结构化输出。

检查点：

- 能说明什么时候需要结构化输出。
- 能将 Agent 输出稳定地交给下一段 Python 逻辑处理。

## 阶段 4：数据库、历史与 Memory

目标：让 Agent 有会话历史和跨会话记忆。

阅读：

- [Database](https://docs.agno.com/database/overview.md)
- [What is Memory?](https://docs.agno.com/memory/overview.md)
- [Agent Memory](https://docs.agno.com/memory/agent/overview.md)
- [Working with Memories](https://docs.agno.com/memory/working-with-memories/overview.md)
- [History](https://docs.agno.com/history/overview.md)

重点理解：

- Session history 保存对话消息，用于连续上下文。
- Memory 保存用户事实和偏好，例如名字、习惯、长期偏好。
- `update_memory_on_run=True` 是自动记忆，适合多数场景。
- `enable_agentic_memory=True` 让 Agent 自己决定何时创建、更新、删除记忆。
- 两种记忆模式不要同时启用，Agentic Memory 会优先生效。

练习：

- 在 `examples/03_storage_memory/` 用 SQLite 保存会话。
- 让 Agent 记住一个用户偏好，然后在下一轮对话中调用出来。
- 手动读取某个 `user_id` 的 memories，写到学习笔记里。

检查点：

- 能区分 history、session state、memory。
- 能说明生产环境中为什么必须关注用户隔离和数据清理。

## 阶段 5：Knowledge 与 RAG

目标：让 Agent 基于自己的资料回答问题，而不是只依赖模型参数知识。

阅读：

- [Knowledge Overview](https://docs.agno.com/knowledge/overview.md)
- [Knowledge Quickstart](https://docs.agno.com/knowledge/quickstart.md)
- [Knowledge for Agents](https://docs.agno.com/knowledge/agents/overview.md)
- [Search and Retrieval](https://docs.agno.com/knowledge/concepts/search-and-retrieval/overview.md)
- [Readers](https://docs.agno.com/knowledge/concepts/readers/overview.md)
- [Chunking](https://docs.agno.com/knowledge/concepts/chunking/overview.md)
- [Embedders](https://docs.agno.com/knowledge/concepts/embedder/overview.md)
- [Vector Stores](https://docs.agno.com/knowledge/vector-stores/pgvector/overview.md)

重点理解：

- Knowledge 包含内容读取、分块、embedding、向量库检索和上下文注入。
- Agentic RAG 是默认思路：Agent 判断何时搜索知识库。
- Traditional RAG 更适合必须始终带上下文的流程。
- 过滤、重排、混合搜索会直接影响回答质量。

练习：

- 在 `examples/04_knowledge/` 用本地 Markdown 或 URL 构建一个小知识库。
- 使用 ChromaDB 或其他本地向量库做第一版。
- 准备 5 个问题，比较有无 knowledge 时的回答差异。
- 记录一次错误引用或答非所问，尝试通过 chunking 或过滤改善。

检查点：

- 能解释 reader、chunker、embedder、vector db 各自负责什么。
- 能说明 Agentic RAG 与 Traditional RAG 的差别。

## 阶段 6：Teams 多 Agent 协作

目标：用多个专长 Agent 分工解决复杂任务。

阅读：

- [What are Teams?](https://docs.agno.com/teams/overview.md)
- [Building Teams](https://docs.agno.com/teams/building-teams.md)
- [Running Teams](https://docs.agno.com/teams/running-teams.md)
- [Debugging Teams](https://docs.agno.com/teams/debugging-teams.md)
- [Delegation](https://docs.agno.com/teams/delegation.md)

重点理解：

- Team 是一组 Agents 或子 Teams，由 leader 根据角色进行协调。
- Team 适合多领域、多工具、多上下文的任务。
- 单 Agent 更便宜、更简单；不确定时先从单 Agent 开始。
- Team 模式包括 coordinate、route、broadcast 等协作方式。

练习：

- 在 `examples/05_teams/` 创建一个研究团队：Researcher、Writer、Reviewer。
- 给不同成员配置不同工具和角色。
- 比较单 Agent 与 Team 在同一任务上的效果、成本和可调试性。

检查点：

- 能判断“这个任务是否真的需要 Team”。
- 能定位某个成员输出质量差时该改 role、instructions 还是工具。

## 阶段 7：Workflows 固化流程

目标：把重复任务变成可预测、可审计的步骤流水线。

阅读：

- [What are Workflows?](https://docs.agno.com/workflows/overview.md)
- [Building Workflows](https://docs.agno.com/workflows/building-workflows.md)
- [Running Workflows](https://docs.agno.com/workflows/running-workflows.md)
- [Conversational Workflows](https://docs.agno.com/workflows/conversational-workflows.md)

重点理解：

- Workflow 由 Steps 组成，Step 可以是 Agent、Team、Function 或嵌套 Workflow。
- 步骤可以顺序、并行、循环或按条件执行。
- 需要可重复、可审计、输入输出明确的任务时，优先考虑 Workflow。
- 需要开放式协作和动态分工时，优先考虑 Team。

练习：

- 在 `examples/06_workflows/` 写一个“资料收集 -> 摘要 -> 审稿 -> 输出”的 Workflow。
- 加一个普通 Python function 作为中间步骤，例如清洗输入或保存结果。
- 记录每一步的输入输出。

检查点：

- 能解释 Team 和 Workflow 的边界。
- 能把一个自由对话任务改造成可重复流程。

## 阶段 8：AgentOS 服务化

目标：把本地 Agent 系统变成可运行的 API 服务。

阅读：

- [What is AgentOS?](https://docs.agno.com/agent-os/introduction.md)
- [Run Your AgentOS](https://docs.agno.com/agent-os/run-your-os.md)
- [Connect Your AgentOS](https://docs.agno.com/agent-os/connect-your-os.md)
- [Using the API](https://docs.agno.com/agent-os/using-the-api.md)
- [AgentOS Configuration](https://docs.agno.com/agent-os/config.md)
- [AgentOS Security](https://docs.agno.com/agent-os/security/overview.md)
- [Tracing](https://docs.agno.com/agent-os/tracing/overview.md)

重点理解：

- AgentOS 是 FastAPI app，用于运行 agents、teams、workflows。
- 它提供流式 API、会话隔离、持久化、追踪、调度、RBAC、审计和审批。
- Control Plane 是管理和调试 UI，运行时和数据仍在自己的基础设施里。

练习：

- 在 `examples/07_agentos/` 把前面做过的 Agent 包装成 AgentOS。
- 启动本地服务并打开 `/docs`。
- 连接 [os.agno.com](https://os.agno.com)，查看 sessions 和 traces。
- 给服务添加 SQLite 数据库，确认重启后会话仍可查询。

检查点：

- 能说明 SDK 和 AgentOS 的职责差异。
- 能解释为什么服务化后必须考虑 auth、隔离、日志和审计。

## 阶段 9：评测、观测与生产化

目标：从“能跑”推进到“能评估、能调试、能上线”。

阅读：

- [Evals](https://docs.agno.com/evals/overview.md)
- [Examples: Evals](https://docs.agno.com/examples/evals/overview.md)
- [AgentOS Tracing](https://docs.agno.com/agent-os/tracing/overview.md)
- [Deploy AgentOS](https://docs.agno.com/deploy/introduction.md)
- [Human-in-the-Loop](https://docs.agno.com/agent-os/usage/hitl.md)
- [Approvals](https://docs.agno.com/agent-os/approvals/overview.md)

练习：

- 在 `examples/08_evals_observability/` 建一个小评测集。
- 对同一个 Agent 的不同 instructions 版本做对比。
- 记录 token、延迟、失败样例、工具调用次数。
- 尝试一个需要人工审批的工具调用流程。

检查点：

- 能定义一个 Agent 的成功标准。
- 能用 traces 定位回答错误、工具失败或上下文污染。

## 综合项目建议

选择一个足够小但覆盖核心能力的项目：

1. 文档问答助手：读取本仓库笔记，回答 Agno 学习问题。
2. 研究写作流水线：搜索资料、生成摘要、写文章、审稿。
3. 个人助理：记住用户偏好，调用工具整理日程或任务。
4. 本地文件整理 Agent：参考官方 Sorting Hat 示例分析并整理目录。

最低验收标准：

- 有一个 Agent 使用至少一个自定义工具。
- 有持久化数据库，能保存 session 或 memory。
- 有 Knowledge 或 Team/Workflow 中的任意一个进阶能力。
- 能通过 AgentOS 作为服务运行。
- README 或 `notes/` 中有运行方式、失败记录和复盘。

## 学习节奏

建议每个阶段都按这个循环推进：

1. 阅读对应官方文档。
2. 写一个最小可运行示例。
3. 记录运行命令、输出截图或关键日志。
4. 写下一个失败案例和修正方式。
5. 提交一次 Git commit。

推荐 commit 粒度：

```text
docs: add agno concept notes
feat: add first agno agent example
feat: add custom tool example
feat: add sqlite memory example
feat: add knowledge rag example
feat: add research team example
feat: add content workflow example
feat: serve agent with agentos
docs: summarize eval findings
```

## 官方文档入口

- [Agno Introduction](https://docs.agno.com/introduction.md)
- [First Agent](https://docs.agno.com/first-agent.md)
- [Agents](https://docs.agno.com/agents/overview.md)
- [Tools](https://docs.agno.com/tools/overview.md)
- [Models](https://docs.agno.com/models/overview.md)
- [Database](https://docs.agno.com/database/overview.md)
- [Memory](https://docs.agno.com/memory/overview.md)
- [Knowledge](https://docs.agno.com/knowledge/overview.md)
- [Teams](https://docs.agno.com/teams/overview.md)
- [Workflows](https://docs.agno.com/workflows/overview.md)
- [AgentOS](https://docs.agno.com/agent-os/introduction.md)
- [Examples](https://docs.agno.com/examples/introduction.md)
- [API Reference](https://docs.agno.com/api-reference/home/api-information.md)
- [Full Documentation Index](https://docs.agno.com/llms.txt)

## 当前进度

- [ ] 阶段 0：理解 Agno 全貌
- [ ] 阶段 1：第一个 Agent
- [ ] 阶段 2：工具调用 Tools
- [ ] 阶段 3：模型、输入输出与结构化结果
- [ ] 阶段 4：数据库、历史与 Memory
- [ ] 阶段 5：Knowledge 与 RAG
- [ ] 阶段 6：Teams 多 Agent 协作
- [ ] 阶段 7：Workflows 固化流程
- [ ] 阶段 8：AgentOS 服务化
- [ ] 阶段 9：评测、观测与生产化

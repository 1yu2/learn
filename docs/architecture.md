# 项目代码规划

这个仓库采用“阶段示例 + 共享包 + 最终服务化”的结构。每个学习阶段都先在 `examples/` 中形成一个可独立运行的最小示例；当逻辑开始重复时，再沉淀到 `src/agno_learn/` 共享包里；最后通过 AgentOS 把可复用 Agent、Team、Workflow 组合成服务。

## 设计原则

- 示例优先：每个阶段都有独立目录、入口脚本和 README，方便单独运行和回滚。
- 共享逻辑后置：先允许示例重复，确认模式稳定后再抽到 `src/agno_learn/`。
- 本地数据隔离：数据库、向量库、缓存和临时文件统一进入 `tmp/`，不提交到 Git。
- 文档伴随代码：每个阶段完成后补充运行命令、失败记录和复盘。
- 服务化收敛：AgentOS 阶段只组合已验证的 Agent、Tool、Knowledge 和 Workflow，不在服务层堆业务逻辑。

## 目录职责

```text
.
├── docs/
│   ├── architecture.md       # 仓库结构、模块边界、代码演进方式
│   └── milestones.md         # 阶段交付计划
├── examples/
│   ├── 01_first_agent/       # 最小 Agent 与指令实验
│   ├── 02_tools/             # 自定义工具与 Toolkit
│   ├── 03_storage_memory/    # SQLite、history、memory
│   ├── 04_knowledge/         # Knowledge 与 RAG
│   ├── 05_teams/             # 多 Agent Team
│   ├── 06_workflows/         # 可重复 Workflow
│   ├── 07_agentos/           # AgentOS 服务化
│   └── 08_evals_observability/ # evals、tracing、审批
├── notes/
│   ├── concepts.md           # 概念笔记
│   └── troubleshooting.md    # 错误与排查记录
├── src/
│   └── agno_learn/
│       ├── config.py         # 环境变量与运行配置
│       ├── paths.py          # 项目路径与本地数据目录
│       ├── agents/           # 可复用 Agent factory
│       ├── tools/            # 自定义工具
│       ├── knowledge/        # 知识库加载、分块、检索封装
│       └── workflows/        # 可复用 Workflow builder
├── tests/                    # 共享逻辑的单元测试
├── tmp/                      # 本地运行数据，不提交
├── .env.example              # 本地环境变量模板
├── pyproject.toml            # Python 项目配置
└── README.md                 # 学习路线与入口
```

## 代码演进路线

### 1. 示例层

先在 `examples/<stage>/` 写最小脚本。示例脚本可以直接创建 Agent、Tool、Knowledge 或 Workflow，但必须保持一个文件只演示一个主题。

推荐命名：

- `main.py`：当前阶段主入口。
- `README.md`：运行命令、依赖、观察点。
- `fixtures/`：小型测试资料或 prompt 样例。

### 2. 共享包层

当两个以上示例需要同一段逻辑时，再抽到 `src/agno_learn/`。

- `config.py`：读取 `AGNO_MODEL_ID`、`AGNO_LEARN_DATA_DIR` 等环境变量。
- `paths.py`：提供仓库路径、示例路径、本地数据目录。
- `agents/`：封装稳定的 Agent factory，例如文档问答 Agent、研究 Agent。
- `tools/`：封装自定义工具，例如文件摘要、计算器、受控 shell 工具。
- `knowledge/`：封装知识库加载、reader、chunking、vector db 选择。
- `workflows/`：封装研究写作、文档问答等可复用流程。

### 3. 服务层

`examples/07_agentos/` 只负责把共享包中的 Agent、Team、Workflow 注入 AgentOS。服务层需要清楚处理：

- 数据库位置。
- session 与 user 隔离。
- tracing 开关。
- API 与 Control Plane 连接方式。
- 本地开发与未来部署的配置差异。

## 数据流

```text
User input
  -> example script or AgentOS API
  -> RuntimeConfig
  -> Agent / Team / Workflow
  -> Tools / Knowledge / Memory / Database
  -> Run output
  -> notes, eval results, traces
```

## 测试策略

- 示例脚本以手动运行和记录输出为主。
- `src/agno_learn/` 中的纯 Python 逻辑必须补单元测试。
- 外部模型调用不进入默认单元测试，避免测试依赖 API key 和网络。
- 工具函数优先拆成“纯逻辑 + Agno tool wrapper”，纯逻辑可以用 pytest 覆盖。
- AgentOS 阶段至少保留一个本地启动命令和 `/docs` 验证步骤。

## 提交策略

每个阶段单独提交，commit message 使用下面的模式：

```text
docs: plan agno learning repository
feat: add first agent example
feat: add custom tool example
feat: add sqlite memory example
feat: add knowledge rag example
feat: add research team example
feat: add content workflow example
feat: serve examples with agentos
docs: add eval and tracing notes
```

# 阶段交付计划

| 阶段 | 主题 | 主要代码位置 | 验收标准 |
| --- | --- | --- | --- |
| 0 | 项目地图与概念 | `README.md`, `docs/`, `notes/` | 能解释 Agno SDK、AgentOS、Agent、Tool、Memory、Knowledge、Team、Workflow 的关系 |
| 1 | First Agent | `examples/01_first_agent/` | 有一个可运行 Agent，支持流式输出 |
| 2 | Tools | `examples/02_tools/`, `src/agno_learn/tools/` | 至少一个自定义工具和一个 Toolkit 示例 |
| 3 | Structured Output | `examples/01_first_agent/` | 有 Pydantic 结构化输出示例 |
| 4 | Storage & Memory | `examples/03_storage_memory/` | SQLite 保存 session 或 memory，能跨轮次读取 |
| 5 | Knowledge / RAG | `examples/04_knowledge/`, `src/agno_learn/knowledge/` | 本地文档可被检索并用于回答 |
| 6 | Teams | `examples/05_teams/`, `src/agno_learn/agents/` | Researcher/Writer/Reviewer 等多角色协作 |
| 7 | Workflows | `examples/06_workflows/`, `src/agno_learn/workflows/` | 有顺序步骤、函数步骤和明确输入输出 |
| 8 | AgentOS | `examples/07_agentos/` | 本地服务启动后可访问 `/docs`，能连接 Control Plane |
| 9 | Evals & Observability | `examples/08_evals_observability/` | 有小评测集、trace 记录和失败复盘 |

## 当前优先级

1. 完成阶段 1 的最小 Agent。
2. 抽出 `load_runtime_config()` 的使用方式，统一模型配置。
3. 完成阶段 2 的自定义工具示例。
4. 再引入数据库、记忆和知识库，避免一开始就把服务层复杂化。

## 不做的事

- 不在早期引入 Web 前端。
- 不把真实 API key、数据库文件、向量库文件提交到仓库。
- 不在示例阶段追求复杂抽象；只有重复两次以上的逻辑才进入 `src/`。
- 不让默认测试依赖真实模型调用。

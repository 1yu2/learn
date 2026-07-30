# Palantir (Foundry / AIP) 学习

企业级数据平台，核心概念是 **Ontology（本体）**——把数据资产映射成业务对象、属性和动作，是 Foundry 和 AIP 的中枢。

> ⚠️ **环境前提**：Foundry 是企业软件，学习需要平台访问权限（公司账号、AIP 试用/Developer Tier，或 [learn.palantir.com](https://learn.palantir.com/) 内的沙盒课程）。没有环境时先学概念文档。

## 官方资源

| 资源 | 链接 | 用途 |
| --- | --- | --- |
| Palantir Learn 官方培训 | https://learn.palantir.com/ | 主入口：分角色课程、视频教程、认证 |
| Foundry 官方文档 | https://palantir.com/docs/foundry/getting-started/overview/ | 权威功能参考 |
| Foundry Learning Paths | https://learn.palantir.com/page/foundry-learning-paths | 官方路径（含 60-90min 端到端 Speedrun） |
| 认证 | learn.palantir.com 内 Foundry Certification | 检验目标 |

## 学习路径与进度

### 阶段 1：平台基础

| # | 主题 | 说明 | 状态 |
| --- | --- | --- | --- |
| 1 | Foundry 概览与核心概念 | 平台架构、Project、Dataset、Lineage | ⬜ |
| 2 | 数据接入（Data Connection / Sync） | 源系统接入、增量同步 | ⬜ |
| 3 | 数据转换（Pipeline Builder / Code Repositories） | 低代码管道 vs Spark/SQL/Python 代码管道 | ⬜ |
| 4 | 数据治理与安全 | Markings、Restricted Views、权限模型 | ⬜ |

### 阶段 2：Ontology 与应用（核心）

| # | 主题 | 说明 | 状态 |
| --- | --- | --- | --- |
| 5 | Ontology 本体设计 | Object Types、Link Types、Properties——把数据集映射为业务对象 | ⬜ |
| 6 | Actions 与 Functions | 写回操作、业务逻辑函数（TypeScript） | ⬜ |
| 7 | Workshop / 应用搭建 | 基于 Ontology 搭操作型应用 | ⬜ |
| 8 | 端到端实战（官方 Speedrun） | 从原始数据到可操作应用全流程 | ⬜ |

### 阶段 3：AIP（AI 平台）

| # | 主题 | 说明 | 状态 |
| --- | --- | --- | --- |
| 9 | AIP 概览：LLM + Ontology | Agent 如何基于本体调用工具/数据 | ⬜ |
| 10 | AIP Logic / Agent Studio | 搭 AI Agent 工作流 | ⬜ |

## 与其他线的呼应

- **Palantir Ontology ↔ Neo4j 本体/知识图谱**（[topics/neo4j](../neo4j/)）：同一套"实体-关系-语义建模"思想，建议相邻学习，互相印证（方法论 L2 检验：写出两者建模哲学的差异）
- **AIP ↔ AI Agent 书**（[books/ai-agent-book](../../books/ai-agent-book.md)）第 4 章工具/MCP：AIP Agent 的工具调用就是企业版的 function calling

## 笔记

写在 `topics/palantir/notes/`。核对"真懂"：概念主题做 L2（白纸写出 Foundry 各组件关系图），操作主题必须 L4（在平台上不看教程做出管道/Ontology 对象）。

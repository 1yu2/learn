# Neo4j 知识图谱与本体学习

图数据库 + 知识图谱 + 本体（Ontology）建模。官方课程全部免费且带沙盒实验。

## 官方资源

| 资源 | 链接 | 用途 |
| --- | --- | --- |
| GraphAcademy（免费官方课程） | https://graphacademy.neo4j.com/ | 主入口，全部免费、带在线实验环境 |
| Neo4j 官方文档 | https://neo4j.com/docs/ | 权威参考 |
| Neo4j Aura 免费实例 | https://neo4j.com/cloud/platform/aura-graph-database/ | 免费云数据库，做实验用 |
| 开发者指南：构建知识图谱 | https://neo4j.com/developer/knowledge-graph/ | KG 实战指南 |
| n10s（neosemantics） | https://neo4j.com/labs/neosemantics/ | RDF/OWL/本体导入导出插件 |

## 学习路径与进度

### 阶段 1：图数据库基础（GraphAcademy Fundamentals 路径）

| # | 课程 | 时长 | 状态 |
| --- | --- | --- | --- |
| 1 | [Neo4j Fundamentals](https://graphacademy.neo4j.com/courses/neo4j-fundamentals/) | ~1h | ⬜ |
| 2 | [Cypher Fundamentals](https://graphacademy.neo4j.com/courses/cypher-fundamentals/) | ~1h | ⬜ |
| 3 | [Graph Data Modeling Fundamentals](https://graphacademy.neo4j.com/courses/modeling-fundamentals/) | ~2h | ⬜ |
| 4 | [Importing Data Fundamentals](https://graphacademy.neo4j.com/courses/importing-fundamentals/) | ~2h | ⬜ |

### 阶段 2：知识图谱与本体（重点）

| # | 主题 | 说明 | 状态 |
| --- | --- | --- | --- |
| 5 | 知识图谱概念与建模 | 实体/关系/属性图 vs 三元组；[官方 KG 开发者指南](https://neo4j.com/developer/knowledge-graph/) | ⬜ |
| 6 | 本体基础：RDF / RDFS / OWL | 类、属性、层级、约束——本体的形式语义 | ⬜ |
| 7 | n10s 实战：导入/导出本体 | 用 neosemantics 把 OWL 本体映射到属性图 | ⬜ |
| 8 | 图数据建模进阶 | 关系 vs 属性取舍、超节点、时间建模 | ⬜ |

### 阶段 3：GraphRAG 与 LLM 结合（选学）

| # | 主题 | 说明 | 状态 |
| --- | --- | --- | --- |
| 9 | [Generative AI & GraphRAG 路径](https://graphacademy.neo4j.com/categories/generative-ai)（10 门课） | 向量检索 + 图遍历的混合 RAG | ⬜ |
| 10 | LLM 抽取构建知识图谱 | [llm-graph-builder 课程](https://graphacademy.neo4j.com/)：非结构化文本 → KG | ⬜ |

## 与其他线的呼应

- **本体建模 ↔ Palantir Ontology**（[topics/palantir](../palantir/)）：同一思想在图数据库和企业数据平台两种落地，L2 检验：写出两者的建模差异
- **知识图谱 ↔ AI Agent 书**（[books/ai-agent-book](../../books/ai-agent-book.md)）第 3 章"用户记忆和知识库"：RAG/知识图谱小节可直接互相印证
- **GraphRAG ↔ CS224n** slides 10（Agents, Tool Use, and RAG）

## 笔记

写在 `topics/neo4j/notes/`。核对"真懂"：阶段 1 以 L4 为主（Cypher 查询不看答案写出来），阶段 2 本体部分加 L3——给一个业务场景，能独立设计出本体 schema（类、关系、约束）并说明取舍理由。

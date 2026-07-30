# Learning Notes

个人学习笔记仓库，按课程（courses）和技术主题（topics）组织。

> 学习执行规范（怎么核对"真的学会了"）：[docs/learning-methodology.md](docs/learning-methodology.md)

## Courses

| 课程 | 简介 | 状态 |
| --- | --- | --- |
| [cs336](courses/cs336/) | Language Modeling from Scratch (Stanford) | 未开始 |
| [cs224n](courses/cs224n/) | NLP with Deep Learning (Stanford) | 未开始 |

## Books

| 书 | 简介 | 状态 |
| --- | --- | --- |
| [深入理解 AI Agent](books/ai-agent-book.md) | AI Agent 设计原理与工程实践（bojieli，10 章 + 93 个实验） | 0/10 章 |
| [设计数据密集型应用（DDIA）](books/ddia.md) | 数据系统原理经典（Kleppmann 著，冯若航译，14 章） | 0/14 章 |

## Topics

| 主题 | 简介 | 状态 |
| --- | --- | --- |
| [dbt](topics/dbt/) | 数据转换工具（官方文档 + dbt Learn 课程，16 主题三阶段） | 未开始 |
| [Palantir](topics/palantir/) | Foundry/AIP 企业数据平台，重点是 Ontology 本体 | 未开始 |
| [Neo4j](topics/neo4j/) | 图数据库、知识图谱与本体建模（GraphAcademy 免费课程） | 未开始 |

## 目录约定

- `courses/<course>/notes/` — 讲义笔记（markdown）
- `courses/<course>/assignments/` — 作业代码，独立依赖环境
- 每门课的 `README.md` 记录课程链接、进度和笔记索引
- `books/<书名>.md` — 书籍索引与章节进度；`books/<书仓库>/` 为 clone 的源码仓库（gitignore，不入本库）；笔记放 `books/notes/<书名>/`

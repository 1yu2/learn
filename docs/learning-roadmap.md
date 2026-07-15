# AgentScope 学习路线与掌握度验收

## 使用方式

每周按“阅读 -> 实现 -> 解释 -> 测试 -> 复盘”推进。每个阶段至少保留四类证据：

1. 一个可以运行或编译的代码产出。
2. 一段概念笔记，说明组件职责和数据流。
3. 一次故意制造的失败，以及日志、原因和修复。
4. 一段 3-5 分钟的口头或书面讲解。

阶段问题得分至少 7/10 且四类证据齐全，才算阶段通过。全部十阶段通过，且题库总分达到 80%，才算具备进入项目开发的基础。

## 四个检查点

| 检查点 | 阶段 | 目标 |
| --- | --- | --- |
| 基础 | 1-3 | 能运行 Agent，理解消息事件和模型输出 |
| 执行与控制 | 4-6 | 能接工具、处理权限和拆解任务 |
| 上下文与检索 | 7-8 | 能控制上下文，并完成本地 RAG/记忆实验 |
| 服务与项目 | 9-10 | 能说明服务化边界并交付一个可验收项目 |

## 十阶段任务

| 阶段 | 目标与阅读 | 实践任务 | 产出与掌握证据 |
| --- | --- | --- | --- |
| 1. Quickstart | 理解 `Agent`、`Model`、`Toolkit`；阅读 Quickstart | 运行 `hello_agent.py`，比较 `reply` 与 `reply_stream` | 运行日志、三者关系笔记、一次错误 key 诊断 |
| 2. Message/Event | 区分 `Msg`、`Event`、事件 block 和 `reply_id` | 运行事件检查脚本并重建最终消息 | 事件数据流图、字段解释、一个事件缺失实验 |
| 3. Model/结构化输出 | 理解 credential、formatter、chat response 和 schema | 用 Pydantic schema 生成结构化结果 | schema 示例、非法输出处理、provider 差异记录 |
| 4. Agent/Tool/Toolkit | 理解 ReAct、工具 schema、FunctionTool 与 ToolBase | 实现只读函数工具和受控自定义工具 | 工具调用日志、参数校验测试、一次拒绝调用 |
| 5. 权限/HITL | 理解 `ALLOW`、`DENY`、`ASK` 和恢复事件 | 对比权限模式并处理用户确认 | 权限矩阵、确认恢复演示、误授权复盘 |
| 6. Plan | 理解任务状态、依赖和 agent state | 用 Plan tools 拆解三步以上任务 | 任务状态记录、阻塞任务实验、计划设计说明 |
| 7. Context/Middleware | 理解 context、workspace、压缩和 middleware hook | 触发压缩并记录 model call 耗时 | 压缩前后对比、middleware 测试、token 超限诊断 |
| 8. RAG/Memory | 区分 parser、chunker、embedding、vector store 和长期记忆 | 索引本地 Markdown 并完成检索 | 检索样例、引用来源、RAG 与 memory 对比笔记 |
| 9. Service/Team | 理解 session、tenant、事件流、leader/worker 边界 | 画请求链路并阅读官方服务示例 | 服务架构图、隔离风险清单、失败恢复说明 |
| 10. Capstone | 把前九阶段组合成可演示应用 | 完成个人研究 Agent 的 MVP 和验收 | README 演示、自动化测试、评分表和复盘 |

## 推荐节奏

- 第 1 天：只读官方 2.x 文档，写出 5 个自己的问题。
- 第 2-3 天：完成示例最小改动，禁止只复制代码不解释。
- 第 4 天：故意改变一个参数或权限，记录失败证据。
- 第 5 天：补测试、整理笔记、完成题库自测。
- 第 6-7 天：复盘本周概念，并把仍低于 1 分的题目加入下周任务。

## 进入项目的门槛

- 十个阶段的四类证据齐全。
- 每阶段问题不少于 7/10，总题库不少于 80%。
- 能不看资料解释 `Msg` 与 `Event`、工具权限、RAG 检索链路和 context 压缩。
- 能运行离线测试，并能说明真实模型调用与离线验证的边界。
- 至少完成一次“工具被拒绝”或“模型输出不符合 schema”的排错记录。

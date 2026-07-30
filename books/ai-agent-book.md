# 深入理解 AI Agent：设计原理与工程实践

- 作者：李博杰（bojieli）
- 仓库：https://github.com/bojieli/ai-agent-book （已 clone 到 `books/ai-agent-book/`，不入本库）
- 在线阅读：https://bojieli.github.io/ai-agent-book/
- 核心公式：**Agent = LLM + 上下文 + 工具**
- 进度：0/10 章

## 章节进度

| 章 | 主题 | 核心内容 | 阅读 | 实验 | 思考题 |
| --- | --- | --- | --- | --- | --- |
| 1 | Agent 基础知识 | Agent = LLM + 上下文 + 工具；Harness 工程才是竞争力 | ⬜ | ⬜ 0/4 | ⬜ 0/10 |
| 2 | 上下文工程 | KV Cache、提示工程、Agent Skills、上下文压缩 | ⬜ | ⬜ 0/9 | ⬜ 0/9 |
| 3 | 用户记忆和知识库 | 用户记忆、RAG、结构化索引、知识图谱 | ⬜ | ⬜ 0/13 | ⬜ 0/9 |
| 4 | 工具 | MCP 协议、感知/执行/协作工具、事件驱动异步 Agent | ⬜ | ⬜ 0/7 | ⬜ 0/7 |
| 5 | Coding Agent 与代码生成 | 生产级 Coding Agent 全景 | ⬜ | ⬜ 0/12 | ⬜ 0/10 |
| 6 | Agent 的评估 | 评估环境、指标、统计显著性、评估驱动选型 | ⬜ | ⬜ 0/11 | ⬜ 0/8 |
| 7 | 模型后训练 | SFT/RL 选择、工具调用内化、样本效率 | ⬜ | ⬜ 0/16 | ⬜ 0/13 |
| 8 | Agent 的持续进化 | 从运行轨迹学习：知识、指令、程序与参数更新 | ⬜ | ⬜ 0/8 | ⬜ 0/6 |
| 9 | 多模态与实时交互 | 语音三范式、Computer Use、机器人 | ⬜ | ⬜ 0/7 | ⬜ 0/9 |
| 10 | 多 Agent 协作 | 协作框架、上下文共享/隔离、Agent 社会 | ⬜ | ⬜ 0/7 | ⬜ 0/12 |

## 使用说明

- 正文：`books/ai-agent-book/book/chapterN.md`；配套实验代码：`books/ai-agent-book/chapterN/`
- 全书共 93 个配套实验（70+ 可独立运行），每章实验清单见各 `chapterN/README.md`
- 自己的笔记写在 `books/notes/ai-agent-book/`（如 `ch01-basics.md`）

## 思考题（全书 93 题）

每章末尾有带难度星级的思考题（★-★★★），官方参考答案在 `books/ai-agent-book/book/reference-answers.md`。

做题规范（这是检验"真懂"的 L2/L3 手段，关键在**闭卷**）：

1. 每章一个答题文件：`books/notes/ai-agent-book/chNN-questions.md`（第一章已建好模板：[ch01-questions.md](notes/ai-agent-book/ch01-questions.md)）
2. **先闭卷答完全章**，再看参考答案——提前看答案等于白做
3. 逐题对照：把漏掉/答错的点写在"对照差距"里并标 ❌，最后写一句话结论
4. 状态标记：⬜ 未答 → 📝 已答未对照 → ✅ 对照完成；全章 ✅ 后更新上表进度
5. ★★★ 题值得多花时间，答完可以拿自己的答案和 LLM 讨论一轮（参考答案本身也是 AI 生成的，不必当标准答案）
6. 复习时只重做 ❌ 过的题（对应方法论的间隔复习）

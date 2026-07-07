# Tests

默认测试只覆盖 `src/agno_learn/` 中不依赖真实模型调用的纯 Python 逻辑。

计划：

- 配置、路径、输入解析等逻辑用 pytest 覆盖。
- Agent 调用、外部模型调用和 AgentOS 服务启动作为手动验证或单独集成测试。

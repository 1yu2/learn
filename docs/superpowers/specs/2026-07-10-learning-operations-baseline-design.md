# AgentScope 学习运营与验收基线设计

## 背景

仓库已有 AgentScope 2.x 的十阶段学习路线、可编译示例和个人研究 Agent 的项目骨架，但缺少可复制的环境配置、统一的配置读取方式，以及可客观判断“是否掌握”和“项目完成度”的材料。

## 目标

本次工作建立一套不依赖真实模型调用的学习基线：

1. 用 `.env.example` 描述默认的 DashScope 配置和本地目录配置。
2. 用轻量 Python 配置对象统一读取、校验和脱敏展示配置。
3. 用学习路线、题库和项目验收文档把学习过程转成可执行、可复盘的证据。
4. 让真实模型示例通过 `Settings` 获取 `DASHSCOPE_API_KEY` 和模型配置，避免各脚本重复读取环境变量。

## 非目标

- 不实现完整的多模型 provider 工厂。
- 不在测试中访问 DashScope 或其他外部 API。
- 不提交真实 `.env`、API key、向量库数据或运行时 workspace。
- 不把仓库改造成 Web UI、CLI 或生产服务。

## 配置设计

新增 `src/agentscope_learn/config.py`，提供不可变的 `Settings` 配置对象。

- `Settings.from_env()` 从当前进程环境读取配置；可选读取项目根目录的 `.env`，但不覆盖已经存在的系统环境变量。
- 默认 provider 为 `dashscope`，默认模型为 `qwen-plus`。
- `require_model_credentials()` 只在真正需要模型调用时检查 provider 对应的凭据。
- `redacted_summary()` 只返回可安全打印的配置，不返回 key 原文。
- provider、日志级别、布尔值和路径使用显式规则校验，错误信息指出修复方向。
- 配置模块导入时不读取环境，避免测试和普通 import 产生隐式副作用。

`.env.example` 只包含字段名和安全默认值。`.env` 继续由 `.gitignore` 忽略。

## 文档设计

- `docs/environment.md`：安装、复制模板、DashScope 配置、切换兼容 provider、离线验证和安全注意事项。
- `docs/learning-roadmap.md`：十阶段按学习目标、练习、产出和掌握证据组织，并给出阶段通过标准。
- `docs/interview-question-bank.md`：按主题收集 AgentScope 八股题，包含答题要点、追问和 0-2 分评分方式。
- `docs/capstone-evaluation.md`：比较候选项目方向，推荐本地资料研究助手，定义 MVP、增强项、测试证据和完成度评分。
- `README.md`：只增加入口链接和配置模块说明，保留现有阶段内容作为详细参考。

## 测试设计

新增 `tests/test_config.py`，覆盖：

- 默认配置和环境变量覆盖；
- provider、日志级别、布尔值和路径校验；
- 缺少模型凭据时的明确错误；
- 脱敏摘要不泄露 key；
- `.env` 的读取优先级（系统环境优先）。

测试只使用 `monkeypatch` 和临时目录，不需要 API key、网络或 AgentScope 运行时。

## 验收标准

- `.env.example` 可直接复制为 `.env` 并按文档填写。
- 配置模块在无 key 的离线环境中可导入和测试。
- `python3 -m pytest -q`、`python3 -m compileall src examples tests` 和 `git diff --check` 通过。
- 学习计划每个阶段都有可观察产出，题库有评分门槛，项目文档有可复现的验收证据。
- 文档不包含真实密钥，不引入与当前 AgentScope 2.x 基线冲突的 API 约定。

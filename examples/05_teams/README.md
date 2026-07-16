# 05 Teams

本目录学习如何把多个专长 Agent 组成 Team，由 Team Leader 选择成员、分配任务并生成结果。
重点不是“Agent 越多越好”，而是理解什么时候需要分工、如何选择协作模式，以及如何观察
成员调用、工具结果、延迟和 token 成本。

## 版本基线

- 本仓库 `uv.lock` 锁定 Agno `2.7.3`。
- 本文按照 Agno 官方 Teams 文档和本地 `2.7.3` API 交叉核对。
- 官方在线文档会继续更新；代码能否运行以 `uv.lock` 和本地签名为准。
- 本目录统一使用 DeepSeek，Key 必须放在 `DEEPSEEK_API_KEY`。

官方入口：

- [What are Teams?](https://docs.agno.com/teams/overview.md)
- [Building Teams](https://docs.agno.com/teams/building-teams.md)
- [Delegation](https://docs.agno.com/teams/delegation.md)
- [Running Teams](https://docs.agno.com/teams/running-teams.md)
- [Debugging Teams](https://docs.agno.com/teams/debugging-teams.md)

## 核心模型

```text
User
  |
  v
Team Leader
  |-- 选择成员或子 Team
  |-- 为成员构造任务
  |-- 调用自己的工具
  |-- 协调成员执行
  `-- 汇总最终响应（Route 模式除外）
        |
        +-- Agent Member -> Tools
        +-- Agent Member -> Tools
        `-- Nested Team   -> Members
```

成员的 `name` 用于显示，`role` 是 Leader 选择成员的重要依据。官方还建议为成员设置稳定的
`id`，因为委派和 run tracking 使用成员 ID。

### Agent、Team 和 Workflow

| 组件 | 适用场景 | 控制方式 |
| --- | --- | --- |
| Agent | 单一职责、单组工具可以完成任务 | 一个模型自主调用工具 |
| Team | 需要动态选择专长成员、并行收集观点或由 Leader 综合 | 模型驱动的委派 |
| Workflow | 必须保证固定步骤、分支、重试和数据依赖 | 代码定义的确定性流程 |

Team 的自然语言指令可以要求“先研究、再分析”，但这不是确定性编排。业务上必须保证
`采集 -> 清洗 -> 分析` 时，应使用 Workflow；需要模型自主拆解依赖任务时，可学习
`TeamMode.tasks`。

## 官方四种 TeamMode

当前官方模式是 `coordinate`、`route`、`broadcast` 和 `tasks`，没有 `collaborate` 模式。
推荐显式使用 `TeamMode`：

```python
from agno.team import Team
from agno.team.mode import TeamMode

team = Team(
    members=[...],
    mode=TeamMode.coordinate,
)
```

| 模式 | 行为 | Leader 最终综合 | 适用场景 |
| --- | --- | --- | --- |
| `coordinate` | Leader 选择成员、构造不同任务并协调执行 | 有 | 研究、分析、写作等需要分解和质检的任务 |
| `route` | Leader 选择一个成员并直接返回该成员结果 | 无 | 语言路由、分类处理、低延迟专长分发 |
| `broadcast` | 把同一任务交给全部成员，再综合多方结果 | 有 | 多来源研究、多个观点、候选方案比较 |
| `tasks` | Leader 建立共享任务列表并迭代执行 | 有 | 需要自主拆解、任务依赖和多轮推进的目标 |

Broadcast 要获得成员并发执行，应使用 `arun()` 或 `aprint_response()`。同步
`print_response()` 不能作为“并行执行”的依据。

### 旧参数映射

旧布尔参数仍兼容，但官方推荐 `mode=`，且显式 `mode` 会覆盖旧参数：

| 旧写法 | 对应模式或作用 |
| --- | --- |
| 默认两个开关均为 `False` | `TeamMode.coordinate` |
| `respond_directly=True` | `TeamMode.route` |
| `delegate_to_all_members=True` | `TeamMode.broadcast` |
| `determine_input_for_members=False` | 直接把用户输入传给成员，不是模式选择器 |

## 环境准备

```bash
uv sync --extra dev
```

`.env` 至少配置：

```dotenv
DEEPSEEK_API_KEY=your-deepseek-key
DEEPSEEK_MODEL_ID=deepseek-v4-flash
TAVILY_API_KEY=your-tavily-key
```

注意：

- `03_math.py` 只需要 DeepSeek Key。
- 其他示例还需要 Tavily Key。
- `02_team_member.py` 还会访问 Yahoo Finance，但不需要 Yahoo API Key。
- 不要把真实 `OPENAI_API_KEY` 作为 DeepSeek Key。Agno 的 `DeepSeek` 默认请求
  `https://api.deepseek.com`，错误的凭证只会导致认证或连接失败。

## 示例索引

| 文件 | 官方模式 | 学习重点 | 外部依赖 |
| --- | --- | --- | --- |
| [`01_basic_team.py`](./01_basic_team.py) | Coordinate | 自定义 Team/Agent 事件流，展示委派、工具参数、工具结果和错误 | DeepSeek、Tavily |
| [`02_team_member.py`](./02_team_member.py) | Coordinate | 四个固定角色协作，YFinance 工具启用，成员交互共享 | DeepSeek、Tavily、Yahoo Finance |
| [`03_math.py`](./03_math.py) | Route | 把一个数学问题路由给一个专长成员并直接返回 | DeepSeek |
| [`04_collaborate.py`](./04_collaborate.py) | Broadcast | 同一研究任务广播给三类研究员，异步收集后综合 | DeepSeek、Tavily |
| [`05_coordinate.py`](./05_coordinate.py) | Coordinate | Leader 自主委派数据收集、清洗和分析 | DeepSeek、Tavily |
| [`06_p1.py`](./06_p1.py) | Coordinate | 入门研究 Team：研究员与总结员 | DeepSeek、Tavily |
| [`07_p2.py`](./07_p2.py) | Coordinate | 内容研究、撰写与审校的模型驱动协作 | DeepSeek、Tavily |

文件名 `04_collaborate.py` 为已有学习文件名；当前官方概念和代码均使用
`TeamMode.broadcast`。

### 推荐学习顺序

1. `06_p1.py`：先理解默认 Coordinate 的 Leader 和成员。
2. `03_math.py`：比较 Route 为什么只执行一个成员且没有 Leader 综合。
3. `04_collaborate.py`：学习 Broadcast 和异步成员并发。
4. `05_coordinate.py`：理解 Coordinate 的动态委派不是 Workflow。
5. `02_team_member.py`：学习更完整的角色、工具和成员交互。
6. `01_basic_team.py`：进入事件流和可观测性。
7. `07_p2.py`：观察多阶段内容生产的质量与成本。

### 运行命令

```bash
uv run python examples/05_teams/01_basic_team.py
uv run python examples/05_teams/02_team_member.py
uv run python examples/05_teams/03_math.py
uv run python examples/05_teams/04_collaborate.py
uv run python examples/05_teams/05_coordinate.py
uv run python examples/05_teams/06_p1.py
uv run python examples/05_teams/07_p2.py
```

这些示例会产生多次模型和工具请求。Coordinate Team 通常至少包含 Leader 的任务分析、
成员执行和 Leader 汇总，因此比单 Agent 更慢、token 成本更高。

## 关键参数

| 参数 | 作用 | 容易混淆的点 |
| --- | --- | --- |
| `members` | Agent、子 Team，或运行时成员工厂 | Team 可以嵌套 Team |
| `mode` | 选择官方协调模式 | 优先级高于旧布尔开关 |
| `role` | 向 Leader 描述成员专长 | 角色重叠会导致选错成员 |
| `instructions` | 约束 Leader 或成员行为 | 能引导顺序，但不能提供确定性保证 |
| `determine_input_for_members` | 是否由 Leader 重写成员任务 | `False` 不等于 Route |
| `share_member_interactions` | 把当前 run 已发生的成员交互给后续成员 | 不保证成员执行顺序 |
| `stream` | 流式返回正文 | 程序化使用时返回事件迭代器 |
| `stream_events` | 额外返回工具、错误、推理、hook 等事件 | 只开 `stream` 不等于获得全部事件 |
| `stream_member_events` | 是否把成员事件冒泡到 Team 流 | 异步时事件按到达顺序出现 |
| `show_members_responses` | Team 上设置成员展示的默认值 | 主要服务开发期终端展示 |
| `show_member_responses` | `print_response()` 的单次展示参数 | 注意这里是单数 `member` |
| `store_member_responses` | 持久化 Team run 时保留成员 runs | 与终端展示不是同一功能 |
| `max_iterations` | Tasks 模式最大任务循环次数 | 防止任务循环失控 |
| `debug_mode` / `debug_level` | 输出请求、工具、委派和指标日志 | 详细日志可能包含敏感上下文 |

## Streaming 与事件

开发期只想看格式化结果：

```python
team.print_response(
    "研究 AI Agent 的最新进展",
    stream=True,
    show_member_responses=True,
)
```

程序需要处理工具调用、成员结果和错误时，应使用事件流：

```python
from agno.run.team import TeamRunEvent

for event in team.run(
    "研究 AI Agent 的最新进展",
    stream=True,
    stream_events=True,
):
    if event.event == TeamRunEvent.run_content.value:
        print(event.content, end="", flush=True)
    elif event.event == TeamRunEvent.tool_call_started.value:
        print("Team tool started")
    elif event.event == TeamRunEvent.run_error.value:
        print(f"Team failed: {event.content}")
```

`01_basic_team.py` 进一步区分 Team 事件和成员 Agent 事件，并输出 Tavily 原始结果。它适合
学习可观测性，不再是本目录最简单的 Team 示例。

事件流展示的是公开的任务、工具输入输出和响应，不应把它描述为模型私有思维链。
DeepSeek 是否启用 provider thinking 由模型参数控制；提示词中的 `/nothink` 不是 Agno SDK
开关，本目录统一使用 `use_thinking=False`。

## 三类历史与共享

以下参数解决的问题不同：

| 能力 | 范围 | 典型配置 |
| --- | --- | --- |
| 成员交互共享 | 当前一次 Team run | `share_member_interactions=True` |
| Team 历史给成员 | 同一 session 的多次 run | `add_team_history_to_members=True` |
| 成员自身聊天历史 | 某个成员自己的历史 | 成员 `add_history_to_context=True` |

跨 run 历史需要配置数据库并保持稳定的 `session_id`。`user_id` 表示用户归属，不能替代
`session_id`。

还要区分：

- Session history：某个会话中的消息和运行历史。
- Session state：显式业务状态，可在同一会话中读取和更新。
- Memory：跨 session 保留的用户事实，需要 MemoryManager 和数据库。
- Knowledge：通过检索获得的外部资料；Team 的 knowledge 默认供 Leader 使用，成员需要时
  应单独配置或由 Leader 传递结果。

官方示例：

- [Share Member Interactions](https://docs.agno.com/examples/teams/basics/share-member-interactions.md)
- [Team History](https://docs.agno.com/examples/teams/basics/team-history.md)
- [History of Members](https://docs.agno.com/examples/teams/basics/history-of-members.md)
- [Persistent Team Session](https://docs.agno.com/examples/teams/session/persistent-session.md)
- [Team State Sharing](https://docs.agno.com/examples/teams/state/state-sharing.md)
- [Team Memories in Context](https://docs.agno.com/examples/teams/memory/memories-in-context.md)

## 本目录已校准的官方差异

| 原有问题 | 官方语义 | 当前处理 |
| --- | --- | --- |
| README 写成三种模式并使用 `collaborate` | 官方是四种 TeamMode，协作广播叫 `broadcast` | README 和示例统一为官方名称 |
| `03_math.py` 的 Route 输入要求先加再乘 | Route 只选择一个成员并直接返回 | 输入改为单一运算 |
| `04_collaborate.py` 实际设置为 Route | 多来源同题研究应使用 Broadcast | 改为 `TeamMode.broadcast` 和异步执行 |
| `05_coordinate.py` 设置 `delegate_to_all_members=True` | 该开关映射到 Broadcast | 改为显式 Coordinate，并删除确定性流水线表述 |
| 多处使用 `/nothink` | 只是普通提示文本 | 删除，保留模型 `use_thinking=False` |
| DeepSeek 示例回退读取 `OPENAI_API_KEY` | DeepSeek 有自己的 Key 和 API 地址 | 只读取 `DEEPSEEK_API_KEY` |
| YFinance 用 `include_tools` 启用默认关闭函数 | `include_tools` 只过滤已注册工具 | 使用 `enable_company_info` 等显式开关 |

### 仍需注意的限制

- `02`、`05`、`06`、`07` 中的多阶段顺序由 Leader 决定，不是 Workflow 保证。
- 搜索结果和股票信息来自外部服务，可能超时、限流或返回空数据。
- 2026 等时效性问题必须核对来源日期；模型总结不能替代来源验证。
- `show_member_responses=True` 适合开发调试，生产环境应使用结构化事件、tracing 和指标。

## 尚缺知识点与后续练习

本轮不新增示例文件，建议按以下优先级继续补充：

| 优先级 | 建议示例 | 学习内容 | 官方资料 |
| --- | --- | --- | --- |
| P0 | `08_tasks_mode.py` | Tasks 模式、依赖任务、`max_iterations` | [Tasks Dependencies](https://docs.agno.com/examples/teams/modes/tasks/dependencies.md) |
| P0 | Team 与 Workflow 对比 | 自适应委派和确定性流水线的边界 | [What are Workflows?](https://docs.agno.com/workflows/overview.md) |
| P1 | `09_nested_team.py` | 子 Team、模型继承、稳定成员 ID | [Nested Teams](https://docs.agno.com/examples/teams/basics/nested-teams.md) |
| P1 | `10_team_storage.py` | DB、session、history、state、member run 持久化 | [Persistent Session](https://docs.agno.com/examples/teams/session/persistent-session.md) |
| P1 | `11_structured_output.py` | Pydantic 输入输出、`output_schema` | [Structured Team I/O](https://docs.agno.com/examples/teams/structured-input-output/overview.md) |
| P1 | `12_async_team.py` | `arun()`、异步事件、取消和后台执行 | [Concurrent Members](https://docs.agno.com/examples/teams/basics/concurrent-member-agents.md) |
| P2 | Team Knowledge/Reasoning | Leader 检索、Team reasoning 与模型 thinking 的区别 | [Team Knowledge](https://docs.agno.com/examples/teams/knowledge/team-with-knowledge.md) |
| P2 | HITL 与 Guardrails | 确认、用户输入、暂停继续、输入防护、hooks | [Team HITL](https://docs.agno.com/examples/teams/human-in-the-loop/overview.md) |
| P2 | `13_agentos.py` | AgentOS Studio、Team 服务化与 tracing | [Studio Teams](https://docs.agno.com/agent-os/studio/teams.md) |

其他值得补充的生产能力：fallback models、retries、tool hooks、pre/post hooks、run metrics、
token 成本统计、run cancellation、callable member factories 和多租户 session 设计。

## 常见问题

### 运行后长时间没有输出

Team 通常要经过 Leader 委派、成员工具调用和最终综合。使用 `stream=True`；需要完整过程时
再使用 `stream_events=True`。同时为模型设置合理的超时和重试次数。

### 为什么只执行了一个成员

先检查是否使用了 `TeamMode.route`。Coordinate 下还要检查成员 `role` 是否清晰、Team 指令
是否要求使用其他成员，以及成员是否提前失败。

### 为什么看不到成员响应

`print_response()` 需要 `show_member_responses=True`。程序化处理请读取
`TeamRunOutput.member_responses` 或消费成员事件。

### YFinance 为什么提示工具不存在

`YFinanceTools` 默认只启用当前股价。公司信息和分析师建议需要显式注册：

```python
YFinanceTools(
    enable_stock_price=True,
    enable_company_info=True,
    enable_analyst_recommendations=True,
)
```

### 使用 DeepSeek 为什么日志里出现 OpenAI API

Agno 的 DeepSeek 实现基于 OpenAI-compatible transport，因此底层错误日志可能包含 OpenAI
API 字样。应检查实际模型对象、`base_url` 和 `DEEPSEEK_API_KEY`，不能只根据日志标题判断
请求了哪家模型。

## 官方延伸阅读

- [Team Reference](https://docs.agno.com/reference/teams/team.md)
- [TeamRunOutput](https://docs.agno.com/reference/teams/team-response.md)
- [Team Events](https://docs.agno.com/examples/teams/streaming/team-events.md)
- [Structured Team Input/Output](https://docs.agno.com/examples/teams/structured-input-output/overview.md)
- [Team Knowledge](https://docs.agno.com/examples/teams/knowledge/team-with-knowledge.md)
- [Team Reasoning](https://docs.agno.com/examples/teams/reasoning/reasoning-multi-purpose-team.md)
- [Team Guardrails](https://docs.agno.com/examples/teams/guardrails/overview.md)
- [Basic Team Tracing](https://docs.agno.com/agent-os/tracing/usage/basic-team-tracing.md)
- [Full Documentation Index](https://docs.agno.com/llms.txt)

# 03 Storage And Memory

本目录通过 10 个示例学习 Agno 的会话存储、聊天历史、用户长期记忆、`MemoryManager`、Agentic Memory、记忆 CRUD 和跨 Agent 记忆筛选。

> 本仓库的 `uv.lock` 当前锁定 Agno `2.7.3`。本文同时参考了 2026-07-15 的 Agno 在线文档；在线文档可能对应更新版本，涉及 AgentOS `user_isolation` 的内容不能直接套用到 `2.7.3`。

## 核心概念

Agno 当前官方文档首先区分 **Session History** 和 **Memory**，而不是把本目录内容划分为“短期、长期、语义”三类。

| 概念 | 保存什么 | 默认位置 | 如何使用 |
| --- | --- | --- | --- |
| Session Storage | 完整 run、消息、响应、状态和元数据 | `agno_sessions` | 给 Agent 配置 `db` 后自动持久化 |
| Chat History | 从已存 session 中取出的历史消息 | `agno_sessions` | `add_history_to_context=True` 才会放回模型上下文 |
| User Memory | 从对话中提取的用户事实、偏好、目标或长期问题 | `agno_memories` | 使用 `update_memory_on_run=True` 或 `enable_agentic_memory=True` |
| MemoryManager | 决定提取、更新和读取哪些 memory，可使用独立模型和规则 | 不单独建表 | 通过 `memory_manager=MemoryManager(...)` 注入 Agent |

### Memory 不等于历史消息

```text
Session History：用户刚才具体说了什么？
User Memory：系统从历史交互中学到了哪些可复用事实？
```

给 Agent 配置数据库只代表 session 会被保存，不代表旧消息会自动发送给模型。要让模型看到同一会话的历史，需要同时满足：

1. 使用稳定的 `session_id`。
2. 使用正确且稳定的 `user_id`。
3. 设置 `add_history_to_context=True`。
4. 使用 `num_history_runs` 或 `num_history_messages` 控制加入上下文的范围。

长期记忆则是模型选择性提取的摘要，不保证每条输入都生成 memory，也不适合代替订单、医疗记录或其他权威业务数据。

## 两种记忆模式

| 模式 | 配置 | 谁决定写入 | 特点 |
| --- | --- | --- | --- |
| 自动记忆 | `update_memory_on_run=True` | MemoryManager 在每轮 run 中分析输入 | 官方推荐用于大多数场景，行为相对可预测 |
| Agentic Memory | `enable_agentic_memory=True` | 主 Agent 决定是否调用 `update_user_memory` 工具 | 更灵活，但更依赖模型决策和工具调用 |

不要同时开启两种模式。Agno 官方文档说明：当两者同时为 `True` 时，Agentic Memory 优先，自动更新会被忽略。

`enable_user_memories=True` 是当前 `2.7.3` 仍可使用的兼容别名，内部会映射到 `update_memory_on_run=True`，但已标记为即将弃用。新代码应使用：

```python
agent = Agent(
    db=db,
    update_memory_on_run=True,
)
```

两种模式都会增加模型调用和 token 消耗。自动模式通常每轮增加一次 MemoryManager 调用；Agentic 模式在 Agent 调用记忆工具时还会触发嵌套的记忆模型调用。用户记忆不断增长时，加载到上下文中的 token 也会持续增加。

## 运行准备

在仓库根目录执行：

```bash
uv sync
```

项目要求 Python 3.12 或更高版本。在仓库根目录创建或修改 `.env`：

```dotenv
DEEPSEEK_API_KEY=你的_DeepSeek_API_Key
# 可选；不设置时示例默认使用 deepseek-v4-flash
DEEPSEEK_MODEL_ID=deepseek-v4-flash
```

示例代码虽然会回退读取 `OPENAI_API_KEY`，但创建的始终是 `agno.models.deepseek.DeepSeek`。真实的 OpenAI API Key 不能用于 DeepSeek，推荐始终配置 `DEEPSEEK_API_KEY`。

所有命令都应从仓库根目录运行，因此相对路径 `my_agent.db` 对应根目录下的数据库文件。`.env` 和 `*.db` 已被仓库的 `.gitignore` 忽略。

## 示例索引

| 文件 | 模式 | 学习重点 | Memory 表 |
| --- | --- | --- | --- |
| `01_auto_memory.py` | 自动记忆（旧别名） | 基础记忆、回忆和 topics | `agno_memories` |
| `02_multi_user.py` | 自动记忆 + Chat History | 多用户、稳定 session、记忆与历史配合 | `customer_memories` |
| `03_manual.py` | 自动记忆（旧别名） | 查询 memory 元数据和删除意图 | `agno_memories` |
| `04_self_mem.py` | 自定义 MemoryManager | 提取规则、隐私提示和 debug | `agno_memories` |
| `05_agentic_mem.py` | Agentic Memory | 由 Agent 判断创建和更新记忆 | `agno_memories` |
| `06_curd_mem.py` | 自动记忆（旧别名） | Create、Read、Update、Delete | `agno_memories` |
| `07_safe_multi_person.py` | 自动记忆 + 自定义读取 | 使用 `(agent_id, user_id)` 筛选共享表 | `agno_memories` |
| `08_p1.py` | 自动记忆（旧别名） | 电商客服案例 | `agno_memories` |
| `09_p2.py` | 自动记忆（旧别名） | 健康管理案例 | `agno_memories` |
| `10_p3.py` | 自动记忆（旧别名） | 学习辅导案例 | `agno_memories` |

## 01：自动记忆与 Topics

`01_auto_memory.py` 先让 Agent 记住李明的咖啡和爬山偏好，再进行回忆和活动推荐；随后切换到张三，打印提取出的 memory 和 topics。

```bash
uv run python examples/03_storage_memory/01_auto_memory.py
```

注意事项：

- 使用了即将弃用的 `enable_user_memories=True`。
- 同一个 Agent 从 `user_001` 切换到 `user_zhangsan`，但没有显式传入不同的 `session_id`。Agno 的自动 session ID 会粘在 Agent 实例上，因此第二个用户的 session 持久化不可靠。
- 长期 memory 仍按 `user_id` 提取和保存，但这不能替代正确的多用户 session 设计。

## 02：多用户与独立会话

`02_multi_user.py` 是本目录中更完整的多用户示例：

- 使用当前 API `update_memory_on_run=True`。
- 使用 `add_history_to_context=True`，最多加载最近 3 个 run。
- 客户 A、B 使用稳定且不同的 `session_id`。
- 使用自定义记忆表 `customer_memories`，session 仍保存在 `agno_sessions`。

```bash
uv run python examples/03_storage_memory/02_multi_user.py
```

具体咨询主题不一定会被 MemoryManager 判断为长期记忆，但相同客户的第二次请求仍可通过独立的 Chat History 看到前一次消息。重复执行会复用固定 session，并继续累积历史。

## 03：查询记忆元数据

`03_manual.py` 将模拟用户资料作为 prompt，由自动记忆提取，然后通过 `get_user_memories()` 查看：

- `memory_id`
- `memory`
- `topics`
- `updated_at`

```bash
uv run python examples/03_storage_memory/03_manual.py
```

文件名虽然是 `manual`，当前实现并不是直接手动插入 memory。删除代码也处于注释状态，而且示例中的 `agent.delete_user_memory(...)` 不是 Agno `2.7.3` 的 Agent 方法；需要删除时应使用数据库接口，例如 `agent.db.delete_user_memory(...)`，并明确提供 `memory_id` 和 `user_id`。

## 04：自定义 MemoryManager

`04_self_mem.py` 通过 `memory_capture_instructions` 和 `additional_instructions` 自定义提取规则，并尝试阻止身份证、电话号码、密码和财务信息进入 memory 摘要。

```bash
uv run python examples/03_storage_memory/04_self_mem.py
```

这个示例只展示 prompt 级隐私规则，不是数据防泄漏系统：

- 原始敏感文本仍会发送给主模型和记忆模型。
- 触发 memory 的原始输入会进入 `agno_memories.input`。
- `debug_mode=True` 可能在日志中暴露处理细节。
- 代码只能在生成后检查是否泄漏，无法在模型调用前强制阻断。

只能使用虚构数据测试，不能把它直接用于真实个人敏感信息。

## 05：Agentic Memory

`05_agentic_mem.py` 使用 `enable_agentic_memory=True`，让主 Agent 获得 `update_user_memory` 工具。示例依次测试固定会议、临时天气、回忆会议和更新会议时间。

```bash
uv run python examples/03_storage_memory/05_agentic_mem.py
```

Agentic Memory 不会像自动模式一样无条件启动后台提取；主模型必须先决定调用工具。默认 MemoryManager 主要支持添加和更新，本示例描述中的“删除无用记忆”并未显式开启删除、清空能力。

## 06：记忆 CRUD

文件名是 `06_curd_mem.py`，运行命令必须保持现有拼写：

```bash
uv run python examples/03_storage_memory/06_curd_mem.py
```

实际流程：

1. Create：通过用户消息让自动 MemoryManager 创建 memory。
2. Read：通过 `get_user_memories()` 查询。
3. Update：通过新的用户消息让模型决定更新或新增。
4. Delete：通过 `agent.db.delete_user_memory()` 删除指定 memory。
5. Clear：只展示被注释的危险操作。

代码注释写的是“Agentic 记忆模式”，但实际使用 `enable_user_memories=True`，属于自动模式。模型可能合并、更新或新增 memory，因此 Update 后的条数并不固定。不要随意启用 `memory_manager.clear()`，它会清空整张记忆表，而不是只清理当前用户。

## 07：按 Agent 筛选共享记忆

Agno 官方文档明确说明：多个 Agent 连接同一个数据库并使用相同 `user_id` 时，默认会共享用户记忆。`07_safe_multi_person.py` 则演示如何选择退出这种默认行为。

```bash
uv run python examples/03_storage_memory/07_safe_multi_person.py
```

两个 Agent 仍然共用：

```text
数据库：my_agent.db
Memory 表：agno_memories
user_id：customer_001
```

自定义 `AgentScopedMemoryManager.read_from_db()` 会强制调用：

```python
db.get_user_memories(
    user_id=user_id,
    agent_id=self.agent_id,
)
```

因此企业 B 的同步 MemoryManager 不会读取企业 A 写入的 memory。

这个示例的边界必须明确：

- 它只覆盖当前同步用户记忆读取路径，不是完整的多租户授权系统。
- 直接调用不带 `agent_id` 的数据库接口仍可绕过筛选。
- Session、Knowledge、删除、导出和异步 `aread_from_db()` 没有统一加上相同作用域。
- `agent_id` 必须由服务端生成并保持稳定，不能信任模型或客户端自报。
- 同一家企业的多个 Agent 也会互相隔离；若需要企业内部共享，应使用独立且可信的 tenant 作用域。

## 08：电商客服案例

`08_p1.py` 模拟选购、下单、追加购买和售后咨询，最后打印客户 memory 档案。

```bash
uv run python examples/03_storage_memory/08_p1.py
```

当前代码存在一个会影响结果的拼写错误：首次请求使用 `customer_limig`，后续使用 `customer_liming`。这两个值会被当成不同用户，因此首次记录的姓名、预算和用途不会自动出现在后续客户档案中。此外，示例没有接入真实订单或物流系统，所有订单事实都只是用户陈述。

## 09：健康管理案例

`09_p2.py` 模拟目标设定、饮食运动、体重变化、瓶颈和阶段回顾，并按 memory topics 生成健康档案。

```bash
uv run python examples/03_storage_memory/09_p2.py
```

“第 1 周”到“第 8 周”只是同一次脚本执行中的文字标签，不代表真实时间、定时任务或数据采集。健康信息会发送给模型并持久化到 SQLite；prompt 中的“不是医生”也不是医疗合规或诊断安全机制。该示例不能作为真实健康档案或医疗建议系统。

## 10：学习辅导案例

`10_p3.py` 模拟学生的目标、薄弱点、学习反馈和考试进展，最后使用中文关键词对 memory 进行分类。

```bash
uv run python examples/03_storage_memory/10_p3.py
```

分类逻辑只是简单的字符串包含判断，不是语义分类器；当前 `强项` 分类也没有对应写入分支。时间阶段同样是文字模拟，生成建议和成绩信息没有连接教务系统核验。真实应用还必须保护学生及可能涉及未成年人的数据。

## 查看 SQLite 数据

列出当前数据库中的表：

```bash
sqlite3 -readonly my_agent.db '.tables'
```

查看默认用户记忆：

```bash
sqlite3 -readonly -header -column my_agent.db \
  "SELECT memory_id, user_id, agent_id, json_extract(memory, '$') AS memory, topics
   FROM agno_memories
   ORDER BY created_at;"
```

查看 `02_multi_user.py` 的自定义记忆表：

```bash
sqlite3 -readonly -header -column my_agent.db \
  "SELECT memory_id, user_id, agent_id, json_extract(memory, '$') AS memory
   FROM customer_memories
   ORDER BY created_at;"
```

查看 session 所属关系：

```bash
sqlite3 -readonly -header -column my_agent.db \
  "SELECT session_id, user_id, agent_id, created_at, updated_at
   FROM agno_sessions
   ORDER BY created_at;"
```

这些示例共享 `my_agent.db`。重复执行会沿用以前的 session 和 memory，因此结果可能受旧数据影响。需要独立实验时，优先修改脚本使用新的 `db_file`，不要把一次运行的结果误认为空数据库行为。

## 多用户与数据隔离

### 默认行为

官方文档把 User Memory 设计为可在多个 Agent 之间共享：只要连接同一个数据库并使用相同 `user_id`，不同 Agent 默认可以读取同一用户的 memory。Memory 行虽然包含 `agent_id`，默认 MemoryManager 在当前 `2.7.3` 中读取时只按 `user_id` 查询。

因此：

```text
不同 Agent 实例 != 自动的数据隔离
记录 agent_id != 默认按 agent_id 过滤
API 鉴权 != 数据行隔离
```

### 最新 AgentOS 在线文档

最新在线文档提供 `AuthorizationConfig(user_isolation=True)`，使用 JWT 的 `sub` 作为非管理员调用者的 `user_id`，并对 session、memory 和 trace 的读写进行用户级约束。

```python
AuthorizationConfig(
    verification_keys=["你的 JWT 验证公钥"],
    algorithm="RS256",
    user_isolation=True,
)
```

但当前项目安装的 Agno `2.7.3` 中没有 `user_isolation` 配置。使用这项能力前必须升级 Agno、检查升级说明并重新运行隔离测试。它提供的是按认证用户隔离；企业多租户系统仍需从可信认证上下文确定 tenant 与 user 的映射。

## 常见问题

### Agent 记不住上一句话

依次检查：

1. 是否配置了 `db`。
2. 前后两次是否使用完全相同的 `user_id`。
3. 是否使用同一个稳定 `session_id`。
4. 如果需要原始对话，是否启用了 `add_history_to_context=True`。
5. 如果依赖长期 memory，该输入是否真的满足 MemoryManager 的提取标准。

### 不同用户或 Agent 看到彼此内容

首先检查是否遗漏 `user_id`。未提供时会落到默认用户，多个调用可能共享同一批 memory。跨 Agent 共享同一数据库和同一 `user_id` 是 Agno 的默认设计；需要 Agent 级筛选时参考 `07_safe_multi_person.py`，生产多租户系统则应在统一数据访问层强制可信 tenant/user 作用域。

### 日志出现 `OpenAI API`

Agno 的 DeepSeek 适配器使用 OpenAI-compatible 客户端，因此底层异常可能出现 `OpenAI API` 字样。这不代表代码自动切换到了 OpenAI；请求模型仍由 `DeepSeek(...)` 配置决定。

### Memory 数量和输出每次不同

Memory 的创建、合并、更新和 topics 生成都由模型参与，结果不是确定性的。重复执行还会读取之前的数据库内容。`topics` 也可能为空，生产代码不应无条件执行 `for topic in memory.topics` 或 `join(memory.topics)`。

### 导入脚本时自动调用模型

除 `07_safe_multi_person.py` 外，其余示例都在模块顶层运行对话。导入这些文件也会立即检查 API Key、调用 DeepSeek 并可能修改数据库，不适合直接作为无副作用模块复用。

## 当前代码限制

- `01`、`03` 到 `06`、`08` 到 `10` 仍使用旧参数 `enable_user_memories=True`。
- `01`、`03` 到 `06`、`08` 到 `10` 存在重复导入或未使用导入；当前 Ruff 共报告 21 个相关问题，但 10 个脚本均可通过 Python 语法检查。
- `08_p1.py` 的 `customer_limig/customer_liming` 不一致会直接破坏连续记忆效果。
- `04`、`09`、`10` 涉及隐私、健康或学生数据，只适合虚构数据教学。
- 记忆内容是模型生成的用户事实摘要，不应作为认证、授权、订单、付款、医疗或成绩系统的事实来源。

## 官方资料

- [What is Memory?](https://docs.agno.com/memory/overview.md)
- [Agent Memory](https://docs.agno.com/memory/agent/overview.md)
- [Working with Memories](https://docs.agno.com/memory/working-with-memories/overview.md)
- [Memory Production Best Practices](https://docs.agno.com/memory/best-practices.md)
- [Chat History](https://docs.agno.com/history/overview.md)
- [Session Storage](https://docs.agno.com/database/session-storage.md)
- [SQLite](https://docs.agno.com/database/providers/sqlite/overview.md)
- [AgentOS Per-User Data Isolation](https://docs.agno.com/agent-os/security/authorization/user-isolation.md)

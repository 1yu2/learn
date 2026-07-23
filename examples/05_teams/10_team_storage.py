"""
团队持久化存储（Team Storage）

演示如何将团队的会话（session）数据持久化到 PostgreSQL 数据库中。
支持设置是否注入历史上下文，让团队能"记住"之前的对话。

架构图:
┌───────────────────────────────────────────────────────────┐
│                    Team with DB Storage                    │
│                                                           │
│         ┌──────────────────┐    ┌──────────────────┐      │
│         │   basic_team     │    │   history_team    │      │
│         │ (无历史上下文)    │    │ (注入历史上下文)   │      │
│         │                  │    │                  │      │
│         │ Agent → DB      │    │ Agent → DB      │      │
│         │ (不注入历史)     │    │ (注入最近3轮历史)  │      │
│         └────────┬─────────┘    └────────┬─────────┘      │
│                  │                       │                │
│                  └───────────┬───────────┘                │
│                              │                            │
│                              ▼                            │
│               ┌────────────────────────┐                  │
│               │  PostgreSQL (PostgresDb)│                 │
│               │  sessions 表           │                  │
│               │  omni:omnipass@:15432  │                  │
│               └────────────────────────┘                  │
└───────────────────────────────────────────────────────────┘

运行命令: python examples/05_teams/10_team_storage.py
"""

from os import getenv                           # 获取环境变量

from agno.agent import Agent                    # Agent（智能体）基类
from agno.team import Team                      # 团队类
from agno.db.postgres import PostgresDb         # PostgreSQL 数据库适配器
from agno.models.deepseek import DeepSeek       # DeepSeek 模型接口
from dotenv import load_dotenv                  # 加载 .env 环境变量
from httpx import Timeout                       # HTTP 超时配置


def create_deepseek_model(api_key: str) -> DeepSeek:
    """创建一个 DeepSeek 模型实例"""
    return DeepSeek(
        id=getenv("DEEPSEEK_MODEL_ID", "deepseek-v4-flash"),  # 模型 ID，默认 deepseek-v4-flash
        api_key=api_key,
        use_thinking=False,                     # 不启用思维链
        timeout=Timeout(
            connect=15.0, read=180.0, write=60.0, pool=60.0,  # 超时配置：连接/读取/写入/连接池
        ),
        max_retries=1,                          # 最大重试次数
    )


# 加载 .env 文件中的环境变量
load_dotenv()

# 获取 DeepSeek API Key
api_key = getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError("缺少 DeepSeek API Key，请设置 DEEPSEEK_API_KEY。")


# ---------------------------------------------------------------------------
# 数据库配置
# ---------------------------------------------------------------------------

# PostgreSQL 连接 URL（使用 psycopg 3 驱动）
# 格式: postgresql+psycopg://用户名:密码@主机:端口/数据库名
db_url = "postgresql+psycopg://omni:omnipass@localhost:15432/postgres"
db = PostgresDb(db_url=db_url, session_table="sessions")  # 使用 sessions 表存储会话

# ---------------------------------------------------------------------------
# 创建成员
# ---------------------------------------------------------------------------

agent = Agent(model=create_deepseek_model(api_key))

# ---------------------------------------------------------------------------
# 创建团队
# ---------------------------------------------------------------------------

# 基础团队 —— 不注入历史上下文，每次对话独立
basic_team = Team(
    model=create_deepseek_model(api_key),
    members=[agent],
    db=db,                                          # 绑定数据库，存储会话记录
)

# 历史团队 —— 注入历史上下文到提示词中，让 AI"记住"之前的对话
history_team = Team(
    model=create_deepseek_model(api_key),
    members=[agent],
    db=db,
    add_history_to_context=True,                    # 将历史记录注入上下文
    num_history_runs=3,                             # 保留最近 3 轮历史
)

# ---------------------------------------------------------------------------
# 运行团队
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # basic_team: 每次调用都是独立会话，不感知上下文
    basic_team.print_response("Tell me a new interesting fact about space")

    # history_team: 多次调用共享历史上下文
    # 第 1 次：询问太空
    history_team.print_response("Tell me a new interesting fact about space")
    # 第 2 次：询问海洋（模型能记住之前聊过太空）
    history_team.print_response("Tell me a new interesting fact about oceans")
    # 第 3 次：问"我们刚刚聊了什么？"（模型需要回顾历史来回答）
    history_team.print_response("What have we been talking about?")

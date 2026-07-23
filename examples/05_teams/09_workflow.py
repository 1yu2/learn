"""
工作流（Workflow）示例

演示如何使用 Agno 的 Workflow（工作流）功能，将多个 Agent 编排成一个有序的处理流水线。
工作流中的步骤会按顺序依次执行，前一步的输出会自动传递给后一步。

架构图:
┌──────────────────────────────────────────────────────┐
│                  Content Workflow                     │
│                                                      │
│  ┌────────────────────┐    ┌────────────────────┐   │
│  │  Step 1            │    │  Step 2            │   │
│  │  Researcher        │───→│  Writer            │   │
│  │  (搜集信息)         │    │  (撰写文章)         │   │
│  └────────────────────┘    └────────────────────┘   │
│           │                          │               │
│  Tools: HackerNewsTools              │               │
│                                      ▼               │
│                             输出: 最终文章            │
└──────────────────────────────────────────────────────┘

运行命令: python examples/05_teams/09_workflow.py
"""

from os import getenv                           # 获取环境变量

from agno.agent import Agent                    # Agent（智能体）基类
from agno.workflow import Workflow              # 工作流类
from agno.workflow.agent import WorkflowAgent   # 工作流 Agent（用于指定调度模型）
from agno.models.deepseek import DeepSeek       # DeepSeek 模型接口
from agno.tools.hackernews import HackerNewsTools  # HackerNews 工具
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


# ---- 步骤 1：调研员 ----
# 负责搜集与主题相关的信息，使用 HackerNewsTools 获取 HackerNews 上的最新内容
researcher = Agent(
    name="Researcher",
    model=create_deepseek_model(api_key),
    instructions="Find relevant information about the topic",  # 查找与主题相关的资料
    tools=[HackerNewsTools()],
)

# ---- 步骤 2：写手 ----
# 基于调研结果撰写清晰、有吸引力的文章
writer = Agent(
    name="Writer",
    model=create_deepseek_model(api_key),
    instructions="Write a clear, engaging article based on the research",  # 基于调研撰写清晰、引人入胜的文章
)

# ---- 创建工作流 ----
# 工作流使用 WorkflowAgent 来调度步骤的执行
# 工作流会按 steps 列表中的顺序依次执行每个 Agent
# researcher 的输出会自动传递给 writer 作为输入
content_workflow = Workflow(
    name="Content Creation",                                         # 工作流名称：内容创作
    agent=WorkflowAgent(model=create_deepseek_model(api_key)),      # 工作流调度 Agent（指定调度模型）
    steps=[researcher, writer],                                      # 步骤列表（按顺序执行）
)

# ---- 执行工作流 ----
if __name__ == "__main__":
    content_workflow.print_response(
        "Write an article about AI trends",      # 主题：撰写一篇关于 AI 趋势的文章
        stream=True,                             # 启用流式输出
    )

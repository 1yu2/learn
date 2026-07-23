"""
任务依赖示例（Task Dependencies Example）

演示任务模式（task mode）下的复杂任务依赖链。团队负责人创建任务时，
后续任务依赖于前面的任务，从而确保正确的执行顺序。
展示系统如何处理被阻塞的任务。

架构图:
┌───────────────────────────────────────────────────────┐
│              Product Launch Team                      │
│                                                       │
│  ┌──────────────┐    ┌──────────────────┐             │
│  │ ① Market     │    │ ② Product        │             │
│  │   Researcher  │───→│   Strategist     │             │
│  │  (无依赖)     │    │  (依赖 research) │             │
│  └──────────────┘    └────────┬─────────┘             │
│                               │                       │
│                               ▼                       │
│  ┌──────────────────┐    ┌──────────────────┐         │
│  │ ③ Content        │    │ ④ Launch         │         │
│  │   Creator         │───→│   Coordinator    │         │
│  │  (依赖 strategy)  │    │  (依赖全部)      │         │
│  └──────────────────┘    └──────────────────┘         │
│                                                       │
│  ──→ 表示 depends_on（任务依赖方向）                    │
└───────────────────────────────────────────────────────┘
"""

# 导入 Agno 相关模块
from agno.agent import Agent                  # Agent（智能体）基类
from agno.models.openai import OpenAIResponses  # OpenAI 模型接口
from agno.team.mode import TeamMode            # 团队模式枚举（包括 task 模式）
from agno.team.team import Team                # 团队（Team）类

from os import getenv

from agno.agent import Agent
from agno.models.deepseek import DeepSeek      # DeepSeek 模型接口
from agno.team import Team
from agno.team.mode import TeamMode
from agno.tools.tavily import TavilyTools
from dotenv import load_dotenv                 # 加载 .env 环境变量
from httpx import Timeout                      # HTTP 超时配置


def create_deepseek_model(api_key: str) -> DeepSeek:
    """创建一个 DeepSeek 模型实例"""
    return DeepSeek(
        id=getenv("DEEPSEEK_MODEL_ID", "deepseek-v4-flash"),  # 模型 ID，默认 deepseek-v4-flash
        api_key=api_key,
        use_thinking=False,                     # 不启用思维链
        timeout=Timeout(connect=15.0, read=180.0, write=60.0, pool=60.0),  # 超时配置：连接/读取/写入/连接池
        max_retries=1,                          # 最大重试次数
    )

# 加载 .env 文件中的环境变量
load_dotenv()

# 获取 DeepSeek API Key
api_key = getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError("缺少 DeepSeek API Key，请设置 DEEPSEEK_API_KEY。")


# ---------------------------------------------------------------------------
# 创建团队成员（Create Members）
# ---------------------------------------------------------------------------

# 市场研究员 Agent —— 进行市场调研和竞争分析
market_researcher = Agent(
    name="Market Researcher",
    role="Conducts market research and competitive analysis",  # 角色：进行市场研究和竞争分析
    model=create_deepseek_model(api_key),
    instructions=[
        "You are a market researcher.",
        "Analyze target markets, customer segments, and competitive landscape.",
        "Provide data-driven insights and recommendations.",
    ],
)

# 产品策略师 Agent —— 基于市场研究制定产品定位和策略
product_strategist = Agent(
    name="Product Strategist",
    role="Develops product positioning and go-to-market strategy",  # 角色：制定产品定位和上市策略
    model=create_deepseek_model(api_key),
    instructions=[
        "You are a product strategist.",
        "Based on market research, develop product positioning and strategy.",
        "Define value propositions, target segments, and differentiation.",
    ],
)

# 内容创作者 Agent —— 创建营销内容
content_creator = Agent(
    name="Content Creator",
    role="Creates marketing content and messaging",  # 角色：创建营销内容和信息传达
    model=create_deepseek_model(api_key),
    instructions=[
        "You are a content creator.",
        "Create compelling marketing copy based on the product strategy.",
        "Write headlines, taglines, and key messages.",
    ],
)

# 发布协调员 Agent —— 创建发布时间线和行动计划
launch_coordinator = Agent(
    name="Launch Coordinator",
    role="Creates launch timelines and action plans",  # 角色：创建发布时间线和执行计划
    model=OpenAIResponses(id="gpt-5-mini"),            # 使用 OpenAI 的 gpt-5-mini 模型
    instructions=[
        "You are a launch coordinator.",
        "Create detailed launch timelines with milestones.",
        "Coordinate all launch activities into a cohesive plan.",
    ],
)

# ---------------------------------------------------------------------------
# 创建团队（Create Team）
# ---------------------------------------------------------------------------

# 产品发布团队 —— 使用 Task 模式，成员之间有依赖关系
launch_team = Team(
    name="Product Launch Team",
    mode=TeamMode.tasks,                         # 设置为任务模式（会进行任务分解和依赖管理）
    model=create_deepseek_model(api_key),
    members=[
        market_researcher,                       # 1. 市场研究员（无依赖，优先执行）
        product_strategist,                      # 2. 产品策略师（依赖市场研究结果）
        content_creator,                         # 3. 内容创作者（依赖产品策略）
        launch_coordinator,                      # 4. 发布协调员（依赖前面所有成员）
    ],
    instructions=[
        "You are a product launch team leader.",
        "Create tasks with proper dependencies to form a pipeline:",  # 创建带有依赖关系的任务流水线
        "1. First: Market Researcher conducts research (no dependencies)",      # 第1步：市场调研（无依赖）
        "2. Then: Product Strategist develops strategy (depends on research)",   # 第2步：制定策略（依赖调研）
        "3. Then: Content Creator writes messaging (depends on strategy)",       # 第3步：撰写内容（依赖策略）
        "4. Finally: Launch Coordinator creates the launch plan (depends on all above)",  # 第4步：制定发布计划（依赖全部）
        "Use depends_on to enforce this ordering.",   # 使用 depends_on 强制排序
        "Execute the first task, then as each completes, execute the next in the chain.",  # 依次执行，完成后触发下一个
    ],
    show_members_responses=True,                 # 显示每个成员的回答
    markdown=True,                               # 以 Markdown 格式输出
    max_iterations=15,                           # 最大迭代次数
)

# ---------------------------------------------------------------------------
# 运行团队（Run Team）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 启动团队，规划一个全新的 AI 代码审查工具的产品发布
    launch_team.print_response(
        "Plan a product launch for a new AI-powered code review tool "
        "targeting mid-size software companies. The tool uses LLMs to "
        "provide automated code reviews with natural language explanations."
    )
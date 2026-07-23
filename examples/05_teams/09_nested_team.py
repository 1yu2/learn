"""
嵌套团队（Nested Teams）

演示如何将子团队作为成员嵌入到更高层级的协调团队中。
这种嵌套结构可以实现多层级的分工与协作。

架构图:
┌─────────────────────────────────────────────────┐
│           Program Team  (parent_team)            │
│  ┌─────────────────┐  ┌─────────────────┐       │
│  │  Research Team  │  │   Writing Team  │       │
│  │  ┌───────────┐  │  │  ┌───────────┐  │       │
│  │  │ Research  │  │  │  │  Writing   │  │       │
│  │  │  Agent    │  │  │  │  Agent     │  │       │
│  │  └───────────┘  │  │  └───────────┘  │       │
│  │  ┌───────────┐  │  │  ┌───────────┐  │       │
│  │  │ Analysis  │  │  │  │  Editing   │  │       │
│  │  │  Agent    │  │  │  │  Agent     │  │       │
│  │  └───────────┘  │  │  └───────────┘  │       │
│  └─────────────────┘  └─────────────────┘       │
│        ① 收集证据 ───────→ ② 综合撰写            │
└─────────────────────────────────────────────────┘

运行命令: python examples/05_teams/09_nested_team.py
"""

from os import getenv                           # 获取环境变量

from agno.agent import Agent                    # Agent（智能体）基类
from agno.team import Team                      # 团队类
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
# 创建子团队的成员 Agent
# ---------------------------------------------------------------------------
# 调研 Agent —— 负责收集参考资料和原始素材
research_agent = Agent(
    name="Research Agent",
    model=create_deepseek_model(api_key),
    role="Gather references and source material",  # 角色：收集参考资料和源素材
)

# 分析 Agent —— 负责提取关键发现和潜在影响
analysis_agent = Agent(
    name="Analysis Agent",
    model=create_deepseek_model(api_key),
    role="Extract key findings and implications",  # 角色：提取关键发现和影响
)

# 写作 Agent —— 负责起草精炼的叙述性输出
writing_agent = Agent(
    name="Writing Agent",
    model=create_deepseek_model(api_key),
    role="Draft polished narrative output",       # 角色：起草打磨后的叙述内容
)

# 编辑 Agent —— 负责改进内容的清晰度和结构
editing_agent = Agent(
    name="Editing Agent",
    model=create_deepseek_model(api_key),
    role="Improve clarity and structure",         # 角色：提升清晰度和结构
)

# ---- 子团队 1：调研团队 ----
# 由调研 Agent 和分析 Agent 组成，负责信息收集与分析
research_team = Team(
    name="Research Team",
    members=[research_agent, analysis_agent],
    model=create_deepseek_model(api_key),
    instructions=[
        "Collect relevant information and summarize evidence.",  # 收集相关信息并总结证据
        "Highlight key takeaways and uncertainties.",             # 突出关键结论和不确定因素
    ],
)

# ---- 子团队 2：写作团队 ----
# 由写作 Agent 和编辑 Agent 组成，负责内容起草和润色
writing_team = Team(
    name="Writing Team",
    members=[writing_agent, editing_agent],
    model=create_deepseek_model(api_key),
    instructions=[
        "Draft and refine final output from provided research.",  # 基于调研结果起草并完善最终输出
        "Keep language concise and decision-oriented.",            # 保持语言简洁、以决策为导向
    ],
)

# ---------------------------------------------------------------------------
# 创建父团队（协调层）
# ---------------------------------------------------------------------------
# 父团队将两个子团队作为成员，协调它们的协作
# 工作流程：先让 Research Team 搜集证据，再让 Writing Team 进行综合撰写
parent_team = Team(
    name="Program Team",
    members=[research_team, writing_team],        # 成员是两个子团队
    model=create_deepseek_model(api_key),
    instructions=[
        "Coordinate nested teams to deliver a single coherent response.",  # 协调嵌套团队，交付一致的答复
        "Ask Research Team for evidence first, then Writing Team for synthesis.",  # 先让调研团队提供证据，再让写作团队综合输出
    ],
    markdown=True,                               # 以 Markdown 格式输出
    show_members_responses=True,                 # 显示每个成员的回答
)

# ---------------------------------------------------------------------------
# 运行团队
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parent_team.print_response(
        "Prepare a one-page brief on adopting AI coding assistants in a startup engineering team.",
        stream=True,
    )

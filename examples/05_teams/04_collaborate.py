"""
示例：Broadcast 模式 - 多源研究团队（保留原文件名）
场景：同一任务并发交给所有成员，再由 Leader 综合结果。
"""

import asyncio
from os import getenv

from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from agno.team import Team
from agno.team.mode import TeamMode
from agno.tools.tavily import TavilyTools
from dotenv import load_dotenv
from httpx import Timeout


def create_deepseek_model(api_key: str) -> DeepSeek:
    return DeepSeek(
        id=getenv("DEEPSEEK_MODEL_ID", "deepseek-v4-flash"),
        api_key=api_key,
        use_thinking=False,
        timeout=Timeout(connect=15.0, read=180.0, write=60.0, pool=60.0),
        max_retries=1,
    )


async def main() -> None:
    load_dotenv()

    api_key = getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 DeepSeek API Key，请设置 DEEPSEEK_API_KEY。")
    tavily_api_key = getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise RuntimeError("缺少 Tavily API Key，请设置 TAVILY_API_KEY。")

    reddit_agent = Agent(
        id="reddit_researcher",
        name="Reddit Researcher",
        role="从Reddit收集社区讨论",
        model=create_deepseek_model(api_key),
        tools=[
            TavilyTools(
                api_key=tavily_api_key,
                search_depth="basic",
                max_tokens=2000,
            )
        ],
        description="专注于Reddit社区的讨论和用户观点",
        instructions=[
            "搜索Reddit相关讨论",
            "总结社区主流观点",
            "提取用户真实体验",
            "标注讨论热度和时间",
        ],
        markdown=True,
    )

    news_agent = Agent(
        id="news_researcher",
        name="News Researcher",
        role="收集最新新闻资讯",
        model=create_deepseek_model(api_key),
        tools=[
            TavilyTools(
                api_key=tavily_api_key,
                search_depth="basic",
                max_tokens=2000,
            )
        ],
        description="专注于主流媒体的新闻报道",
        instructions=[
            "搜索权威媒体的最新报道",
            "关注官方声明和数据",
            "提供客观的新闻视角",
            "标注新闻来源和发布时间",
        ],
        markdown=True,
    )

    academic_agent = Agent(
        id="academic_researcher",
        name="Academic Researcher",
        role="查找学术研究",
        model=create_deepseek_model(api_key),
        tools=[
            TavilyTools(
                api_key=tavily_api_key,
                search_depth="basic",
                max_tokens=2000,
            )
        ],
        description="专注于学术论文和研究报告",
        instructions=[
            "搜索相关学术论文",
            "提取研究结论和数据",
            "关注权威机构的报告",
            "标注研究来源和发表时间",
        ],
        markdown=True,
    )

    research_team = Team(
        id="multi_source_broadcast_team",
        name="Multi-Source Research Team",
        mode=TeamMode.broadcast,
        members=[reddit_agent, news_agent, academic_agent],
        model=create_deepseek_model(api_key),
        instructions=[
            "将同一研究任务广播给所有成员",
            "综合不同来源的信息",
            "对比社区观点、媒体报道和学术研究",
            "识别共识与分歧",
            "生成全面的研究报告",
        ],
        show_members_responses=True,
        markdown=True,
    )

    print("=" * 60)
    print("研究任务：量子计算的最新发展")
    print("=" * 60)

    await research_team.aprint_response(
        "研究量子计算的最新发展趋势和应用。",
        stream=True,
        show_member_responses=True,
    )


if __name__ == "__main__":
    asyncio.run(main())

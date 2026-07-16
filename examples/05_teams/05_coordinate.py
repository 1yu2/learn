"""
示例：Coordinate 模式 - 数据分析团队
场景：Leader 根据任务动态选择成员、委派工作并综合结果。

Coordinate 由模型决定委派顺序，它不是保证固定步骤的工作流引擎。
"""

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


def main() -> None:
    load_dotenv()

    api_key = getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 DeepSeek API Key，请设置 DEEPSEEK_API_KEY。")
    tavily_api_key = getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise RuntimeError("缺少 Tavily API Key，请设置 TAVILY_API_KEY。")

    data_collector = Agent(
        id="data_collector",
        name="Data Collector",
        role="采集原始数据",
        model=create_deepseek_model(api_key),
        tools=[
            TavilyTools(
                api_key=tavily_api_key,
                search_depth="basic",
                max_tokens=2000,
            )
        ],
        description="负责从各种来源收集原始数据 ",
        instructions=[
            "搜索并收集相关的原始数据",
            "包含数据来源和时间戳",
            "保持数据的完整性",
            "整理成结构化格式",
        ],
    )

    data_cleaner = Agent(
        id="data_cleaner",
        name="Data Cleaner",
        role="清洗和结构化数据",
        model=create_deepseek_model(api_key),
        description="负责清洗数据、去除噪音、建立清晰结构",
        instructions=[
            "对团队提供的数据去除无关信息和重复内容",
            "标准化数据格式",
            "建立清晰的数据结构",
            "标注数据质量",
        ],
    )

    data_analyzer = Agent(
        id="data_analyzer",
        name="Data Analyzer",
        role="分析数据并得出结论",
        model=create_deepseek_model(api_key),
        description="负责深度分析数据，提取洞察和结论",
        instructions=[
            "基于团队提供的数据进行深度分析",
            "识别趋势和模式",
            "提供数据驱动的洞察",
            "生成分析报告",
        ],
    )

    analysis_team = Team(
        id="data_analysis_coordinate_team",
        name="Data Analysis Team",
        mode=TeamMode.coordinate,
        members=[data_collector, data_cleaner, data_analyzer],
        model=create_deepseek_model(api_key),
        instructions=[
            "根据当前任务选择合适的成员并委派明确任务",
            "将已获得的结果作为后续委派的必要上下文",
            "检查数据来源、质量和结论之间的一致性",
            "综合成一份完整的分析报告",
        ],
        share_member_interactions=True,
        show_members_responses=True,
        markdown=True,
    )

    print("=" * 60)
    print("任务：分析 2026 年电动汽车市场趋势")
    print("=" * 60)

    analysis_team.print_response(
        "分析 2026 年电动汽车市场的发展趋势。",
        stream=True,
        show_member_responses=True,
    )


if __name__ == "__main__":
    main()

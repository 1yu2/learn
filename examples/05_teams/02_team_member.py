"""
示例 2：Team 固定成员的角色分工
功能：股票价格查询 + 市场信息收集 + 分析与报告
"""

from os import getenv

from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from agno.team import Team
from agno.team.mode import TeamMode
from agno.tools.tavily import TavilyTools
from agno.tools.yfinance import YFinanceTools
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

    finance_agent = Agent(
        id="finance_001",
        name="Finance Analyst",
        role="获取和分析财务数据",
        description="""
        专门收集和分析财务数据的专家。
        擅长：
        - 股票价格查询
        - 分析师建议汇总
        - 公司财务信息分析
        """,
        model=create_deepseek_model(api_key),
        tools=[
            YFinanceTools(
                enable_stock_price=True,
                enable_company_info=True,
                enable_analyst_recommendations=True,
            )
        ],
        instructions=[
            "使用表格展示数据",
            "提供简洁的财务报告",
            "标注数据来源和更新时间",
        ],
        markdown=True,
    )

    data_collector = Agent(
        id="data_collector",
        name="Data Collector",
        role="搜索公司最新新闻和市场信息",
        model=create_deepseek_model(api_key),
        tools=[
            TavilyTools(
                api_key=tavily_api_key,
                search_depth="basic",
                max_tokens=2000,
            )
        ],
        instructions=["只保留与目标公司直接相关的信息并附上来源链接"],
        markdown=True,
    )

    data_analyzer = Agent(
        id="data_analyzer",
        name="Data Analyzer",
        role="结合财务数据和市场信息分析趋势与风险",
        model=create_deepseek_model(api_key),
        instructions=["区分客观数据、市场观点和推测，不要编造数据"],
        markdown=True,
    )

    report_writer = Agent(
        id="report_writer",
        name="Report Writer",
        role="撰写分析报告",
        model=create_deepseek_model(api_key),
        instructions=["将团队结果整理为结构清晰、包含风险提示的中文报告"],
        markdown=True,
    )

    research_team = Team(
        name="股票研究团队",
        members=[finance_agent, data_collector, data_analyzer, report_writer],
        model=create_deepseek_model(api_key),
        mode=TeamMode.coordinate,
        instructions=[
            "必须先委派 Finance Analyst 查询股票价格、公司信息和分析师建议。",
            "必须再委派 Data Collector 搜索目标公司的最新新闻和市场信息。",
            "将前两位成员的结果交给 Data Analyzer 分析趋势和风险。",
            "最后委派 Report Writer 整理报告，并由 Leader 生成统一结论。",
        ],
        show_members_responses=True,
        markdown=True,
        share_member_interactions=True,
        stream=True,
        stream_events=True,
        stream_member_events=True,
        debug_mode=False,
    )

    research_team.print_response(
        "请结合最新市场信息分析 NVIDIA（NVDA）的股票表现、分析师建议和主要风险。",
        stream=True,
        show_member_responses=True,
    )


if __name__ == "__main__":
    main()

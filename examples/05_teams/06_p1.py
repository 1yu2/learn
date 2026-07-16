"""
实战案例1：简单的研究团队
功能：信息收集 + 内容总结
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

    # 步骤1：创建研究员Agent
    researcher = Agent(
        id="education_researcher",
        name="Researcher",
        role="信息收集专家",
        model=create_deepseek_model(api_key),
        tools=[
            TavilyTools(
                api_key=tavily_api_key,
                search_depth="basic",
                max_tokens=2000,
            )
        ],
        description="""
        专业的网络信息研究员，擅长：
        - 使用搜索引擎查找权威信息
        - 从多个来源收集相关资料
        - 评估信息的可信度和时效性
        """,
        instructions=[
            "使用搜索工具查找最新、最相关的信息",
            "收集至少3-5个可靠来源的资料",
            "每个信息都要标注来源链接",
            "关注信息的发布时间和权威性",
            "整理成结构化的研究笔记",
        ],
        markdown=True,
    )

    # 步骤2：创建总结员Agent
    summarizer = Agent(
        id="education_summarizer",
        name="Summarizer",
        role="内容总结专家",
        model=create_deepseek_model(api_key),
        description="""
        专业的内容总结专家，擅长：
        - 提炼长篇内容的核心要点
        - 生成结构清晰的摘要
        - 保持信息的准确性和完整性
        """,
        instructions=[
            "仔细阅读研究员提供的所有资料",
            "提取关键信息和核心观点",
            "按重要性排序信息",
            "使用清晰的标题和层次结构",
            "生成简洁易读的总结报告",
            "保留重要的数据和引用来源",
        ],
        markdown=True,
    )

    # coordinate 模式由 Leader 动态委派和汇总，不是固定顺序的 Workflow。
    simple_team = Team(
        name="Simple Research Team",
        mode=TeamMode.coordinate,
        members=[researcher, summarizer],
        model=create_deepseek_model(api_key),
        instructions=[
            "协调研究员收集相关信息，并让总结员基于研究结果提炼要点",
            "确保信息传递的完整性",
            "生成一份简洁专业的研究报告",
        ],
        show_members_responses=True,
        markdown=True,
        stream=True,
        stream_events=True,
        stream_member_events=True,
        debug_mode=False,
    )

    print("=" * 60)
    print("🎯 研究任务：AI在教育领域的应用")
    print("=" * 60)

    simple_team.print_response(
        input="研究人工智能在教育领域的最新应用，包括技术趋势和实际案例",
        stream=True,
        show_member_responses=True,
    )


if __name__ == "__main__":
    main()

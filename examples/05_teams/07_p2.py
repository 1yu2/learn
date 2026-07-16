"""
实战案例2：内容创作团队
功能：由 Leader 协调研究、撰写与审校，生成高质量文章
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

    # 步骤1：创建内容研究员
    content_researcher = Agent(
        id="content_researcher",
        name="Content Researcher",
        role="内容研究员",
        model=create_deepseek_model(api_key),
        tools=[
            TavilyTools(
                api_key=tavily_api_key,
                search_depth="basic",
                max_tokens=2000,
            )
        ],
        description="""
        专业的内容研究专家，为文章创作提供坚实的素材基础。
        专长：
        - 深度主题研究
        - 数据和案例收集
        - 信息来源验证
        """,
        instructions=[
            "全面搜索主题相关的信息",
            "收集权威数据、实际案例和专家观点",
            "整理成结构化的研究笔记",
            "标注所有信息来源和发布时间",
            "评估信息的可信度和相关性",
            "为撰稿人提供充足的写作素材",
        ],
        markdown=True,
    )

    # 步骤2：创建专业撰稿人
    writer = Agent(
        id="content_writer",
        name="Writer",
        role="专业撰稿人",
        model=create_deepseek_model(api_key),
        description="""
        资深的内容创作者，擅长将研究资料转化为引人入胜的文章。
        专长：
        - 清晰的文章结构设计
        - 专业而易读的表达
        - 数据和案例的巧妙运用
        """,
        instructions=[
            "仔细阅读研究员提供的所有资料",
            "设计清晰的文章结构：引言-正文-结论",
            "使用研究资料中的数据和案例支撑观点",
            "保持专业性的同时确保可读性",
            "适当使用标题、列表等格式增强可读性",
            "确保逻辑连贯，论证充分",
            "字数控制在要求范围内",
        ],
        markdown=True,
    )

    # 步骤3：创建内容审校员
    editor = Agent(
        id="content_editor",
        name="Editor",
        role="内容审校员",
        model=create_deepseek_model(api_key),
        description="""
        经验丰富的编辑，确保文章达到出版级别的质量。
        专长：
        - 文章逻辑和结构优化
        - 语言表达精炼
        - 事实核查和质量把控
        """,
        instructions=[
            "全面审阅撰稿人的文章",
            "检查文章的逻辑性和连贯性",
            "优化语言表达，提升文笔质量",
            "确保数据和引用的准确性",
            "检查文章结构是否合理",
            "提出具体的修改建议",
            "生成最终的优化版本",
        ],
        markdown=True,
    )

    # coordinate 模式由 Leader 动态委派和汇总，不是固定顺序的 Workflow。
    content_team = Team(
        name="Content Creation Team",
        mode=TeamMode.coordinate,
        members=[content_researcher, writer, editor],
        model=create_deepseek_model(api_key),
        instructions=[
            "协调研究员提供全面的背景资料和素材",
            "让撰稿人基于资料创作文章初稿",
            "根据任务进展让审校员优化并生成最终版本",
            "确保每个环节的输出质量",
            "最终交付高质量的专业内容",
        ],
        show_members_responses=True,
        markdown=True,
        stream=True,
        stream_events=True,
        stream_member_events=True,
        debug_mode=False,
    )

    print("=" * 60)
    print("✍️ 创作任务：区块链技术在供应链管理中的应用")
    print("=" * 60)

    content_team.print_response(
        input="""
        撰写一篇关于区块链技术在供应链管理中应用的科普文章。

        要求：
        - 字数：500字左右
        - 包含实际案例
        - 适合普通读者阅读
        - 既要专业又要通俗易懂
        """,
        stream=True,
        show_member_responses=True,
    )


if __name__ == "__main__":
    main()

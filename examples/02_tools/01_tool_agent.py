from os import getenv

from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from agno.tools.duckduckgo import DuckDuckGoTools
from ddgs.exceptions import DDGSException
from dotenv import load_dotenv
from httpx import Timeout

load_dotenv()

api_key = getenv("DEEPSEEK_API_KEY") or getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "缺少 DeepSeek API Key，请设置 DEEPSEEK_API_KEY（推荐）或 OPENAI_API_KEY。"
    )


def handle_search_errors(function_call, arguments):
    try:
        return function_call(**arguments)
    except DDGSException:
        return "搜索未返回结果。请缩短关键词后再次调用 web_search；当前调用没有可引用的信息来源。"


# 创建具备搜索能力的智能体
web_agent = Agent(
    name="网络搜索助手",
    model=DeepSeek(
        id=getenv("DEEPSEEK_MODEL_ID", "deepseek-v4-flash"),
        api_key=api_key,
        timeout=Timeout(connect=15.0, read=600.0, write=60.0, pool=60.0),
        max_retries=2,
    ),
    tools=[
        DuckDuckGoTools(
            backend="auto",
            enable_news=False,
            fixed_max_results=5,
        )
    ],  # 添加搜索工具
    instructions=[
        "使用搜索工具查找最新信息 ",
        "始终提供信息来源链接 ",
        "以Markdown格式输出结果 "
    ],
    tool_hooks=[handle_search_errors],
    #show_tool_calls=True,  # 显示工具调用过程
    markdown=True
)

# 询问实时信息
web_agent.print_response(
    "2026年人工智能领域有哪些重大突破？",
    stream=True
)

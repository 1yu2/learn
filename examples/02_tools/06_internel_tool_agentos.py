from os import getenv

from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from agno.os import AgentOS
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
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


assistant = Agent(
    id="multi-tool",
    name="多功能助手",
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
        ),
        YFinanceTools(),    # 股票查询
    ],
    instructions=[
       "你是一个多功能智能助手",
        "可以搜索网页信息和查询股票数据",
        "提供准确、及时的信息"
    ],
    tool_hooks=[handle_search_errors],
    markdown=True,
)

agent_os = AgentOS(
    id="web-search-os",
    name="网络搜索 AgentOS",
    description="使用 DeepSeek 和网页搜索、查询股票数据工具回答最新信息问题。",
    agents=[assistant],
)
app = agent_os.get_app()

'''
启动命令：
uv run --extra agentos python examples/02_tools/06_internel_tool_agentos.py
'''

#   - AgentOS：http://localhost:7777 (http://localhost:7777)
#   - API 文档：http://localhost:7777/docs (http://localhost:7777/docs)
#   - 配置：http://localhost:7777/config (http://localhost:7777/config)
#   - Control Plane：https://os.agno.com (https://os.agno.com)，连接地址填写 http://localhost:7777
if __name__ == "__main__":
    agent_os.serve(
        app=app,
        host="localhost",
        port=7777,
    )

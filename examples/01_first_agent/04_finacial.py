import asyncio
from os import getenv

from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from agno.tools.mcp import MCPTools
from dotenv import load_dotenv
from agno.tools.yfinance import YFinanceTools
from curl_cffi.requests import Session


load_dotenv()

api_key = getenv("DEEPSEEK_API_KEY") or getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "缺少 DeepSeek API Key，请设置 DEEPSEEK_API_KEY（推荐）或 OPENAI_API_KEY。"
    )

# Example 1: All financial functions available (default behavior)
agent_full = Agent(
    name="金融分析师",
    role="专业的股票市场分析师",
    model=DeepSeek(
                id=getenv("DEEPSEEK_MODEL_ID", "deepseek-v4-flash"),
                api_key=api_key,
            ),
    tools=[YFinanceTools()],  # All functions enabled by default
    # description="You are a comprehensive investment analyst with access to all financial data functions.",
    instructions=[
        "使用表格展示数据",
        "提供详细的分析和建议",
        "只输出分析报告，不要额外文字"
    ],
    markdown=True,
)

print("\n=== Full Analysis Example ===")
agent_full.print_response(
    "请分析 NVIDIA（NVDA）的股票表现",
    markdown=True,
)


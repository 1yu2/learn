import asyncio
from os import getenv

from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from agno.tools.mcp import MCPTools
from dotenv import load_dotenv
from agno.tools.yfinance import YFinanceTools
from curl_cffi.requests import Session
from pydantic import BaseModel, Field


load_dotenv()

api_key = getenv("DEEPSEEK_API_KEY") or getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "缺少 DeepSeek API Key，请设置 DEEPSEEK_API_KEY（推荐）或 OPENAI_API_KEY。"
    )

# 定义输出结构
class WeatherData(BaseModel):
    month: str = Field(..., description="月份")
    season: str = Field(..., description="季节")
    avg_temp: str = Field(..., description="平均温度")

# 创建智能体
weather_agent = Agent(
    name="天气助手",
    model=DeepSeek(
                id=getenv("DEEPSEEK_MODEL_ID", "deepseek-v4-flash"),
                api_key=api_key,
            ),
    description="提供城市天气信息的助手",
    instructions=[
        "简洁明了",
        "返回Markdown表格格式"
    ],
    expected_output="包含月份、季节和平均温度的表格",
    markdown=True
)

# 查询天气
response = weather_agent.run(
    "纽约一年中每个月的天气如何？"
)
print(response.content)
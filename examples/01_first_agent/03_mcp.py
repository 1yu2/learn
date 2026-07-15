import asyncio
from os import getenv

from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from agno.tools.mcp import MCPTools
from dotenv import load_dotenv

load_dotenv()

api_key = getenv("DEEPSEEK_API_KEY") or getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "缺少 DeepSeek API Key，请设置 DEEPSEEK_API_KEY（推荐）或 OPENAI_API_KEY。"
    )


async def main():
    async with MCPTools(
        transport="streamable-http",
        url="https://docs.agno.com/mcp",
    ) as mcp_tools:
        agent = Agent(
            name="MCP 搜索助手",
            model=DeepSeek(
                id=getenv("DEEPSEEK_MODEL_ID", "deepseek-v4-flash"),
                api_key=api_key,
            ),
            tools=[mcp_tools],
            instructions=[
                "必须先使用 MCP 工具查询 Agno 官方文档，再回答问题。",
                "回答中提供相关文档路径或链接。",
            ],
            markdown=True,
        )

        await agent.aprint_response(
            "请根据 Agno 官方文档说明如何通过 Streamable HTTP 使用 MCPTools，"
            "并给出最小代码示例。/nothink",
            stream=True,
        )


if __name__ == "__main__":
    asyncio.run(main())

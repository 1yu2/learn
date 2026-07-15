from os import getenv

from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from dotenv import load_dotenv

load_dotenv()

api_key = getenv("DEEPSEEK_API_KEY") or getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "缺少 DeepSeek API Key，请设置 DEEPSEEK_API_KEY（推荐）或 OPENAI_API_KEY。"
    )

# 创建智能体
agent = Agent(
    name="助手",
    model=DeepSeek(
        id=getenv("DEEPSEEK_MODEL_ID", "deepseek-v4-flash"),
        api_key=api_key,
    ),
    description="一个友好的AI助手",
    markdown=True,
)

# 运行智能体
agent.print_response("你好，请介绍一下自己", stream=True)

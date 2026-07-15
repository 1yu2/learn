from os import getenv

from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from dotenv import load_dotenv
from httpx import Timeout

load_dotenv()

api_key = getenv("DEEPSEEK_API_KEY") or getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "缺少 DeepSeek API Key，请设置 DEEPSEEK_API_KEY（推荐）或 OPENAI_API_KEY。"
    )


# 定义工具
def get_current_time() -> str:
    """
    获取当前时间
    
    Returns:
        str: 当前时间字符串
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 创建Agent并绑定工具
agent = Agent(
    model=DeepSeek(
        id=getenv("DEEPSEEK_MODEL_ID", "deepseek-v4-flash"),
        api_key=api_key,
        timeout=Timeout(connect=15.0, read=600.0, write=60.0, pool=60.0),
        max_retries=2,
    ),
    tools=[get_current_time],  # 绑定工具列表
    markdown=True,  # 使用Markdown格式输出
)

# 使用Agent
agent.print_response("现在几点了？")
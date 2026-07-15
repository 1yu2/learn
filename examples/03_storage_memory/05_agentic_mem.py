###
##自定义memory
###

import os
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from os import getenv
from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from agno.memory import MemoryManager
from dotenv import load_dotenv
from httpx import Timeout

load_dotenv()

api_key = getenv("DEEPSEEK_API_KEY") or getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "缺少 DeepSeek API Key，请设置 DEEPSEEK_API_KEY（推荐）或 OPENAI_API_KEY。"
    )

db = SqliteDb(db_file="my_agent.db")

# 启用Agentic记忆模式
agentic_agent = Agent(
     model=DeepSeek(
        id=getenv("DEEPSEEK_MODEL_ID", "deepseek-v4-flash"),
        api_key=api_key,
        timeout=Timeout(connect=15.0, read=600.0, write=60.0, pool=60.0),
        max_retries=2,
    ),
    db=db,
    enable_agentic_memory=True,  # 🤖 Agent自主控制记忆
    description="""
    你是一个智能助手，拥有记忆管理能力。
    你可以自主决定：
    - 什么信息值得长期记住
    - 什么信息只是临时性的
    - 何时更新过时的记忆
    - 何时删除无用的记忆
    
    请明智地使用你的记忆工具。 /nothink
    """
)

# 场景1：Agent判断值得记住的信息
print("=" * 50)
print("场景1：提供重要信息")
print("=" * 50)

agentic_agent.print_response(
    "我每周二和周四下午3点有固定会议，请帮我记住。",
    user_id="agentic_user_001"
)

# 场景2：Agent判断临时性信息
print("\n" + "=" * 50)
print("场景2：提供临时信息")
print("=" * 50)

agentic_agent.print_response(
    "今天天气真不错啊！",
    user_id="agentic_user_001"
)


# 场景3：Agent回忆重要信息
print("\n" + "=" * 50)
print("场景3：请求回忆")
print("=" * 50)

agentic_agent.print_response(
    "我的固定会议时间是什么时候？",
    user_id="agentic_user_001"
)

# 场景4：更新记忆
print("\n" + "=" * 50)
print("场景4：更新信息")
print("=" * 50)

agentic_agent.print_response(
    "会议时间改了，现在是每周一和周三下午2点。",
    user_id="agentic_user_001"
)



# 场景5：验证更新
print("\n" + "=" * 50)
print("场景5：验证更新结果")
print("=" * 50)

agentic_agent.print_response(
    "再说一遍我的会议时间？",
    user_id="agentic_user_001"
)

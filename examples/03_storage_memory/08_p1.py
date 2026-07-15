
"""
案例1：电商客服机器人
功能：
- 记住客户的订单历史
- 记住客户的产品偏好
- 提供个性化推荐
"""
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
customer_service_bot = Agent(
     model=DeepSeek(
        id=getenv("DEEPSEEK_MODEL_ID", "deepseek-v4-flash"),
        api_key=api_key,
        timeout=Timeout(connect=15.0, read=600.0, write=60.0, pool=60.0),
        max_retries=2,
    ),
    db=db,
    enable_user_memories=True,
    description="""
    你是一个专业的电商客服助手。
    你需要：
    1. 记住每位客户的购买历史和偏好
    2. 提供个性化的产品推荐
    3. 追踪客户的问题和解决方案
    4. 保持友好、专业的态度 /nothink
    """
)

# 模拟完整的客服流程
print("=" * 60)
print("📅 第一天 - 首次咨询")
print("=" * 60)

customer_service_bot.print_response(
    """
    你好，我是李明。
    我想买一台笔记本电脑，主要用于视频编辑和图形设计，预算在8000-10000元。
    """,
    user_id="customer_limig"
)



print("\n" + "=" * 60)
print("📅 第三天 - 下单后咨询")
print("=" * 60)

customer_service_bot.print_response(
    "我昨天买了你们推荐的那台MacBook，想问一下什么时候发货？",
    user_id="customer_liming"
)

print("\n" + "=" * 60)
print("📅 一周后 - 追加购买")
print("=" * 60)

customer_service_bot.print_response(
    "我想再买一个适合我笔记本的鼠标，有什么推荐吗？",
    user_id="customer_liming"
)

print("\n" + "=" * 60)
print("📅 一个月后 - 售后咨询")
print("=" * 60)

customer_service_bot.print_response(
    "我之前买的MacBook用着很好，但是鼠标有点问题，能帮我看看吗？",
    user_id="customer_liming"
)

# 查看客户的完整记忆档案
print("\n" + "=" * 60)
print("📋 客户档案（记忆汇总）")
print("=" * 60)

memories = customer_service_bot.get_user_memories(user_id="customer_liming")
print(f"客户 李明 的记忆档案（共{len(memories)}条）：\n")
for i, memory in enumerate(memories, 1):
    print(f"{i}. {memory.memory}")
    print(f"   主题: {', '.join(memory.topics)}")
    print(f"   记录时间: {memory.updated_at}")
    print()





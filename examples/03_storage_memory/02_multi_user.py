from os import getenv

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.deepseek import DeepSeek
from dotenv import load_dotenv
from httpx import Timeout

load_dotenv()

api_key = getenv("DEEPSEEK_API_KEY") or getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "缺少 DeepSeek API Key，请设置 DEEPSEEK_API_KEY（推荐）或 OPENAI_API_KEY。"
    )
# 步骤1：创建数据库连接
db = SqliteDb(
    db_file="my_agent.db",
    memory_table="customer_memories",  # 自定义记忆表名
)

# 客服 Agent
customer_service_agent = Agent(
    model=DeepSeek(
        id=getenv("DEEPSEEK_MODEL_ID", "deepseek-v4-flash"),
        api_key=api_key,
        timeout=Timeout(connect=15.0, read=600.0, write=60.0, pool=60.0),
        max_retries=2,
    ),
    db=db,
    update_memory_on_run=True,
    add_history_to_context=True,
    num_history_runs=3,
    description="你是一个专业的客服人员，记住每位客户的需求和问题。/nothink",
)

CUSTOMER_SESSIONS = {
    "customer_A": "support_customer_A",
    "customer_B": "support_customer_B",
}

# 模拟两位不同的客户
print("=" * 50)
print("客户A的咨询")
print("=" * 50)
customer_service_agent.print_response(
    "我是客户A，我购买了iPhone 15，遇到了充电慢的问题。",
    user_id="customer_A",
    session_id=CUSTOMER_SESSIONS["customer_A"],
)

print("\n" + "=" * 50)
print("客户B的咨询")
print("=" * 50)
customer_service_agent.print_response(
    "你好，我是客户B，我想咨询MacBook的保修政策。",
    user_id="customer_B",
    session_id=CUSTOMER_SESSIONS["customer_B"],
)

print("\n" + "=" * 50)
print("客户A再次咨询")
print("=" * 50)
customer_service_agent.print_response(
    "你还记得我之前的问题吗？现在情况如何？",
    user_id="customer_A",
    session_id=CUSTOMER_SESSIONS["customer_A"],
)


print("\n" + "=" * 50)
print("客户B再次咨询")
print("=" * 50)
customer_service_agent.print_response(
    "关于我之前问的问题，能详细说明一下吗？",
    user_id="customer_B",
    session_id=CUSTOMER_SESSIONS["customer_B"],
)

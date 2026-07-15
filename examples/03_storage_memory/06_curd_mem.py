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
agent = Agent(
     model=DeepSeek(
        id=getenv("DEEPSEEK_MODEL_ID", "deepseek-v4-flash"),
        api_key=api_key,
        timeout=Timeout(connect=15.0, read=600.0, write=60.0, pool=60.0),
        max_retries=2,
    ),
    db=db,
    enable_user_memories=True,  # 🤖 Agent自主控制记忆
    description="你是一个专业的客服人员，记住每位客户的需求和问题。/nothink"
)


# ✅ CREATE - 创建记忆
print("1️⃣ 创建记忆...")
agent.print_response(
    "我喜欢意大利菜，尤其是披萨和意面。",
    user_id="user_food_lover"
)

# 📖 READ - 读取记忆
print("\n2️⃣ 读取所有记忆...")
all_memories = agent.get_user_memories(user_id="user_food_lover")
print(f"共有 {len(all_memories)} 条记忆")
for i, mem in enumerate(all_memories, 1):
    print(f"   记忆{i}: {mem.memory}")

# 🔄 UPDATE - 更新记忆
print("\n3️⃣ 更新记忆...")
# 注意：自动记忆模式下，Agent会自动处理更新
agent.print_response(
    "其实我现在更喜欢中餐了，特别是粤菜。",
    user_id="user_food_lover"
)

# 再次读取，查看更新
updated_memories = agent.get_user_memories(user_id="user_food_lover")
print(f"更新后有 {len(updated_memories)} 条记忆")

# ❌ DELETE - 删除记忆
print("\n4️⃣ 删除特定记忆...")
if len(all_memories) > 0:
    # 删除第一条记忆
    memory_to_delete = all_memories[0]
    agent.db.delete_user_memory(
        memory_id=memory_to_delete.memory_id,
        user_id="user_food_lover"
    )
    print(f"✅ 已删除记忆: {memory_to_delete.memory}")

# 再次读取，查看更新
updated_memories = agent.get_user_memories(user_id="user_food_lover")
print(f"删除后有 {len(updated_memories)} 条记忆")

# 🗑️ CLEAR - 清空所有记忆（慎用！）
print("\n5️⃣ 清空所有记忆（演示）...")
# agent.memory_manager.clear()  # ⚠️ 会删除数据库中所有记忆！
print("⚠️ clear()操作已注释，防止误操作")

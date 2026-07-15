import os
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
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
# 步骤1:创建数据库连接
db = SqliteDb(db_file="my_agent.db")

agent = Agent(
    model=DeepSeek(
        id=getenv("DEEPSEEK_MODEL_ID", "deepseek-v4-flash"),
        api_key=api_key,
        timeout=Timeout(connect=15.0, read=600.0, write=60.0, pool=60.0),
        max_retries=2,
    ),
    db=db,
    enable_user_memories=True,  # 开启自动记忆 ✨
    description="你是一个专业的客服人员，记住每位客户的需求和问题。/nothink"

)

# 1️⃣ 手动添加记忆（例如从现有用户数据库导入）
print("=" * 50)
print("步骤1：手动添加用户偏好记忆")
print("=" * 50)

# 模拟从用户数据库读取的信息
user_preferences = {
    "name": "王芳",
    "favorite_food": "川菜",
    "hobby": "摄影",
    "occupation": "平面设计师"
}

# 使用Agent的一次对话来创建记忆
agent.print_response(
    f"我是{user_preferences['name']}，我喜欢{user_preferences['favorite_food']}，"
    f"爱好是{user_preferences['hobby']}，职业是{user_preferences['occupation']}。",
    user_id="user_wangfang"
)

# 2️⃣ 检索用户的所有记忆
print("\n" + "=" * 50)
print("步骤2：检索用户的所有记忆")
print("=" * 50)

memories = agent.get_user_memories(user_id="user_wangfang")
if memories:
    print(f"找到 {len(memories)} 条记忆：")
    for i, memory in enumerate(memories, 1):
        print(f"\n记忆 #{i}:")
        print(f"  - ID: {memory.memory_id}")
        print(f"  - 内容: {memory.memory}")
        print(f"  - 主题: {memory.topics}")
        print(f"  - 更新时间: {memory.updated_at}")
else:
    print("未找到记忆")


# 3️⃣ 测试记忆是否生效
print("\n" + "=" * 50)
print("步骤3：测试记忆效果")
print("=" * 50)

agent.print_response(
    "根据我的职业和爱好，推荐一个适合我的周末活动。",
    user_id="user_wangfang"
)


# 4️⃣ 删除特定记忆（可选）
print("\n" + "=" * 50)
print("步骤4：演示删除记忆（实际使用时谨慎操作）")
print("=" * 50)

if memories and len(memories) > 0:
    memory_to_delete = memories[0]
    print(f"准备删除记忆: {memory_to_delete.memory}")
    
    # 取消注释以下行来实际删除
    # agent.delete_user_memory(
    #     memory_id=memory_to_delete.memory_id,
    #     user_id="user_wangfang"
    # )
    # print("✅ 记忆已删除")
    
    print("⚠️ 删除代码已注释，防止误操作")



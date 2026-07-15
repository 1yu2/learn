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

# 步骤2:启用自动记忆
agent = Agent(
    model=DeepSeek(
        id=getenv("DEEPSEEK_MODEL_ID", "deepseek-v4-flash"),
        api_key=api_key,
        timeout=Timeout(connect=15.0, read=600.0, write=60.0, pool=60.0),
        max_retries=2,
    ),
    db=db,
    enable_user_memories=True,  # 开启自动记忆 ✨
    description="你是一个友好的AI助手，善于记住用户的偏好。"

)
# 步骤3：第一次对话 - Agent会自动记住这些信息
agent.print_response(
    "你好！我叫李明，我喜欢喝咖啡，周末喜欢爬山。/nothink",
    user_id="user_001"  # 指定用户ID，用于区分不同用户
)

# 步骤4：稍后的对话 - Agent会自动回忆起之前的信息
agent.print_response(
    "你还记得我的爱好吗？/nothink", 
    user_id="user_001"
)

# 步骤5：再次询问 - 测试记忆的准确性
agent.print_response(
    "推荐一个适合我周末的活动吧！/nothink",
    user_id="user_001"
)



"""
理解记忆的主题标签系统
"""
# 创建一条复杂的记忆
agent.print_response(
    "我叫张三，是一名软件工程师，喜欢编程、看书和跑步，最喜欢的编程语言是Python。",
    user_id="user_zhangsan"
)

# 查看提取的主题
memories = agent.get_user_memories(user_id="user_zhangsan")
for memory in memories:
    print(f"记忆: {memory.memory}")
    print(f"主题标签: {memory.topics}")
    print("---")
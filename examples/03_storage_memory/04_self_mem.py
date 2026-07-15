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


# 自定义MemoryManager
custom_memory_manager = MemoryManager(
    db=db,
    model=DeepSeek(
        id=getenv("DEEPSEEK_MODEL_ID", "deepseek-v4-flash"),
        api_key=api_key,
        timeout=Timeout(connect=15.0, read=600.0, write=60.0, pool=60.0),
        max_retries=2,
    ),
    
    # 自定义记忆提取指令
    memory_capture_instructions="""
    从对话中提取以下类型的信息作为记忆：
    1. 用户的基本信息（姓名、职业等）
    2. 用户的偏好和兴趣
    3. 用户的目标和需求
    
    不要提取：
    - 敏感个人信息（身份证号、电话号码）
    - 密码或安全问题答案
    - 财务信息
    """,
    
    # 额外的隐私保护指令
    additional_instructions="绝不存储用户的真实姓名，使用昵称或代号代替。",
    
    debug_mode=True  # 开启调试，查看记忆提取过程
)

# 使用自定义MemoryManager创建Agent
privacy_agent = Agent(
    model=DeepSeek(
        id=getenv("DEEPSEEK_MODEL_ID", "deepseek-v4-flash"),
        api_key=api_key,
        timeout=Timeout(connect=15.0, read=600.0, write=60.0, pool=60.0),
        max_retries=2,
    ),
    memory_manager=custom_memory_manager,  # 使用自定义记忆管理器
    enable_user_memories=True,
    description="注重隐私保护的AI助手"
)

# 测试隐私保护
print("=" * 50)
print("测试：提供敏感信息")
print("=" * 50)

privacy_agent.print_response(
    "我叫张伟，身份证号是123456789012345678，我喜欢打篮球。",
    user_id="privacy_user_001"
)

# 检查存储的记忆
print("\n" + "=" * 50)
print("检查存储的记忆内容")
print("=" * 50)

memories = privacy_agent.get_user_memories(user_id="privacy_user_001")
if memories:
    for memory in memories:
        print(f"✅ 记忆: {memory.memory}")
        # 验证是否过滤了敏感信息
        if "身份证" in memory.memory or "123456" in memory.memory:
            print("⚠️ 警告：敏感信息未被过滤！")
        else:
            print("✓ 隐私保护生效")









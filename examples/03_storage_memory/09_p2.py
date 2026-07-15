"""
案例2：个人健康管理AI助手
功能：
- 记录用户的健康目标
- 追踪饮食和运动习惯
- 提供个性化健康建议
- 监测进度和变化
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

health_assistant = Agent(
     model=DeepSeek(
        id=getenv("DEEPSEEK_MODEL_ID", "deepseek-v4-flash"),
        api_key=api_key,
        timeout=Timeout(connect=15.0, read=600.0, write=60.0, pool=60.0),
        max_retries=2,
    ),
     db=db,
    enable_user_memories=True,
    description="""
    你是一位专业的健康管理AI助手。
    你的职责：
    1. 记录并追踪用户的健康目标、体重、饮食和运动情况
    2. 根据用户的历史数据提供个性化建议
    3. 鼓励用户坚持健康习惯
    4. 关注用户的进步和变化
    
    重要：你不是医生，对于严重健康问题应建议用户咨询专业医疗人员。/nothink
    """
)

# 模拟一个月的健康管理过程
print("=" * 60)
print("🏃 第1周 - 设定目标")
print("=" * 60)

health_assistant.print_response(
    """
    你好！我叫王万，今年28岁，身高165cm，体重65kg。
    我的目标是在3个月内减重到58kg，主要想改善体型。
    我平时工作比较忙，很少运动，也经常外食。
    """,
    user_id="user_wangwan"
)

print("\n" + "=" * 60)
print("🥗 第2周 - 饮食记录")
print("=" * 60)

health_assistant.print_response(
    """
    今天的饮食：
    早餐：全麦面包、牛奶、鸡蛋
    午餐：公司食堂的炒菜和米饭
    晚餐：沙拉和鸡胸肉
    
    运动：晚上跑步30分钟
    
    感觉还不错，但有点饿。
    """,
    user_id="user_wangwan"
)

print("\n" + "=" * 60)
print("💪 第3周 - 进展汇报")
print("=" * 60)

health_assistant.print_response(
    "这周坚持运动了5天，体重降到了63kg！感觉很有动力。",
    user_id="user_wangwan"
)

print("\n" + "=" * 60)
print("😫 第4周 - 遇到瓶颈")
print("=" * 60)

health_assistant.print_response(
    """
    最近一周体重没什么变化，一直在63kg徘徊。
    而且这几天工作压力大，又开始吃外卖了。
    感觉有点想放弃...
    """,
    user_id="user_wangwan"
)

print("\n" + "=" * 60)
print("🎉 第8周 - 回顾进展")
print("=" * 60)

health_assistant.print_response(
    "嗨！两个月过去了，帮我回顾一下我的进展如何？",
    user_id="user_wangwan"
)

# 生成健康报告
print("\n" + "=" * 60)
print("📊 健康档案报告")
print("=" * 60)

memories = health_assistant.get_user_memories(user_id="user_wangwan")
print(f"用户 王万 的健康档案（共{len(memories)}条记忆）：\n")

# 按主题分类显示
topics_dict = {}
for memory in memories:
    for topic in memory.topics:
        if topic not in topics_dict:
            topics_dict[topic] = []
        topics_dict[topic].append(memory.memory)

for topic, items in topics_dict.items():
    print(f"📌 {topic}:")
    for item in items:
        print(f"   - {item}")
    print()







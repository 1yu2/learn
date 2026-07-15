"""
案例3：AI学习辅导导师
功能：
- 追踪学生的学习进度和掌握情况
- 识别知识盲点和薄弱环节
- 记录学习风格和偏好
- 提供个性化的学习计划
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

ai_tutor = Agent(
     model=DeepSeek(
        id=getenv("DEEPSEEK_MODEL_ID", "deepseek-v4-flash"),
        api_key=api_key,
        timeout=Timeout(connect=15.0, read=600.0, write=60.0, pool=60.0),
        max_retries=2,
    ),
    db=db,
    enable_user_memories=True,
    description="""
    你是一位经验丰富的AI学习导师。
    
    你的职责：
    1. 了解每位学生的学习目标和现有水平
    2. 追踪学习进度，识别知识盲点
    3. 记录学习偏好（如视觉型、听觉型等）
    4. 提供个性化的学习建议和资源
    5. 定期回顾和调整学习计划
    
    教学原则：
    - 因材施教，根据学生特点调整教学方式
    - 鼓励为主，帮助学生建立自信
    - 注重理解而非死记硬背 /nothink
    """
)
# 模拟一个学期的辅导过程
print("=" * 60)
print("📖 第1周 - 初次见面")
print("=" * 60)

ai_tutor.print_response(
    """
    老师好！我是张小明，今年高二，想提高数学成绩。
    目前数学成绩中等，大约70-80分（满分100）。
    
    我的问题：
    - 函数这一块总是搞不懂
    - 做题速度慢
    - 容易粗心
    
    我比较喜欢看图表和动画来理解概念，纯文字的东西容易走神。
    目标是下次考试能考到85分以上。
    """,
    user_id="student_xiaoming"
)

print("\n" + "=" * 60)
print("📝 第3周 - 学习反馈")
print("=" * 60)

ai_tutor.print_response(
    """
    老师，这两周我按照你的建议：
    - 看了3Blue1Brown的函数视频，感觉好多了！
    - 每天练习5道函数题
    
    但是遇到复合函数还是会卡住。
    另外，二次函数的图像平移规律总是记混。
    """,
    user_id="student_xiaoming"
)

print("\n" + "=" * 60)
print("✅ 第6周 - 小测验")
print("=" * 60)

ai_tutor.print_response(
    """
    老师！今天数学小测验，函数部分我全对了！
    总分82分，进步了不少。
    
    不过立体几何部分错了很多，空间想象力比较差。
    """,
    user_id="student_xiaoming"
)

print("\n" + "=" * 60)
print("🎯 第12周 - 期中考试前")
print("=" * 60)

ai_tutor.print_response(
    "老师，下周就要期中考试了，能帮我回顾一下重点吗？我有点紧张。",
    user_id="student_xiaoming"
)

# 生成学习档案
print("\n" + "=" * 60)
print("📊 学生学习档案")
print("=" * 60)

memories = ai_tutor.get_user_memories(user_id="student_xiaoming")

# 分类整理
categories = {
    "基本信息": [],
    "学习目标": [],
    "强项": [],
    "薄弱环节": [],
    "学习偏好": [],
    "进步记录": []
}

for memory in memories:
    content = memory.memory
    # 简单的分类逻辑（实际应用中可以更智能）
    if any(word in content for word in ["叫", "年级", "学生"]):
        categories["基本信息"].append(content)
    elif any(word in content for word in ["目标", "想要", "希望"]):
        categories["学习目标"].append(content)
    elif any(word in content for word in ["全对", "进步", "提高"]):
        categories["进步记录"].append(content)
    elif any(word in content for word in ["搞不懂", "薄弱", "不会", "差"]):
        categories["薄弱环节"].append(content)
    elif any(word in content for word in ["喜欢", "偏好", "视觉"]):
        categories["学习偏好"].append(content)

print("学生档案 - 张小明\n")
for category, items in categories.items():
    if items:
        print(f"📌 {category}:")
        for item in items:
            print(f"   • {item}")
        print()

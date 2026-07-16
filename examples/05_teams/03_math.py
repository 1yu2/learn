"""
示例：Route 模式 - 数学运算团队
场景：Leader 从多个专家中选择一个成员处理请求。
"""

from os import getenv

from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from agno.team import Team
from agno.team.mode import TeamMode
from dotenv import load_dotenv
from httpx import Timeout


def create_deepseek_model(api_key: str) -> DeepSeek:
    return DeepSeek(
        id=getenv("DEEPSEEK_MODEL_ID", "deepseek-v4-flash"),
        api_key=api_key,
        use_thinking=False,
        timeout=Timeout(connect=15.0, read=180.0, write=60.0, pool=60.0),
        max_retries=1,
    )


def main() -> None:
    load_dotenv()

    api_key = getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 DeepSeek API Key，请设置 DEEPSEEK_API_KEY。")

    addition_agent = Agent(
        id="addition_agent",
        name="Addition Agent",
        role="执行加法运算",
        model=create_deepseek_model(api_key),
        description="专门处理加法问题，返回精确结果",
        instructions=[
            "只处理加法运算",
            "返回计算步骤和最终结果",
            "使用数学符号清晰展示",
        ],
    )

    subtraction_agent = Agent(
        id="subtraction_agent",
        name="Subtraction Agent",
        role="执行减法运算",
        model=create_deepseek_model(api_key),
        description="专门处理减法问题，返回精确结果",
        instructions=[
            "只处理减法运算",
            "返回计算步骤和最终结果",
            "使用数学符号清晰展示",
        ],
    )

    multiplication_agent = Agent(
        id="multiplication_agent",
        name="Multiplication Agent",
        role="执行乘法运算",
        model=create_deepseek_model(api_key),
        description="专门处理乘法问题，返回精确结果",
        instructions=[
            "只处理乘法运算",
            "返回计算步骤和最终结果",
        ],
    )

    math_team = Team(
        id="math_route_team",
        name="Math Team",
        mode=TeamMode.route,
        members=[addition_agent, subtraction_agent, multiplication_agent],
        model=create_deepseek_model(api_key),
        instructions=[
            "分析问题类型（加法/减法/乘法）",
            "将任务路由给最匹配的一位专家",
        ],
        show_members_responses=True,
        markdown=True,
    )

    math_team.print_response(
        "计算 156 + 789。",
        stream=True,
        show_member_responses=True,
    )


if __name__ == "__main__":
    main()

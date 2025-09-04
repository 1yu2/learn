from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage, AIMessage
from langchain_core.messages import ToolMessage
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
import os
import datetime
import sys

# 设置项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import LangFuseSession

# 初始化 Langfuse
langfuse = Langfuse(
    public_key="pk-lf-df66914f-c364-4cad-b4fb-801be1f00e0e",
    secret_key="sk-lf-26e92544-434f-4b22-b34c-eb09265fced0",
    host="http://192.168.10.60:11300",
)

langfuse_handler = CallbackHandler()

# 初始化模型
model = init_chat_model(
    model="openai:gpt-oss:20b",
    base_url="http://192.168.10.60:11434/v1",
    api_key="vllm",
    temperature=0,
)

# 定义天气查询工具
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。"""
    return f"{city}的天气总是阳光明媚！"

# 创建代理
agent = create_react_agent(
    model=model,
    tools=[get_weather],
    prompt="你是一个乐于助人的助手，用中文回答。收到工具调用结果后，生成完整的中文回复。",
)

# 定义输入模型
class Question(BaseModel):
    messages: list

# 主循环：支持多次查询城市天气
while True:
    # 获取用户输入
    city = input("请输入城市名称（输入 'quit' 或 'q' 退出）：")
    if city.lower() in ["quit", "q"]:
        print("再见！")
        break

    if not city.strip():
        print("城市名称不能为空，请重新输入！")
        continue

    # 更新 LangFuseSession 的 session_id
    LangFuseSession.session_id = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")

    try:
        # 运行代理查询天气
        result = agent.invoke(
            Question(messages=[HumanMessage(content=f"{city}的天气如何？")]),
            config={
                "callbacks": [langfuse_handler],
                "metadata": {
                    "langfuse_session_id": f"{LangFuseSession.prefix}-{LangFuseSession.session_id}",
                },
            },
        )
        print("result:",result)
        # 处理结果
        if isinstance(result, dict) and "messages" in result:
            last_message = result["messages"][-1]

            # 检查是否包含工具调用
            if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_call") and last_message.tool_calls:
                tool_call = last_message.tool_calls[0]
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if tool_name == "get_weather":
                    # 执行工具调用
                    weather_result = get_weather(**tool_args)
                    # 将工具结果作为 ToolMessage 传递回模型
                    tool_message = ToolMessage(
                        content=weather_result,
                        name=tool_name,
                        tool_call_id=tool_call["id"],
                 )
                    # 再次调用代理，生成最终回复
                    final_result = agent.invoke(
                        Question(messages=[HumanMessage(content=f"{city}的天气如何？"), last_message, tool_message]),
                        config={
                            "callbacks": [langfuse_handler],
                            "metadata": {
                                "langfuse_session_id": f"{LangFuseSession.prefix}-{LangFuseSession.session_id}",
                            },
                        },
                    )
                    # 输出最终回复
                    print(f"助手：{final_result['messages'][-1].content}")
                else:
                    print(f"助手：未知工具调用 {tool_name}")
            else:
                # 直接输出非工具调用的结果
                print(f"助手：{last_message.content}")
        else:
            print(f"助手：{result}")

    except Exception as e:
        print(f"查询失败：{str(e)}")
        print("请检查模型服务或输入格式后重试。")

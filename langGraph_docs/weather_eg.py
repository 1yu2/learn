from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
import os
import sys

# 确保可以从项目根目录导入 `config.py`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import model_configs

# 调用 `init_chat_model` 并传递配置参数
llm = init_chat_model(**model_configs)

def get_weather(city: str) -> str:
    """Get Weather for a given city"""
    return f"The weather in {city} is sunny with a high of 75°F."

# 创建代理
agent = create_react_agent(
    model=llm,
    tools=[get_weather],
    prompt='you are a helpful assistant'
)

# 调用代理进行推理
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "北京的天气怎么样?"}
    ]
})

msgs = response["messages"] if isinstance(response, dict) else response.messages
assistant_msgs = [m for m in msgs if getattr(m, "type", None) == "ai" or getattr(m, "role", None) == "assistant"]
final_msg = assistant_msgs[-1] if assistant_msgs else msgs[-1]
# ai回复内容
print(final_msg.content)

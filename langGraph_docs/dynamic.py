from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.chat_agent_executor import AgentState
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
def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:  
    user_name = config["configurable"].get("user_name")
    system_msg = f"You are a helpful assistant. Address the user as {user_name}."
    return [{"role": "system", "content": system_msg}] + state["messages"]

def get_weather(city: str) -> str:
    """Get Weather for a given city"""
    return f"The weather in {city} is sunny with a high of 75°F."


agent = create_react_agent(
    model=llm,
    tools=[get_weather],
    prompt=prompt
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "北京的天气怎么样"}]},
    config={"configurable": {"user_name": "鱼鱼"}}
)

msgs = response["messages"] if isinstance(response, dict) else response.messages
assistant_msgs = [m for m in msgs if getattr(m, "type", None) == "ai" or getattr(m, "role", None) == "assistant"]
final_msg = assistant_msgs[-1] if assistant_msgs else msgs[-1]
# ai回复内容
print(final_msg.content)
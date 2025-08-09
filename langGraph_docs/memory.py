from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
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

config = {'configurable': {'thread_id': '1'}}

checkpoint = InMemorySaver()
agent = create_react_agent(
    model=llm,
    tools=[get_weather],
    checkpointer=checkpoint
)

# run agent

sf_response = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather like in San Francisco?"}]},
    config
)

ny_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what about new york?"}]},
    config
)

print("San Francisco Response:")
msgs = sf_response["messages"] if isinstance(sf_response, dict) else sf_response.messages
assistant_msgs = [m for m in msgs if getattr(m, "type", None) == "ai" or getattr(m, "role", None) == "assistant"]
sf_ = assistant_msgs[-1] if assistant_msgs else msgs[-1]
# ai回复内容
print(sf_.content)
print("New York Response:")
msgs = ny_response["messages"] if isinstance(ny_response, dict) else ny_response.messages
assistant_msgs = [m for m in msgs if getattr(m, "type", None) == "ai" or getattr(m, "role", None) == "assistant"]
ny_ = assistant_msgs[-1] if assistant_msgs else msgs[-1]
print(ny_.content)
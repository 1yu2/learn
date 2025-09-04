from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
import os,sys
# 设置项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from config import LangFuseSession

langfuse = Langfuse(
    public_key="pk-lf-df66914f-c364-4cad-b4fb-801be1f00e0e",
    secret_key="sk-lf-26e92544-434f-4b22-b34c-eb09265fced0",
    host="http://192.168.10.60:11300",
)

langfuse_handler = CallbackHandler()


model = init_chat_model(
    "openai:Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
    base_url="http://192.168.10.60:31665/v1",
    api_key="vllm",
    temperature=0,
)


class Question(BaseModel):
    messages: list


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_react_agent(
    model=model,
    tools=[get_weather],
    prompt="You are a helpful assistant",
)

import datetime

LangFuseSession.session_id = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
# Run the agent
r = agent.invoke(
    Question(messages=[HumanMessage("what is the weather in sf")]),   
    config={"callbacks": [langfuse_handler], "metadata": {
                    "langfuse_session_id": f"{LangFuseSession.prefix}-{LangFuseSession.session_id}",
        }},
)

print(r)

from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

langfuse = Langfuse(
    public_key="pk-lf-df66914f-c364-4cad-b4fb-801be1f00e0e",
    secret_key="sk-lf-26e92544-434f-4b22-b34c-eb09265fced0",
    host="http://192.168.10.60:11300",
)

langfuse_handler = CallbackHandler()


model = init_chat_model(
    "openai:Qwen/Qwen3-30B-A3B-Instruct-2507",
    base_url="http://192.168.0.130:31665/v1",
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

# Run the agent
r = agent.invoke(
    Question(messages=[HumanMessage("what is the weather in sf")]),
    config={"callbacks": [langfuse_handler]},
)

print(r)

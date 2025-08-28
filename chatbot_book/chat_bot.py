import os
import sys
from langchain.chat_models import init_chat_model
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 确保可以从项目根目录导入 `config.py`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import model_configs, langfuse_configs


# 1. 初始化 Langfuse 和回调
langfuse = Langfuse(**langfuse_configs)
langfuse_handler = CallbackHandler()

# 2. 初始化 LLM，并绑定回调
llm = init_chat_model(
    **model_configs,
    callbacks=[langfuse_handler]  # 绑定 Langfuse 回调
)


# 3. 定义状态类型
class State(TypedDict):
    messages: Annotated[list, add_messages]


# 4. 创建状态图
graph_builder = StateGraph(State)


# 节点函数：调用 LLM
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()


while True:
    user_input = input("User: ")

    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    # 每次调用 graph 时，把回调传进去
    for event in graph.stream(
        {"messages": ("user", user_input)},
        config={"callbacks": [langfuse_handler], "run_name": "chatbot"}
    ):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

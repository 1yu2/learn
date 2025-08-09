from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model

# 1.定义状态类
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# 定义llm
import os
import sys

# 确保可以从项目根目录导入 `config.py`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import model_configs

# 2.定义llm,调用 `init_chat_model` 并传递配置参数
llm = init_chat_model(**model_configs)

# 3.将聊天模型集成到节点中
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)

# 4.添加边，指定开始与结束
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# 6.编译图
graph = graph_builder.compile()

# 7.获取 PNG 二进制数据
png_data = graph.get_graph().draw_mermaid_png()

# 保存到脚本所在目录
output_dir = os.path.dirname(__file__)
output_path = os.path.join(output_dir, "chatbot.png")
with open(output_path, "wb") as f:
    f.write(png_data)
print(f"Graph image saved to: {output_path}")


# 8.运行chatbot
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
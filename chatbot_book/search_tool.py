import os
from langfuse.langchain import CallbackHandler
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
import os,sys
# 设置项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from config import LangFuseSession,model_configs


langfuse_configs = {
    "public_key":"pk-lf-df66914f-c364-4cad-b4fb-801be1f00e0e",
    "secret_key":"sk-lf-26e92544-434f-4b22-b34c-eb09265fced0",
    "host":"http://192.168.10.60:11300",
}
llm = init_chat_model(
    **model_configs
)

# 设置 Tavily API Key（替换为你的实际 API Key）
os.environ["TAVILY_API_KEY"] = "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"  # 请替换为实际值

# 定义状态类型
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 初始化 Tavily 搜索工具
tavily = TavilySearch(max_results=2)

@tool
def web_search(query: str) -> dict:
    """使用 Tavily 搜索网络以获取最新信息。"""
    print(f"\n🔍 搜索: {query}")
    result = tavily.invoke({"query": query})
    print("\n" + "="*50)
    print("🌐 搜索结果:")
    for i, item in enumerate(result.get("results", []), 1):
        print(f" {i}. [{item.get('title')}]({item.get('url')})")
        print(f"    {item.get('content')[:100]}...")
    print("="*50 + "\n")
    return result

# 绑定工具到 LLM
tools = [web_search]
llm_with_tools = llm.bind_tools(tools)

# 创建状态图
graph_builder = StateGraph(State)

langfuse_handler = CallbackHandler()
# 聊天机器人节点
def chatbot(state: State):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages,
       config={"callbacks": [langfuse_handler], "metadata": {
                    "langfuse_session_id": f"{LangFuseSession.prefix}-{LangFuseSession.session_id}",
                }})

    return {"messages": [response]}

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
        {"messages": ("user", user_input)}
    ):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

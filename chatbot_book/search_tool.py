import os
import sys
import json
from langchain.chat_models import init_chat_model
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_core.messages import ToolMessage, BaseMessage

# 设置项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import model_configs, langfuse_configs, TAVILY_API_KEY

# 设置 Tavily API Key
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# 定义状态类型
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 初始化 Langfuse
langfuse = Langfuse(**langfuse_configs)
langfuse_handler = CallbackHandler()

# 初始化 LLM
llm = init_chat_model(
    **model_configs,
    callbacks=[langfuse_handler]
)

tavily = TavilySearch(max_results=2)

@tool
def web_search(query: str) -> dict:
    """使用 Tavily 搜索网络以获取最新信息。返回原始结果字典，便于后续处理。"""
    print(f"\n🔍 正在执行网络搜索: {query}")
    result = tavily.invoke({"query": query})
    
    # 打印真实搜索结果（可读格式）
    print("\n" + "="*60)
    print("🌐 搜索结果详情:")
    for i, item in enumerate(result.get("results", []), 1):
        print(f" {i}. [{item.get('title')}]({item.get('url')})")
        print(f"    {item.get('content')[:200]}...")
    print("="*60 + "\n")
    
    # 返回原始 dict，不要 json.dumps！LangChain 会自动序列化
    return result  # 直接返回 dict

tools = [web_search]
llm_with_tools = llm.bind_tools(tools)

# 创建状态图
graph_builder = StateGraph(State)

# 定义聊天机器人节点
def chatbot(state: State):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

graph_builder.add_node("chatbot", chatbot)

# 工具节点（无需修改，但增强打印）
class BasicToolNode:
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("输入中未找到消息")

        outputs = []
        for tool_call in message.tool_calls:
            try:
                print(f"🛠️ 执行工具调用: {tool_call['name']} (参数: {tool_call['args']})")
                tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
                outputs.append(
                    ToolMessage(
                        content=json.dumps(tool_result, ensure_ascii=False, indent=2),  # 格式化字符串用于调试
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
            except Exception as e:
                print(f"❌ 工具调用失败: {e}")
                outputs.append(
                    ToolMessage(
                        content=f"工具执行出错: {str(e)}",
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
        return {"messages": outputs}

tool_node = BasicToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# 路由函数
def route_tools(state: State) -> Literal["tools", "__end__"]:
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"输入状态中未找到消息: {state}")

    tool_calls = getattr(ai_message, "tool_calls", [])
    print(f"🚦 路由判断: {'需要调用工具' if tool_calls else '无需工具'}")
    return "tools" if tool_calls else "__end__"

# 添加条件边
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", "__end__": END},
)

# 添加边
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# 编译图
graph = graph_builder.compile()

# 主循环
while True:
    user_input = input("\nUser: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("👋 Goodbye!")
        break

    # 流式执行 ceshi
    events = graph.stream({"messages": [("user", user_input)]})
    final_assistant_message = None

    for event in events:
        for value in event.values():
            messages = value.get("messages", [])
            if not messages:
                continue
            last_message = messages[-1]

            if isinstance(last_message, ToolMessage):
                try:
                    result = json.loads(last_message.content)  # 为了打印格式化
                    print(f"✅ 工具返回结果 ({last_message.name}):")
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                except:
                    print(f"✅ 工具返回: {last_message.content}")

            elif isinstance(last_message, BaseMessage):
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    print(f"🤖 Assistant: 正在调用工具 {[(t['name'], t['args']) for t in last_message.tool_calls]}")
                else:
                    final_assistant_message = last_message.content

    if final_assistant_message:
        print(f"🤖 Assistant: {final_assistant_message}")
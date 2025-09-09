import os
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.chat_models import init_chat_model
from config import  TAVILY_API_KEY,model_configs

# 环境变量配置
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"] = "pr-diligent-spray-77"
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# 定义状态
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 定义工具
@tool
def human_assistance(query: str) -> str:
    """请求人工干预以回答问题"""
    return input(f"\n=== 人工干预请求 ===\n问题: {query}\n请输入您的回答: ")

# 初始化 LLM 和工具
llm = init_chat_model(**model_configs)
tools = [TavilySearch(max_results=2), human_assistance]
llm_with_tools = llm.bind_tools(tools)

# 定义 chatbot 节点
def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    return {"messages": [message]}

# 创建图
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# 使用 MemorySaver 支持中断
checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

# 流式更新函数
def stream_graph_updates(user_input: str, config: dict):
    events = graph.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config,
        stream_mode="values",
        interrupt_before=["tools"],
    )
    for event in events:
        last_message = event["messages"][-1]
        if isinstance(last_message, HumanMessage):
            print(f"\n用户输入: {last_message.content}")
        elif isinstance(last_message, AIMessage):
            if last_message.content:
                print(f"\nAI 响应: {last_message.content}")
            if last_message.tool_calls:
                for tool_call in last_message.tool_calls:
                    print(f"\n调用工具: {tool_call['name']}，参数: {tool_call['args']}")
        elif isinstance(last_message, ToolMessage):
            print(f"\n工具结果: {last_message.content}")

# 主循环
def main():
    print("欢迎使用聊天机器人！输入 'quit'、'exit' 或 'q' 退出。")
    config = {"configurable": {"thread_id": "1"}}
    while True:
        try:
            user_input = input("\n用户: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("再见！")
                break

            # 运行图并处理中断
            stream_graph_updates(user_input, config)

            # 检查是否有中断
            state = graph.get_state(config)
            if state.next and state.next[0] == "tools":
                for tool_call in state.values["messages"][-1].tool_calls:
                    if tool_call["name"] == "human_assistance":
                        query = tool_call["args"]["query"]
                        human_response = input(f"\n=== 人工干预请求 ===\n问题: {query}\n请输入您的回答: ")
                        graph.update_state(
                            config,
                            {"messages": [ToolMessage(
                                content=human_response,
                                tool_call_id=tool_call["id"]
                            )]},
                            as_node="tools"
                        )
                        # 继续流式输出
                        for event in graph.stream(None, config, stream_mode="values"):
                            last_message = event["messages"][-1]
                            if isinstance(last_message, HumanMessage):
                                print(f"\n用户输入: {last_message.content}")
                            elif isinstance(last_message, AIMessage):
                                if last_message.content:
                                    print(f"\nAI 响应: {last_message.content}")
                            elif isinstance(last_message, ToolMessage):
                                print(f"\n工具结果: {last_message.content}")

        except Exception as e:
            print(f"\n发生错误: {e}")
            user_input = "What do you know about LangGraph?"
            print(f"\n用户: {user_input}")
            stream_graph_updates(user_input, config)
            break

if __name__ == "__main__":
    main()
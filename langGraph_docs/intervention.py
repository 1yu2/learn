"""
START
  │
  ├─→ human_node_1 → interrupt → 等待人工输入 → 更新 text_1
  │
  └─→ human_node_2 → interrupt → 等待人工输入 → 更新 text_2
                              ↓
                     收到 resume 命令后恢复
                              ↓
                       输出最终结果
"""
from typing import TypedDict
import uuid
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command


# 定义状态
class State(TypedDict):
    text_1: str
    text_2: str


# 模拟人工交互的节点
def human_node_1(state: State):
    value = interrupt({"text_to_revise": state["text_1"]})
    return {"text_1": value}


def human_node_2(state: State):
    value = interrupt({"text_to_revise": state["text_2"]})
    return {"text_2": value}


# 构建图
graph_builder = StateGraph(State)
graph_builder.add_node("human_node_1", human_node_1)
graph_builder.add_node("human_node_2", human_node_2)

# 从 START 并行启动两个节点
graph_builder.add_edge(START, "human_node_1")
graph_builder.add_edge(START, "human_node_2")

# 编译
checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

# 设置线程 ID
thread_id = str(uuid.uuid4())
config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

# 第一次执行 -> 会触发 interrupt
result = graph.invoke(
    {"text_1": "original text 1", "text_2": "original text 2"}, config=config
)

# 获取中断点
state = graph.get_state(config)
print("Interrupts:")
for i in state.interrupts:
    print(" -", i.id, i.value)

# 构造人工输入的 resume_map
resume_map = {
    i.id: f"edited text for {i.value['text_to_revise']}"
    for i in state.interrupts
}

# 恢复执行
final_result = graph.invoke(Command(resume=resume_map), config=config)
print("Final result:", final_result)

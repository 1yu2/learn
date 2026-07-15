from os import getenv
from typing import Optional

from agno.agent import Agent
from agno.db.schemas import UserMemory
from agno.db.sqlite import SqliteDb
from agno.memory import MemoryManager
from agno.models.deepseek import DeepSeek
from dotenv import load_dotenv
from httpx import Timeout


class AgentScopedMemoryManager(MemoryManager):
    """只读取当前 Agent 创建的用户记忆。"""

    def __init__(self, *, agent_id: str, **kwargs):
        if not agent_id.strip():
            raise ValueError("agent_id 不能为空")
        super().__init__(**kwargs)
        self.agent_id = agent_id

    def read_from_db(
        self,
        user_id: Optional[str] = None,
    ) -> Optional[dict[str, list[UserMemory]]]:
        if self.db is None:
            return None

        memories = self.db.get_user_memories(  # type: ignore[union-attr]
            user_id=user_id,
            agent_id=self.agent_id,
        )
        memories_by_user: dict[str, list[UserMemory]] = {}
        for memory in memories:
            if memory.user_id is not None and memory.memory_id is not None:
                memories_by_user.setdefault(memory.user_id, []).append(memory)
        return memories_by_user


def create_support_agent(
    *,
    agent_id: str,
    db: SqliteDb,
    api_key: str,
) -> Agent:
    model = DeepSeek(
        id=getenv("DEEPSEEK_MODEL_ID", "deepseek-v4-flash"),
        api_key=api_key,
        timeout=Timeout(connect=15.0, read=600.0, write=60.0, pool=60.0),
        max_retries=2,
    )
    memory_manager = AgentScopedMemoryManager(
        agent_id=agent_id,
        model=model,
        db=db,
    )
    return Agent(
        id=agent_id,
        model=model,
        db=db,
        memory_manager=memory_manager,
        update_memory_on_run=True,
        description="你是一个专业的客服人员，记住每位客户的需求和问题。",
        instructions=[
            "只能根据当前 Agent 可见的用户记忆回答历史问题。",
            "没有明确记录时直接说明不知道，不要猜测客户信息。",
        ],
    )


def main() -> None:
    load_dotenv()

    api_key = getenv("DEEPSEEK_API_KEY") or getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "缺少 DeepSeek API Key，请设置 DEEPSEEK_API_KEY（推荐）或 OPENAI_API_KEY。"
        )

    # 两个 Agent 共用同一个数据库和同一张记忆表。
    db = SqliteDb(db_file="my_agent.db")
    company_a_agent = create_support_agent(
        agent_id="company_a_support",
        db=db,
        api_key=api_key,
    )
    company_b_agent = create_support_agent(
        agent_id="company_b_support",
        db=db,
        api_key=api_key,
    )

    company_a_agent.print_response(
        "我购买了你们的产品A",
        user_id="customer_001",
    )

    # user_id 相同，但 B 的 MemoryManager 只会读取 company_b_support 的记忆。
    company_b_agent.print_response(
        "这位客户之前买了什么？",
        user_id="customer_001",
    )


if __name__ == "__main__":
    main()

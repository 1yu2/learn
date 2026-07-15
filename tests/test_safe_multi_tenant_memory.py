import sys
from pathlib import Path
from runpy import run_path
from types import ModuleType

from agno.db.schemas import UserMemory
from agno.db.sqlite import SqliteDb


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "03_storage_memory"
    / "07_safe_multi_person.py"
)


class FakeAgent:
    instances = []

    def __init__(self, **kwargs):
        self.config = kwargs
        self.calls = []
        self.instances.append(self)

    def print_response(self, message, **kwargs):
        self.calls.append((message, kwargs))


class FakeDependency:
    def __init__(self, **kwargs):
        self.config = kwargs


class FakeMemoryManager:
    def __init__(self, *, model=None, db=None, **kwargs):
        self.model = model
        self.db = db
        self.config = kwargs


def fake_module(name, **attributes):
    module = ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


def execute_script(monkeypatch):
    FakeAgent.instances.clear()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    for package_name in ["agno", "agno.db", "agno.models"]:
        package = fake_module(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    modules = {
        "agno.agent": fake_module("agno.agent", Agent=FakeAgent),
        "agno.db.sqlite": fake_module("agno.db.sqlite", SqliteDb=FakeDependency),
        "agno.memory": fake_module("agno.memory", MemoryManager=FakeMemoryManager),
        "agno.models.deepseek": fake_module(
            "agno.models.deepseek", DeepSeek=FakeDependency
        ),
        "dotenv": fake_module("dotenv", load_dotenv=lambda: None),
        "httpx": fake_module("httpx", Timeout=FakeDependency),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    run_path(str(SCRIPT_PATH), run_name="__main__")


def test_agent_scoped_memory_filters_same_user_by_agent(tmp_path):
    namespace = run_path(str(SCRIPT_PATH))

    assert "AgentScopedMemoryManager" in namespace
    manager_type = namespace["AgentScopedMemoryManager"]

    db = SqliteDb(db_file=str(tmp_path / "shared.db"))
    db.upsert_user_memory(
        UserMemory(
            memory_id="memory-a",
            memory="用户购买了产品A",
            user_id="customer_001",
            agent_id="company_a_support",
        )
    )
    db.upsert_user_memory(
        UserMemory(
            memory_id="memory-b",
            memory="用户购买了产品B",
            user_id="customer_001",
            agent_id="company_b_support",
        )
    )

    company_a_manager = manager_type(agent_id="company_a_support", db=db)
    company_b_manager = manager_type(agent_id="company_b_support", db=db)

    company_a_memories = company_a_manager.get_user_memories(user_id="customer_001")
    company_b_memories = company_b_manager.get_user_memories(user_id="customer_001")

    assert [memory.memory for memory in company_a_memories] == ["用户购买了产品A"]
    assert [memory.memory for memory in company_b_memories] == ["用户购买了产品B"]
    assert db.memory_table_name == "agno_memories"


def test_demo_uses_separate_agent_filters_with_same_database_and_user(monkeypatch):
    execute_script(monkeypatch)

    assert len(FakeAgent.instances) == 2
    company_a_agent, company_b_agent = FakeAgent.instances
    assert company_a_agent.config["db"] is company_b_agent.config["db"]
    assert company_a_agent.config["db"].config == {"db_file": "my_agent.db"}

    company_a_manager = company_a_agent.config["memory_manager"]
    company_b_manager = company_b_agent.config["memory_manager"]
    assert company_a_manager.agent_id == "company_a_support"
    assert company_b_manager.agent_id == "company_b_support"

    _, company_a_call = company_a_agent.calls[0]
    _, company_b_call = company_b_agent.calls[0]
    assert company_a_call["user_id"] == "customer_001"
    assert company_b_call["user_id"] == "customer_001"

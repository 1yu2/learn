from pathlib import Path
from runpy import run_path
from types import ModuleType

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "03_storage_memory"
    / "02_multi_user.py"
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


def fake_module(name, **attributes):
    module = ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


@pytest.fixture
def executed_agent(monkeypatch):
    FakeAgent.instances.clear()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    package_names = ["agno", "agno.db", "agno.models"]
    for package_name in package_names:
        package = fake_module(package_name)
        package.__path__ = []
        monkeypatch.setitem(__import__("sys").modules, package_name, package)

    modules = {
        "agno.agent": fake_module("agno.agent", Agent=FakeAgent),
        "agno.db.sqlite": fake_module("agno.db.sqlite", SqliteDb=FakeDependency),
        "agno.models.deepseek": fake_module(
            "agno.models.deepseek", DeepSeek=FakeDependency
        ),
        "dotenv": fake_module("dotenv", load_dotenv=lambda: None),
        "httpx": fake_module("httpx", Timeout=FakeDependency),
    }
    for name, module in modules.items():
        monkeypatch.setitem(__import__("sys").modules, name, module)

    run_path(str(SCRIPT_PATH), run_name="__main__")

    assert len(FakeAgent.instances) == 1
    return FakeAgent.instances[0]


def test_agent_loads_recent_session_history(executed_agent):
    assert executed_agent.config.get("update_memory_on_run") is True
    assert "enable_user_memories" not in executed_agent.config
    assert executed_agent.config.get("add_history_to_context") is True
    assert executed_agent.config.get("num_history_runs") == 3


def test_each_customer_uses_a_stable_isolated_session(executed_agent):
    sessions_by_user = {}

    for _, call_options in executed_agent.calls:
        user_id = call_options["user_id"]
        session_id = call_options.get("session_id")

        assert session_id
        if user_id in sessions_by_user:
            assert sessions_by_user[user_id] == session_id
        else:
            sessions_by_user[user_id] = session_id

    assert set(sessions_by_user) == {"customer_A", "customer_B"}
    assert sessions_by_user["customer_A"] != sessions_by_user["customer_B"]

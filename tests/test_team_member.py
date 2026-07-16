import sys
from pathlib import Path
from runpy import run_path
from types import ModuleType


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "examples" / "05_teams" / "02_team_member.py"


class FakeDeepSeek:
    instances = []

    def __init__(self, **kwargs):
        self.config = kwargs
        self.instances.append(self)


class FakeAgent:
    instances = []

    def __init__(self, **kwargs):
        self.config = kwargs
        self.instances.append(self)


class FakeTeam:
    instances = []

    def __init__(self, **kwargs):
        self.config = kwargs
        self.calls = []
        self.instances.append(self)

    def print_response(self, message=None, **kwargs):
        if message is None:
            message = kwargs.pop("input")
        self.calls.append((message, kwargs))


class FakeToolkit:
    instances = []

    def __init__(self, **kwargs):
        self.config = kwargs
        self.instances.append(self)


class FakeYFinanceTools(FakeToolkit):
    instances = []


class FakeTavilyTools(FakeToolkit):
    instances = []


class FakeTimeout:
    def __init__(self, **kwargs):
        self.config = kwargs


class FakeTeamMode:
    coordinate = "coordinate"


def fake_module(name, **attributes):
    module = ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


def install_fake_dependencies(monkeypatch):
    FakeDeepSeek.instances.clear()
    FakeAgent.instances.clear()
    FakeTeam.instances.clear()
    FakeYFinanceTools.instances.clear()
    FakeTavilyTools.instances.clear()

    for package_name in ["agno", "agno.models", "agno.tools"]:
        package = fake_module(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    modules = {
        "agno.agent": fake_module("agno.agent", Agent=FakeAgent),
        "agno.models.deepseek": fake_module("agno.models.deepseek", DeepSeek=FakeDeepSeek),
        "agno.team": fake_module("agno.team", Team=FakeTeam),
        "agno.team.mode": fake_module("agno.team.mode", TeamMode=FakeTeamMode),
        "agno.tools.tavily": fake_module("agno.tools.tavily", TavilyTools=FakeTavilyTools),
        "agno.tools.yfinance": fake_module("agno.tools.yfinance", YFinanceTools=FakeYFinanceTools),
        "dotenv": fake_module("dotenv", load_dotenv=lambda: None),
        "httpx": fake_module("httpx", Timeout=FakeTimeout),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_finance_tools_and_team_members_are_configured_and_executed(monkeypatch):
    install_fake_dependencies(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-test-key")

    run_path(str(SCRIPT_PATH), run_name="__main__")

    assert len(FakeYFinanceTools.instances) == 1
    finance_tools = FakeYFinanceTools.instances[0].config
    assert finance_tools == {
        "enable_stock_price": True,
        "enable_company_info": True,
        "enable_analyst_recommendations": True,
    }

    assert len(FakeAgent.instances) == 4
    assert all(isinstance(agent.config.get("model"), FakeDeepSeek) for agent in FakeAgent.instances)

    assert len(FakeTeam.instances) == 1
    team = FakeTeam.instances[0]
    assert team.config["members"] == FakeAgent.instances
    assert isinstance(team.config.get("model"), FakeDeepSeek)
    assert team.config.get("mode") == FakeTeamMode.coordinate
    assert team.config.get("stream_member_events") is True

    assert len(team.calls) == 1
    message, options = team.calls[0]
    assert "NVDA" in message
    assert options.get("stream") is True
    assert options.get("show_member_responses") is True

    assert len(FakeTavilyTools.instances) == 1
    assert FakeTavilyTools.instances[0].config.get("api_key") == "tavily-test-key"

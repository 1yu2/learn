import sys
from pathlib import Path
from runpy import run_path
from types import ModuleType, SimpleNamespace

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "examples" / "05_teams" / "01_basic_team.py"


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
    events = []

    def __init__(self, **kwargs):
        self.config = kwargs
        self.calls = []
        self.instances.append(self)

    def print_response(self, message=None, **kwargs):
        if message is None:
            message = kwargs.pop("input")
        self.calls.append((message, kwargs))

    def run(self, message=None, **kwargs):
        if message is None:
            message = kwargs.pop("input")
        self.calls.append((message, kwargs))
        yield from self.events


class FakeDependency:
    def __init__(self, **kwargs):
        self.config = kwargs


class FakeEventName:
    def __init__(self, value):
        self.value = value


class FakeRunEvent:
    run_started = FakeEventName("RunStarted")
    run_content = FakeEventName("RunContent")
    run_error = FakeEventName("RunError")
    tool_call_started = FakeEventName("ToolCallStarted")
    tool_call_completed = FakeEventName("ToolCallCompleted")
    tool_call_error = FakeEventName("ToolCallError")
    run_completed = FakeEventName("RunCompleted")


class FakeTeamRunEvent:
    tool_call_started = FakeEventName("TeamToolCallStarted")
    tool_call_completed = FakeEventName("TeamToolCallCompleted")
    tool_call_error = FakeEventName("TeamToolCallError")
    run_content = FakeEventName("TeamRunContent")
    run_completed = FakeEventName("TeamRunCompleted")
    run_error = FakeEventName("TeamRunError")


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
    FakeTeam.events = []

    for package_name in ["agno", "agno.db", "agno.models", "agno.run", "agno.tools"]:
        package = fake_module(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    modules = {
        "agno.agent": fake_module("agno.agent", Agent=FakeAgent),
        "agno.db.sqlite": fake_module("agno.db.sqlite", SqliteDb=FakeDependency),
        "agno.memory": fake_module("agno.memory", MemoryManager=FakeDependency),
        "agno.models.deepseek": fake_module("agno.models.deepseek", DeepSeek=FakeDeepSeek),
        "agno.run.agent": fake_module("agno.run.agent", RunEvent=FakeRunEvent),
        "agno.run.team": fake_module("agno.run.team", TeamRunEvent=FakeTeamRunEvent),
        "agno.team": fake_module("agno.team", Team=FakeTeam),
        "agno.team.mode": fake_module("agno.team.mode", TeamMode=FakeTeamMode),
        "agno.tools.tavily": fake_module("agno.tools.tavily", TavilyTools=FakeDependency),
        "dotenv": fake_module("dotenv", load_dotenv=lambda: None),
        "httpx": fake_module("httpx", Timeout=FakeDependency),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_team_streams_progress_and_disables_deepseek_thinking(monkeypatch):
    install_fake_dependencies(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-test-key")

    run_path(str(SCRIPT_PATH), run_name="__main__")

    assert len(FakeDeepSeek.instances) == 3
    assert all(model.config.get("use_thinking") is False for model in FakeDeepSeek.instances)

    assert len(FakeTeam.instances) == 1
    message, options = FakeTeam.instances[0].calls[0]
    assert message == "分析下2026年企业级问数和本体论实际落地方案"
    assert options.get("stream") is True
    assert options.get("stream_events") is True
    assert FakeTeam.instances[0].config.get("mode") == FakeTeamMode.coordinate
    assert FakeTeam.instances[0].config.get("stream_member_events") is True
    assert FakeTeam.instances[0].config.get("debug_mode") is False


def test_team_process_prints_delegation_search_and_both_agent_responses(monkeypatch, capsys):
    install_fake_dependencies(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-test-key")

    FakeTeam.events = [
        SimpleNamespace(
            event="TeamToolCallStarted",
            team_name="Research Team",
            tool=SimpleNamespace(
                tool_name="delegate_task_to_member",
                tool_args={
                    "member_id": "web-agent",
                    "task": "搜索医疗 AI 的最新应用",
                },
            ),
        ),
        SimpleNamespace(event="RunStarted", agent_name="Web Agent", run_id="web-run"),
        SimpleNamespace(
            event="RunContent",
            agent_name="Web Agent",
            run_id="web-run",
            content="准备查询",
        ),
        SimpleNamespace(
            event="ToolCallStarted",
            agent_name="Web Agent",
            tool=SimpleNamespace(
                tool_name="search",
                tool_args={"query": "2026 医疗 AI 最新应用"},
            ),
        ),
        SimpleNamespace(
            event="ToolCallCompleted",
            agent_name="Web Agent",
            tool=SimpleNamespace(
                tool_name="search",
                tool_args={"query": "2026 医疗 AI 最新应用"},
                result="搜索结果：AI 辅助诊断",
            ),
            content=None,
        ),
        SimpleNamespace(
            event="RunContent",
            agent_name="Web Agent",
            run_id="web-run",
            content="Web Agent 已整理来源",
        ),
        SimpleNamespace(
            event="RunCompleted",
            agent_name="Web Agent",
            run_id="web-run",
            content="Web Agent 已整理来源",
        ),
        SimpleNamespace(
            event="TeamToolCallStarted",
            team_name="Research Team",
            tool=SimpleNamespace(
                tool_name="delegate_task_to_member",
                tool_args={
                    "member_id": "research-agent",
                    "task": "分析搜索结果并提炼趋势",
                },
            ),
        ),
        SimpleNamespace(event="RunStarted", agent_name="Research Agent"),
        SimpleNamespace(
            event="RunContent",
            agent_name="Research Agent",
            run_id="research-run",
            content="正在分析趋势",
        ),
        SimpleNamespace(
            event="RunCompleted",
            agent_name="Research Agent",
            run_id="research-run",
            content="Research Agent 已完成趋势分析",
        ),
        SimpleNamespace(
            event="TeamRunContent",
            team_name="Research Team",
            content="最终汇总内容",
        ),
    ]

    run_path(str(SCRIPT_PATH), run_name="__main__")

    output = capsys.readouterr().out
    assert "Leader 委派" in output
    assert "搜索医疗 AI 的最新应用" in output
    assert "Web Agent 开始执行" in output
    assert "准备查询\nWeb Agent 调用工具" in output
    assert "2026 医疗 AI 最新应用" in output
    assert "搜索结果：AI 辅助诊断" in output
    assert "Web Agent 已整理来源" in output
    assert "Research Agent 开始执行" in output
    assert "Research Agent 实时响应" in output
    assert "正在分析趋势" in output
    assert "Team 最终汇总" in output
    assert "最终汇总内容" in output


def test_team_process_prints_run_errors_without_empty_summary(monkeypatch, capsys):
    install_fake_dependencies(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-test-key")
    FakeTeam.events = [
        SimpleNamespace(
            event="RunError",
            agent_name="Research Agent",
            content="member timeout",
        ),
        SimpleNamespace(event="TeamRunError", content="team timeout"),
    ]

    run_path(str(SCRIPT_PATH), run_name="__main__")

    output = capsys.readouterr().out
    assert "Research Agent 执行失败" in output
    assert "member timeout" in output
    assert "Team 执行失败" in output
    assert "team timeout" in output
    assert "Team 最终汇总" not in output


def test_failed_agent_tool_is_not_printed_as_a_success(monkeypatch, capsys):
    install_fake_dependencies(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-test-key")
    failed_tool = SimpleNamespace(
        tool_name="web_search_using_tavily",
        tool_args={"query": "医疗 AI"},
        tool_call_error=True,
        result="search timeout",
    )
    FakeTeam.events = [
        SimpleNamespace(
            event="ToolCallCompleted",
            agent_name="Web Agent",
            tool=failed_tool,
            content=None,
        ),
        SimpleNamespace(
            event="ToolCallError",
            agent_name="Web Agent",
            tool=failed_tool,
            error="search timeout",
        ),
    ]

    run_path(str(SCRIPT_PATH), run_name="__main__")

    output = capsys.readouterr().out
    assert "Web Agent 工具结果" not in output
    assert "Web Agent 工具调用失败" in output
    assert output.count("search timeout") == 1


def test_team_tool_result_is_printed_when_no_member_run_was_created(monkeypatch, capsys):
    install_fake_dependencies(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-test-key")
    FakeTeam.events = [
        SimpleNamespace(
            event="TeamToolCallCompleted",
            team_name="Research Team",
            content=None,
            tool=SimpleNamespace(
                tool_name="delegate_task_to_member",
                tool_args={"member_id": "missing-agent", "task": "分析资料"},
                tool_call_error=False,
                child_run_id=None,
                result="Member with ID missing-agent not found",
            ),
        )
    ]

    run_path(str(SCRIPT_PATH), run_name="__main__")

    output = capsys.readouterr().out
    assert "Team 工具结果" in output
    assert "Member with ID missing-agent not found" in output


def test_missing_tavily_key_fails_before_team_run(monkeypatch):
    install_fake_dependencies(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        run_path(str(SCRIPT_PATH), run_name="__main__")

    assert FakeTeam.instances == []

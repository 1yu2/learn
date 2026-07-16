"""
示例 1：研究 Team 的事件流
功能：搜索信息 + 深度分析 + 成员过程可视化
"""

import json
from os import getenv
from typing import Any

from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from agno.run.agent import RunEvent
from agno.run.team import TeamRunEvent
from agno.team import Team
from agno.team.mode import TeamMode
from agno.tools.tavily import TavilyTools
from dotenv import load_dotenv
from httpx import Timeout
from rich.console import Console


def create_deepseek_model(api_key: str) -> DeepSeek:
    return DeepSeek(
        id=getenv("DEEPSEEK_MODEL_ID", "deepseek-v4-flash"),
        api_key=api_key,
        use_thinking=False,
        timeout=Timeout(connect=15.0, read=180.0, write=60.0, pool=60.0),
        max_retries=1,
    )


def format_event_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def print_tool_call(console: Console, source: str, tool: Any) -> None:
    console.print(
        f"[bold yellow]{source} 调用工具[/bold yellow]: {getattr(tool, 'tool_name', 'unknown')}"
    )
    tool_args = getattr(tool, "tool_args", None)
    if tool_args:
        console.print("参数:")
        console.print(format_event_value(tool_args), markup=False, highlight=False)


def print_team_process(team: Team, query: str) -> None:
    console = Console()
    team_content: list[str] = []
    streamed_agent_runs: set[str] = set()
    open_stream_run: str | None = None
    team_failed = False

    console.rule("[bold cyan]用户请求[/bold cyan]")
    console.print(query, markup=False)

    for event in team.run(query, stream=True, stream_events=True):
        event_name = getattr(event, "event", "")
        tool = getattr(event, "tool", None)
        event_run_key = str(getattr(event, "run_id", None) or getattr(event, "agent_name", "Agent"))
        if open_stream_run is not None and (
            event_name != RunEvent.run_content.value or event_run_key != open_stream_run
        ):
            console.print()
            open_stream_run = None

        if event_name == TeamRunEvent.tool_call_started.value and tool is not None:
            tool_name = getattr(tool, "tool_name", "")
            tool_args = getattr(tool, "tool_args", None) or {}
            if tool_name == "delegate_task_to_member":
                console.rule("[bold cyan]Leader 委派[/bold cyan]")
                console.print(f"成员: {tool_args.get('member_id', 'unknown')}")
                console.print(f"任务: {tool_args.get('task', '')}", markup=False, highlight=False)
            else:
                print_tool_call(console, "Team", tool)

        elif event_name == TeamRunEvent.tool_call_completed.value and tool is not None:
            tool_name = getattr(tool, "tool_name", "")
            tool_failed = bool(getattr(tool, "tool_call_error", False))
            delegated_run_created = bool(getattr(tool, "child_run_id", None))
            result = getattr(tool, "result", None) or getattr(event, "content", None)
            if (
                result
                and not tool_failed
                and not (tool_name == "delegate_task_to_member" and delegated_run_created)
            ):
                console.print("[bold green]Team 工具结果[/bold green]:")
                console.print(format_event_value(result), markup=False, highlight=False)

        elif event_name == RunEvent.run_started.value:
            agent_name = getattr(event, "agent_name", "Agent") or "Agent"
            console.rule(f"[bold magenta]{agent_name} 开始执行[/bold magenta]")

        elif event_name == RunEvent.run_content.value:
            agent_name = getattr(event, "agent_name", "Agent") or "Agent"
            run_key = str(getattr(event, "run_id", None) or agent_name)
            content = getattr(event, "content", None)
            if content:
                if run_key not in streamed_agent_runs:
                    console.print(f"[bold magenta]{agent_name} 实时响应[/bold magenta]:")
                    streamed_agent_runs.add(run_key)
                console.print(
                    format_event_value(content),
                    end="" if isinstance(content, str) else "\n",
                    markup=False,
                    highlight=False,
                )
                if isinstance(content, str):
                    open_stream_run = run_key

        elif event_name == RunEvent.tool_call_started.value and tool is not None:
            agent_name = getattr(event, "agent_name", "Agent") or "Agent"
            print_tool_call(console, agent_name, tool)

        elif event_name == RunEvent.tool_call_completed.value and tool is not None:
            agent_name = getattr(event, "agent_name", "Agent") or "Agent"
            result = getattr(tool, "result", None) or getattr(event, "content", None)
            if result and not getattr(tool, "tool_call_error", False):
                console.print(f"[bold green]{agent_name} 工具结果[/bold green]:")
                console.print(format_event_value(result), markup=False, highlight=False)

        elif event_name == RunEvent.tool_call_error.value:
            agent_name = getattr(event, "agent_name", "Agent") or "Agent"
            error = getattr(event, "error", None) or getattr(tool, "result", None)
            console.print(f"[bold red]{agent_name} 工具调用失败[/bold red]: {error}")

        elif event_name == RunEvent.run_completed.value:
            agent_name = getattr(event, "agent_name", "Agent") or "Agent"
            run_key = str(getattr(event, "run_id", None) or agent_name)
            content = getattr(event, "content", None)
            if run_key in streamed_agent_runs:
                console.print(f"[bold magenta]{agent_name} 完成[/bold magenta]")
            else:
                console.print(f"[bold magenta]{agent_name} 完成[/bold magenta]:")
            if content and run_key not in streamed_agent_runs:
                console.print(format_event_value(content), markup=False, highlight=False)

        elif event_name == RunEvent.run_error.value:
            agent_name = getattr(event, "agent_name", "Agent") or "Agent"
            error = getattr(event, "content", None) or "未知错误"
            console.print(f"[bold red]{agent_name} 执行失败[/bold red]:")
            console.print(format_event_value(error), markup=False, highlight=False)

        elif event_name == TeamRunEvent.tool_call_error.value:
            error = getattr(event, "error", None) or getattr(tool, "result", None)
            console.print(f"[bold red]Team 工具调用失败[/bold red]: {error}")

        elif event_name == TeamRunEvent.run_content.value:
            content = getattr(event, "content", None)
            if isinstance(content, str):
                team_content.append(content)

        elif event_name == TeamRunEvent.run_completed.value:
            content = getattr(event, "content", None)
            if isinstance(content, str):
                team_content = [content]

        elif event_name == TeamRunEvent.run_error.value:
            team_failed = True
            error = getattr(event, "content", None) or "未知错误"
            console.rule("[bold red]Team 执行失败[/bold red]")
            console.print(format_event_value(error), markup=False, highlight=False)

    if open_stream_run is not None:
        console.print()

    if not team_failed:
        console.rule("[bold blue]Team 最终汇总[/bold blue]")
        console.print("".join(team_content), markup=False, highlight=False)


def main() -> None:
    load_dotenv()

    api_key = getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 DeepSeek API Key，请设置 DEEPSEEK_API_KEY。")

    tavily_api_key = getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise RuntimeError("缺少 Tavily API Key，请设置 TAVILY_API_KEY。")

    web_agent = Agent(
        id="web-agent",
        name="Web Agent",
        role="搜索最新网络信息并保留来源链接",
        model=create_deepseek_model(api_key),
        tools=[
            TavilyTools(
                api_key=tavily_api_key,
                search_depth="basic",
                max_tokens=3000,
            )
        ],
        instructions=[
            "搜索与任务直接相关的最新信息。",
            "在结论中保留可核验的信息来源链接。",
        ],
        markdown=True,
    )

    research_agent = Agent(
        id="research-agent",
        name="Research Agent",
        role="分析搜索结果并提炼关键趋势",
        model=create_deepseek_model(api_key),
        instructions=[
            "基于团队提供的资料进行分析，不要编造来源。",
            "区分已经落地的应用、研究进展和未来预测。",
        ],
        markdown=True,
    )

    research_team = Team(
        name="Research Team",
        members=[web_agent, research_agent],
        model=create_deepseek_model(api_key),
        mode=TeamMode.coordinate,
        instructions=[
            "必须先委派 Web Agent 搜索最新资料，不能跳过该成员。",
            "必须再把 Web Agent 的结果交给 Research Agent 分析，不能跳过该成员。",
            "最后汇总团队结论并保留来源链接。",
        ],
        markdown=True,
        stream=True,
        stream_events=True,
        stream_member_events=True,
        debug_mode=False,
    )

    print_team_process(research_team, "分析下2026年企业级问数和本体论实际落地方案")


if __name__ == "__main__":
    main()

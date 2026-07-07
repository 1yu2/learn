"""Stage 2: inspect streaming events from an AgentScope reply."""

from __future__ import annotations

import os


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Set {name} before running this example.")
    return value


def main() -> None:
    require_env("DASHSCOPE_API_KEY")

    from agentscope.agent import Agent
    from agentscope.message import Msg, append_event
    from agentscope.model import DashScopeChatModel

    agent = Agent(
        name="event-inspector",
        system_prompt="Explain your work briefly.",
        model=DashScopeChatModel(model_name="qwen-max"),
    )

    final_message = Msg("assistant", [])
    for event in agent.reply_stream("列出学习 AgentScope 的前三步。"):
        print(type(event).__name__, event)
        final_message = append_event(final_message, event)

    print("\nFinal message:")
    print(final_message)


if __name__ == "__main__":
    main()

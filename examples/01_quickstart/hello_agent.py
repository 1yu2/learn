"""Stage 1: run a minimal AgentScope agent.

Set DASHSCOPE_API_KEY before running this script. The imports live inside
main() so the repository tests can compile examples without installing every
runtime dependency.
"""

from __future__ import annotations

import os


def require_env(name: str) -> str:
    """Return a required environment variable or raise a clear error."""

    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Set {name} before running this example.")
    return value


def main() -> None:
    require_env("DASHSCOPE_API_KEY")

    from agentscope.agent import Agent
    from agentscope.model import DashScopeChatModel

    model = DashScopeChatModel(model_name="qwen-max")
    agent = Agent(
        name="assistant",
        system_prompt="You are a concise AgentScope learning assistant.",
        model=model,
    )

    response = agent.reply("用一句话介绍 AgentScope。")
    print(response)


if __name__ == "__main__":
    main()

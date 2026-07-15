"""Stage 1: run a minimal AgentScope agent.

Set DASHSCOPE_API_KEY before running this script. The imports live inside
main() so the repository tests can compile examples without installing every
runtime dependency.
"""

from __future__ import annotations

import asyncio

from agentscope_learn import ConfigurationError, Settings


async def main() -> None:
    settings = Settings.from_env()
    if settings.model_provider != "dashscope":
        raise ConfigurationError(
            "hello_agent.py uses DashScopeChatModel; set MODEL_PROVIDER=dashscope"
        )
    api_key = settings.require_model_credentials()

    from agentscope.agent import Agent
    from agentscope.credential import DashScopeCredential
    from agentscope.message import Msg, TextBlock
    from agentscope.model import DashScopeChatModel

    credential = DashScopeCredential(api_key=api_key)
    model = DashScopeChatModel(credential=credential, model=settings.model_name)
    agent = Agent(
        name="assistant",
        system_prompt="You are a concise AgentScope learning assistant.",
        model=model,
    )

    response = await agent.reply(
        Msg(
            name="user",
            role="user",
            content=[TextBlock(text="用一句话介绍 AgentScope。")],
        )
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(main())

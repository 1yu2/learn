"""Stage 2: inspect streaming events from an AgentScope reply."""

from __future__ import annotations

import asyncio

from agentscope_learn import ConfigurationError, Settings


async def main() -> None:
    settings = Settings.from_env()
    if settings.model_provider != "dashscope":
        raise ConfigurationError(
            "inspect_events.py uses DashScopeChatModel; set MODEL_PROVIDER=dashscope"
        )
    api_key = settings.require_model_credentials()

    from agentscope.agent import Agent
    from agentscope.credential import DashScopeCredential
    from agentscope.message import Msg, TextBlock
    from agentscope.model import DashScopeChatModel

    credential = DashScopeCredential(api_key=api_key)

    agent = Agent(
        name="event-inspector",
        system_prompt="Explain your work briefly.",
        model=DashScopeChatModel(credential=credential, model=settings.model_name),
    )

    final_message = Msg(name="assistant", role="assistant", content=[])
    async for event in agent.reply_stream(
        Msg(
            name="user",
            role="user",
            content=[TextBlock(text="列出学习 AgentScope 的前三步。")],
        )
    ):
        print(type(event).__name__, event)
        final_message.append_event(event)

    print("\nFinal message:")
    print(final_message)


if __name__ == "__main__":
    asyncio.run(main())

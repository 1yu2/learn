from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_EXAMPLES = (
    ROOT / "examples/01_quickstart/hello_agent.py",
    ROOT / "examples/02_message_event/inspect_events.py",
    ROOT / "examples/03_agent_tool/model_structured_output.py",
)


def test_model_examples_use_shared_settings_for_credentials_and_model() -> None:
    for path in MODEL_EXAMPLES:
        source = path.read_text(encoding="utf-8")

        assert "Settings.from_env()" in source
        assert "settings.require_model_credentials()" in source
        assert "model=settings.model_name" in source
        assert "os.getenv" not in source
        assert "require_env" not in source


def test_async_agent_examples_await_agent_operations() -> None:
    quickstart = (ROOT / "examples/01_quickstart/hello_agent.py").read_text(
        encoding="utf-8"
    )
    events = (ROOT / "examples/02_message_event/inspect_events.py").read_text(
        encoding="utf-8"
    )

    assert "async def main()" in quickstart
    assert "response = await agent.reply" in quickstart
    assert 'name="user"' in quickstart
    assert 'TextBlock(text="用一句话介绍 AgentScope。")' in quickstart
    assert "asyncio.run(main())" in quickstart
    assert "async def main()" in events
    assert "async for event in agent.reply_stream" in events
    assert 'TextBlock(text="列出学习 AgentScope 的前三步。")' in events
    assert 'Msg(name="assistant", role="assistant", content=[])' in events
    assert "from agentscope.message import Msg, TextBlock, append_event" not in events
    assert "final_message.append_event(event)" in events
    assert "asyncio.run(main())" in events

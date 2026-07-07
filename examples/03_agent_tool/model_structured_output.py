"""Stage 3: request structured output from an AgentScope chat model."""

from __future__ import annotations

import os

from pydantic import BaseModel, Field


class BookSummary(BaseModel):
    """A small schema for model-validated output."""

    title: str = Field(description="Book title")
    summary: str = Field(description="One paragraph summary")
    keywords: list[str] = Field(description="Three to five keywords")


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Set {name} before running this example.")
    return value


def main() -> None:
    require_env("DASHSCOPE_API_KEY")

    from agentscope.message import Msg
    from agentscope.model import DashScopeChatModel

    model = DashScopeChatModel(model_name="qwen-max")
    result = model.generate_structured_output(
        messages=[
            Msg(
                "user",
                "Summarize The Pragmatic Programmer for an engineering learner.",
            ),
        ],
        schema=BookSummary,
    )
    print(result)


if __name__ == "__main__":
    main()

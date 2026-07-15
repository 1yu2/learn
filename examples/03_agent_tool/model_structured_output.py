"""Stage 3: request structured output from an AgentScope chat model."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentscope_learn import ConfigurationError, Settings


class BookSummary(BaseModel):
    """A small schema for model-validated output."""

    title: str = Field(description="Book title")
    summary: str = Field(description="One paragraph summary")
    keywords: list[str] = Field(description="Three to five keywords")


def main() -> None:
    settings = Settings.from_env()
    if settings.model_provider != "dashscope":
        raise ConfigurationError(
            "model_structured_output.py uses DashScopeChatModel; "
            "set MODEL_PROVIDER=dashscope"
        )
    api_key = settings.require_model_credentials()

    from agentscope.credential import DashScopeCredential
    from agentscope.message import Msg
    from agentscope.model import DashScopeChatModel

    credential = DashScopeCredential(api_key=api_key)
    model = DashScopeChatModel(credential=credential, model=settings.model_name)
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

"""Structured learning-plan metadata used by tests and examples."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LearningStage:
    """A single stage in the AgentScope learning path."""

    number: int
    slug: str
    title: str
    example_dir: str | None
    doc_links: tuple[str, ...]


LEARNING_STAGES: tuple[LearningStage, ...] = (
    LearningStage(
        number=1,
        slug="quickstart",
        title="快速运行第一个 Agent",
        example_dir="examples/01_quickstart",
        doc_links=(
            "https://docs.agentscope.io/versions/2.0.3/en/quickstart",
        ),
    ),
    LearningStage(
        number=2,
        slug="message-event",
        title="Message 与 Event",
        example_dir="examples/02_message_event",
        doc_links=(
            "https://docs.agentscope.io/versions/2.0.3/en/building-blocks/message-and-event",
        ),
    ),
    LearningStage(
        number=3,
        slug="model-structured-output",
        title="Model 层与结构化输出",
        example_dir="examples/03_agent_tool",
        doc_links=(
            "https://docs.agentscope.io/versions/2.0.3/en/building-blocks/model",
        ),
    ),
    LearningStage(
        number=4,
        slug="agent-tool-toolkit",
        title="Agent、Tool 与 Toolkit",
        example_dir="examples/03_agent_tool",
        doc_links=(
            "https://docs.agentscope.io/versions/2.0.3/en/building-blocks/agent",
            "https://docs.agentscope.io/versions/2.0.3/en/building-blocks/tool",
        ),
    ),
    LearningStage(
        number=5,
        slug="permission-human-in-the-loop",
        title="权限系统与人类确认",
        example_dir="examples/04_permission_plan",
        doc_links=(
            "https://docs.agentscope.io/versions/2.0.3/en/building-blocks/permission-system",
        ),
    ),
    LearningStage(
        number=6,
        slug="plan-tools",
        title="Plan 工具与复杂任务拆解",
        example_dir="examples/04_permission_plan",
        doc_links=(
            "https://docs.agentscope.io/versions/2.0.3/en/building-blocks/plan",
        ),
    ),
    LearningStage(
        number=7,
        slug="context-workspace-middleware",
        title="Context、Workspace 与 Middleware",
        example_dir="examples/05_middleware_context",
        doc_links=(
            "https://docs.agentscope.io/versions/2.0.3/en/building-blocks/context",
            "https://docs.agentscope.io/versions/2.0.3/en/building-blocks/workspace",
            "https://docs.agentscope.io/versions/2.0.3/en/building-blocks/middleware",
        ),
    ),
    LearningStage(
        number=8,
        slug="rag-long-term-memory",
        title="RAG 与 Long-Term Memory",
        example_dir="examples/06_rag_memory",
        doc_links=(
            "https://docs.agentscope.io/versions/2.0.3/en/building-blocks/rag",
            "https://docs.agentscope.io/versions/2.0.3/en/building-blocks/long-term-memory",
        ),
    ),
    LearningStage(
        number=9,
        slug="service-team-deploy",
        title="Agent as Service、Agent Team 与部署",
        example_dir="examples/07_service_team",
        doc_links=(
            "https://docs.agentscope.io/versions/2.0.3/en/deploy/agent-service",
            "https://docs.agentscope.io/versions/2.0.3/en/deploy/agent-team",
        ),
    ),
    LearningStage(
        number=10,
        slug="personal-research-agent",
        title="复盘与个人项目",
        example_dir="projects/personal_research_agent",
        doc_links=(
            "https://docs.agentscope.io/api-reference",
        ),
    ),
)


def repository_sections() -> dict[str, list[str]]:
    """Return the canonical top-level learning sections for this repository."""

    return {
        "examples": [
            "01_quickstart",
            "02_message_event",
            "03_agent_tool",
            "04_permission_plan",
            "05_middleware_context",
            "06_rag_memory",
            "07_service_team",
        ],
        "notes": [
            "concepts.md",
            "api-differences.md",
            "troubleshooting.md",
        ],
        "projects": [
            "personal_research_agent",
        ],
    }

from pathlib import Path

from agentscope_learn.learning_plan import LEARNING_STAGES, repository_sections


ROOT = Path(__file__).resolve().parents[1]


def test_learning_plan_has_ten_ordered_stages() -> None:
    assert [stage.number for stage in LEARNING_STAGES] == list(range(1, 11))
    assert LEARNING_STAGES[0].slug == "quickstart"
    assert LEARNING_STAGES[-1].slug == "personal-research-agent"


def test_repository_sections_match_expected_learning_layout() -> None:
    sections = repository_sections()

    assert sections["examples"] == [
        "01_quickstart",
        "02_message_event",
        "03_agent_tool",
        "04_permission_plan",
        "05_middleware_context",
        "06_rag_memory",
        "07_service_team",
    ]
    assert sections["notes"] == [
        "concepts.md",
        "api-differences.md",
        "troubleshooting.md",
    ]
    assert sections["projects"] == ["personal_research_agent"]


def test_scaffold_paths_exist_for_readme_plan() -> None:
    for directory in [
        "examples/01_quickstart",
        "examples/02_message_event",
        "examples/03_agent_tool",
        "examples/04_permission_plan",
        "examples/05_middleware_context",
        "examples/06_rag_memory",
        "examples/07_service_team",
        "notes",
        "projects/personal_research_agent",
    ]:
        assert (ROOT / directory).is_dir()


def test_example_scripts_are_python_compileable() -> None:
    scripts = sorted((ROOT / "examples").glob("**/*.py"))

    assert scripts

    for script in scripts:
        source = script.read_text(encoding="utf-8")
        compile(source, str(script), "exec")

from pathlib import Path
from runpy import run_path

import pytest
from agno.team import Team
from agno.team.mode import TeamMode


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples" / "05_teams"


@pytest.mark.parametrize(
    ("filename", "expected_mode", "expected_runner"),
    [
        ("01_basic_team.py", TeamMode.coordinate, "run"),
        ("02_team_member.py", TeamMode.coordinate, "print_response"),
        ("03_math.py", TeamMode.route, "print_response"),
        ("04_collaborate.py", TeamMode.broadcast, "aprint_response"),
        ("05_coordinate.py", TeamMode.coordinate, "print_response"),
        ("06_p1.py", TeamMode.coordinate, "print_response"),
        ("07_p2.py", TeamMode.coordinate, "print_response"),
    ],
)
def test_examples_use_explicit_official_team_modes(
    monkeypatch, filename, expected_mode, expected_runner
):
    calls = []

    def fake_run(team, input, **kwargs):
        calls.append(("run", team, input, kwargs))
        return iter(())

    def fake_print_response(team, input, **kwargs):
        calls.append(("print_response", team, input, kwargs))

    async def fake_aprint_response(team, input, **kwargs):
        calls.append(("aprint_response", team, input, kwargs))

    monkeypatch.setattr(Team, "run", fake_run)
    monkeypatch.setattr(Team, "print_response", fake_print_response)
    monkeypatch.setattr(Team, "aprint_response", fake_aprint_response)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-test-key")

    path = EXAMPLES_DIR / filename
    run_path(str(path), run_name="__main__")

    assert len(calls) == 1
    runner, team, _, _ = calls[0]
    assert runner == expected_runner
    assert team.mode == expected_mode
    assert f"mode=TeamMode.{expected_mode.value}" in path.read_text()


def test_route_example_sends_one_operation_to_one_member(monkeypatch):
    calls = []

    def fake_print_response(team, input, **kwargs):
        calls.append((team, input, kwargs))

    monkeypatch.setattr(Team, "print_response", fake_print_response)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")

    run_path(str(EXAMPLES_DIR / "03_math.py"), run_name="__main__")

    assert len(calls) == 1
    _, message, _ = calls[0]
    assert "+" in message
    assert "乘" not in message


def test_deepseek_examples_do_not_fallback_to_openai_keys_or_prompt_switches():
    for path in sorted(EXAMPLES_DIR.glob("*.py")):
        source = path.read_text()
        assert "OPENAI_API_KEY" not in source, path.name
        assert "/nothink" not in source, path.name

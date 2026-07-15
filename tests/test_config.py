from pathlib import Path

import pytest

from agentscope_learn.config import ConfigurationError, Settings


def test_settings_use_safe_defaults_without_environment(tmp_path: Path) -> None:
    settings = Settings.from_env({}, dotenv_path=tmp_path / "missing.env")

    assert settings.model_provider == "dashscope"
    assert settings.model_name == "qwen-plus"
    assert settings.log_level == "INFO"
    assert settings.workspace == Path.cwd() / ".workspace"
    assert settings.enable_tracing is False


def test_explicit_environment_values_override_dotenv(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "MODEL_NAME=from-dotenv\n"
        "MODEL_PROVIDER=openai\n"
        "OPENAI_API_KEY=dotenv-key\n",
        encoding="utf-8",
    )

    settings = Settings.from_env(
        {"MODEL_NAME": "from-environment"},
        dotenv_path=dotenv,
    )

    assert settings.model_name == "from-environment"
    assert settings.model_provider == "openai"
    assert settings.api_key == "dotenv-key"


def test_default_dotenv_lookup_walks_up_from_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    examples = project / "examples"
    examples.mkdir(parents=True)
    (project / ".env").write_text(
        "DASHSCOPE_API_KEY=parent-key\nMODEL_NAME=parent-model\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(examples)

    settings = Settings.from_env({})

    assert settings.api_key == "parent-key"
    assert settings.model_name == "parent-model"


def test_relative_paths_are_resolved_from_dotenv_directory(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("AGENTSCOPE_WORKSPACE=runtime\n", encoding="utf-8")

    settings = Settings.from_env({}, dotenv_path=dotenv)

    assert settings.workspace == tmp_path / "runtime"


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("MODEL_PROVIDER", "unknown", "MODEL_PROVIDER"),
        ("AGENTSCOPE_LOG_LEVEL", "verbose", "AGENTSCOPE_LOG_LEVEL"),
        ("ENABLE_TRACING", "sometimes", "ENABLE_TRACING"),
    ],
)
def test_invalid_values_raise_configuration_error(
    name: str, value: str, message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        Settings.from_env({name: value})


def test_malformed_dotenv_line_is_rejected(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("not-an-assignment\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        Settings.from_env({}, dotenv_path=dotenv)


def test_model_credentials_are_checked_only_when_requested(tmp_path: Path) -> None:
    settings = Settings.from_env({}, dotenv_path=tmp_path / "missing.env")

    with pytest.raises(ConfigurationError, match="DASHSCOPE_API_KEY"):
        settings.require_model_credentials()


def test_provider_specific_credentials_are_checked(tmp_path: Path) -> None:
    dotenv = tmp_path / "missing.env"
    openai = Settings.from_env({"MODEL_PROVIDER": "openai"}, dotenv_path=dotenv)
    ollama = Settings.from_env({"MODEL_PROVIDER": "ollama"}, dotenv_path=dotenv)

    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        openai.require_model_credentials()

    assert ollama.require_model_credentials() == "http://localhost:11434/v1"


def test_redacted_summary_never_contains_the_secret() -> None:
    secret = "sentinel-secret-value"
    settings = Settings.from_env({"DASHSCOPE_API_KEY": secret})

    summary = settings.redacted_summary()

    assert secret not in str(summary)
    assert summary["api_key"] == "***configured***"

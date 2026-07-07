"""Runtime configuration shared by examples and future AgentOS services."""

from dataclasses import dataclass
from os import getenv
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    """Configuration values that should stay consistent across examples."""

    model_id: str
    data_dir: Path


def load_runtime_config() -> RuntimeConfig:
    """Load configuration from environment variables with local defaults."""

    data_dir = Path(getenv("AGNO_LEARN_DATA_DIR", "tmp")).expanduser()
    return RuntimeConfig(
        model_id=getenv("AGNO_MODEL_ID", "openai:gpt-5-mini"),
        data_dir=data_dir,
    )

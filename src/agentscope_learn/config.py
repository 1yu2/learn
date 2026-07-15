"""Environment-backed settings for the AgentScope learning workspace."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping


class ConfigurationError(ValueError):
    """Raised when local configuration cannot support the requested operation."""


_PROVIDERS = {"dashscope", "openai", "ollama"}
_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}
_BOOLEAN_VALUES = {
    "1": True,
    "true": True,
    "yes": True,
    "on": True,
    "0": False,
    "false": False,
    "no": False,
    "off": False,
}
_KEY_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Malformed dotenv line {line_number}: expected KEY=VALUE")
        key, value = (part.strip() for part in line.split("=", 1))
        if not _KEY_NAME.fullmatch(key):
            raise ValueError(f"Malformed dotenv line {line_number}: invalid key")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _find_dotenv(start: Path) -> Path | None:
    """Find the nearest project .env while running from a subdirectory."""

    current = start.resolve()
    for directory in (current, *current.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def _value(values: Mapping[str, str], name: str, default: str) -> str:
    return values.get(name, default).strip()


def _path_value(values: Mapping[str, str], name: str, default: str, base: Path) -> Path:
    value = Path(_value(values, name, default)).expanduser()
    return value if value.is_absolute() else (base / value).resolve()


def _bool_value(values: Mapping[str, str], name: str, default: bool) -> bool:
    raw = values.get(name, str(default)).strip().lower()
    try:
        return _BOOLEAN_VALUES[raw]
    except KeyError as exc:
        allowed = ", ".join(sorted(_BOOLEAN_VALUES))
        raise ConfigurationError(
            f"{name} must be one of: {allowed}; received {raw!r}"
        ) from exc


@dataclass(frozen=True)
class Settings:
    """Validated settings shared by examples and the capstone project."""

    model_provider: str
    api_key: str
    model_name: str
    model_base_url: str
    log_level: str
    workspace: Path
    rag_data_dir: Path
    rag_collection: str
    enable_tracing: bool

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        dotenv_path: Path | None = None,
    ) -> "Settings":
        """Build settings without contacting a model provider."""

        explicit = dict(os.environ if env is None else env)
        path = (
            Path(dotenv_path)
            if dotenv_path is not None
            else _find_dotenv(Path.cwd())
        )
        dotenv_values = _parse_dotenv(path) if path is not None and path.is_file() else {}
        values = {**dotenv_values, **explicit}
        base = path.parent.resolve() if path is not None and path.is_file() else Path.cwd().resolve()

        provider = _value(values, "MODEL_PROVIDER", "dashscope").lower()
        if provider not in _PROVIDERS:
            raise ConfigurationError(
                f"MODEL_PROVIDER must be one of {sorted(_PROVIDERS)}; received {provider!r}"
            )

        log_level = _value(values, "AGENTSCOPE_LOG_LEVEL", "INFO").upper()
        if log_level not in _LOG_LEVELS:
            raise ConfigurationError(
                f"AGENTSCOPE_LOG_LEVEL must be one of {sorted(_LOG_LEVELS)}; received {log_level!r}"
            )

        base_url = _value(values, "MODEL_BASE_URL", "")
        if provider == "ollama" and not base_url:
            base_url = "http://localhost:11434/v1"

        api_key_name = {
            "dashscope": "DASHSCOPE_API_KEY",
            "openai": "OPENAI_API_KEY",
        }.get(provider)

        return cls(
            model_provider=provider,
            api_key=values.get(api_key_name, "").strip() if api_key_name else "",
            model_name=_value(values, "MODEL_NAME", "qwen-plus"),
            model_base_url=base_url,
            log_level=log_level,
            workspace=_path_value(values, "AGENTSCOPE_WORKSPACE", "./.workspace", base),
            rag_data_dir=_path_value(values, "RAG_DATA_DIR", "./data/knowledge", base),
            rag_collection=_value(values, "RAG_COLLECTION", "agentscope_learning"),
            enable_tracing=_bool_value(values, "ENABLE_TRACING", False),
        )

    def require_model_credentials(self) -> str:
        """Return the configured credential or raise an actionable error."""

        if self.model_provider == "ollama":
            return self.model_base_url
        if self.api_key:
            return self.api_key
        variable = {
            "dashscope": "DASHSCOPE_API_KEY",
            "openai": "OPENAI_API_KEY",
        }[self.model_provider]
        raise ConfigurationError(
            f"{variable} is required for provider {self.model_provider!r}; "
            "set it in .env or the process environment before model calls"
        )

    def redacted_summary(self) -> dict[str, object]:
        """Return settings suitable for logs without exposing credentials."""

        summary = asdict(self)
        summary["workspace"] = str(self.workspace)
        summary["rag_data_dir"] = str(self.rag_data_dir)
        summary["api_key"] = "***configured***" if self.api_key else "<missing>"
        return summary

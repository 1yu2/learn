"""Project path helpers used by examples."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
NOTES_DIR = PROJECT_ROOT / "notes"


def ensure_data_dir(path: Path) -> Path:
    """Create and return a local runtime data directory."""

    path.mkdir(parents=True, exist_ok=True)
    return path

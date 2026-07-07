"""Stage 8: local RAG exercise scaffold."""

from __future__ import annotations

from pathlib import Path


SAMPLE_KNOWLEDGE = Path(__file__).with_name("sample_knowledge.md")


def load_sample_knowledge() -> str:
    return SAMPLE_KNOWLEDGE.read_text(encoding="utf-8")


def main() -> None:
    print(load_sample_knowledge())


if __name__ == "__main__":
    main()

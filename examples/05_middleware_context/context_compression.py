"""Stage 7: context and middleware exercise scaffold."""

from __future__ import annotations


def compression_experiment_messages() -> list[str]:
    return [
        "Create a long conversation.",
        "Configure ContextConfig.",
        "Observe when compression or offloading happens.",
    ]


def main() -> None:
    for message in compression_experiment_messages():
        print(message)


if __name__ == "__main__":
    main()

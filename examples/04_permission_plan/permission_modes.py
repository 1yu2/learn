"""Stage 5: placeholder script for comparing permission modes.

Run this after installing AgentScope and reading the permission-system docs.
"""

from __future__ import annotations


PERMISSION_MODES = ("DEFAULT", "EXPLORE", "ACCEPT_EDITS", "BYPASS", "DONT_ASK")


def describe_modes() -> list[str]:
    return [f"{mode}: inspect behavior in a controlled workspace" for mode in PERMISSION_MODES]


def main() -> None:
    for line in describe_modes():
        print(line)


if __name__ == "__main__":
    main()

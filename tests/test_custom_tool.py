from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "examples" / "03_agent_tool" / "custom_tool.py"


def load_custom_tool_module():
    spec = spec_from_file_location("custom_tool_example", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calculate_expression_handles_basic_arithmetic() -> None:
    module = load_custom_tool_module()

    assert module.calculate_expression("1 + 2 * 3") == 7.0
    assert module.calculate_expression("(10 - 4) / 2") == 3.0


def test_calculate_expression_rejects_unsupported_operators() -> None:
    module = load_custom_tool_module()

    with pytest.raises(ValueError, match="Unsupported expression"):
        module.calculate_expression("2 ** 8")

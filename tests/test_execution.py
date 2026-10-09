from __future__ import annotations

from typing import Any

import pytest

from kodelet_sdk import ExecutionOptions
from kodelet_sdk.execution import execution_args


@pytest.mark.parametrize(
    "options",
    [
        {"api_key": "secret"},
        {"allowedTools": None},
        {"maxTokens": 0},
        {"max_turns": -1},
        {"no_tools": "false"},
        {"allowed_tools": [""]},
        {"code_mode": "auto"},
        {"codeMode": None},
        {"code_mode": True},
    ],
)
def test_execution_options_reject_unknown_null_and_invalid_values(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ExecutionOptions.model_validate(options)


@pytest.mark.parametrize("key,mode", [
    ("code_mode", "off"), ("codeMode", "on"), ("code_mode", "only"),
])
def test_code_mode_validation_wire_and_cli_encoding(key: str, mode: str) -> None:
    options = ExecutionOptions.model_validate({key: mode})
    assert options.code_mode == mode
    assert options.to_wire() == {"codeMode": mode}
    assert execution_args(options) == [f"--code-mode={mode}"]
    assert ExecutionOptions.model_validate(options.to_wire()) == options


def test_execution_options_preserve_explicit_false_zero_and_empty_lists() -> None:
    options = ExecutionOptions(
        max_turns=0,
        no_tools=False,
        allowed_tools=[],
        allowed_commands=[],
        enableFSSearchTools=True,
    )
    assert options.to_wire() == {
        "maxTurns": 0,
        "noTools": False,
        "allowedTools": [],
        "allowedCommands": [],
        "enableFSSearchTools": True,
    }
    assert execution_args(options) == [
        "--max-turns=0",
        "--no-tools=false",
        "--allowed-tools=",
        "--allowed-commands=",
        "--enable-fs-search-tools=true",
    ]
    assert ExecutionOptions.model_validate(options.to_wire()) == options
    assert ExecutionOptions.model_validate({"enable_fs_search_tools": True}).to_wire() == {
        "enableFSSearchTools": True,
    }


def test_execution_options_return_independent_snapshots() -> None:
    options = ExecutionOptions(allowed_tools=["file_read", "grep_tool", "glob_tool"])
    snapshot = options.to_wire()
    snapshot["allowedTools"].append("bash")
    assert options.allowed_tools == ["file_read", "grep_tool", "glob_tool"]
    assert ExecutionOptions().to_wire() == {}
    assert execution_args(ExecutionOptions()) == []


def test_execution_args_escape_csv_values() -> None:
    options = ExecutionOptions(allowed_commands=['rg "a,b"', "git status"])
    assert execution_args(options) == ['--allowed-commands="rg ""a,b""","git status"']

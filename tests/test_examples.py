from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any

import pytest

from kodelet_sdk import create_test_harness

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path) -> Any:
    loader = SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def assert_executable(path: Path) -> None:
    mode = path.stat().st_mode
    text = path.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env -S uv run --script")
    assert 'dependencies = ["kodelet-sdk"]' in text
    assert 'kodelet-sdk = { path = "../..", editable = true }' in text
    assert mode & stat.S_IXUSR
    assert os.access(path, os.X_OK)


def assert_uv_script(path: Path) -> None:
    mode = path.stat().st_mode
    text = path.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env -S uv run --script")
    assert 'dependencies = ["kodelet-sdk"]' in text
    assert 'kodelet-sdk = { path = "../..", editable = true }' in text
    assert mode & stat.S_IXUSR
    assert os.access(path, os.X_OK)


def read_subprocess_frame(stdout) -> dict[str, Any]:
    header_lines: list[bytes] = []
    while True:
        line = stdout.readline()
        assert line != b""
        if line in (b"\r\n", b"\n"):
            break
        header_lines.append(line)
    header = b"".join(header_lines).decode("ascii")
    length = None
    for line in header.splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() == "content-length":
            length = int(value.strip())
    assert length is not None
    payload = stdout.read(length)
    assert len(payload) == length
    return json.loads(payload.decode("utf-8"))


def write_subprocess_frame(stdin, message: dict[str, Any]) -> None:
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    stdin.write(b"Content-Length: " + str(len(payload)).encode("ascii") + b"\r\n\r\n" + payload)
    stdin.flush()


def test_sdk_agent_examples_are_import_safe_and_executable() -> None:
    examples_dir = ROOT / "examples" / "sdk"
    example_names = [
        "basic-agent-session",
        "streaming-agent-session",
        "inline-extension-session",
    ]

    for example_name in example_names:
        path = examples_dir / example_name
        assert_uv_script(path)
        module = load_module(f"sdk_example_{example_name.replace('-', '_')}", path)
        assert callable(module.main)


@pytest.mark.asyncio
async def test_review_example_registers_recipe_command(tmp_path: Path) -> None:
    entrypoint = ROOT / "examples" / "review" / "kodelet-extension-review"
    assert_executable(entrypoint)
    module = load_module(
        "review_example",
        entrypoint,
    )
    harness = await create_test_harness(module.ext)
    init = harness.initialize({"extension": {"id": "review", "cwd": str(tmp_path)}})

    assert init["name"] == "review"
    assert init["commands"][0]["name"] == "review"
    assert init["commands"][0]["kind"] == "recipe"

    result = await harness.execute_command(
        {
            "name": "/review",
            "input": {"target": "HEAD", "focus": "tests"},
            "context": {"cwd": str(tmp_path)},
            "invocation": {"raw": "/review", "commandName": "review", "args": [], "flags": {}},
        }
    )

    assert result["action"] == "runAgent"
    assert result["recipeName"] == "review"
    assert "Review the code changes relative to `HEAD`" in result["prompt"]
    assert "tests" in result["prompt"]


def test_review_executable_initializes_while_stdin_stays_open(tmp_path: Path) -> None:
    entrypoint = ROOT / "examples" / "review" / "kodelet-extension-review"
    process = subprocess.Popen(
        [str(entrypoint)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert process.stdin is not None
        assert process.stdout is not None
        write_subprocess_frame(
            process.stdin,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "extension.initialize",
                "params": {
                    "protocolVersion": "2026-05-30",
                    "kodelet": {"version": "test"},
                    "extension": {"id": "review", "cwd": str(tmp_path), "dataDir": ""},
                    "capabilities": {},
                },
            },
        )
        response = read_subprocess_frame(process.stdout)
        assert response["id"] == 1
        assert response["result"]["name"] == "review"
    finally:
        process.kill()
        process.wait(timeout=5)


@pytest.mark.asyncio
async def test_workspace_example_registers_choice_tool(tmp_path: Path) -> None:
    entrypoint = ROOT / "examples" / "workspace" / "kodelet-extension-workspace"
    assert_executable(entrypoint)
    module = load_module(
        "workspace_example",
        entrypoint,
    )

    class FakeRPC:
        def __init__(self) -> None:
            self.requests: list[tuple[str, Any]] = []

        async def request(self, method: str, params: Any | None = None) -> Any:
            self.requests.append((method, params))
            if method == "kodelet.ui.select":
                return {"status": "submitted", "value": "Deny and remember this exact command"}
            return {"status": "submitted"}

    fake_rpc = FakeRPC()
    harness = await create_test_harness(module.ext, fake_rpc)
    init = harness.initialize(
        {"extension": {"id": "workspace", "cwd": str(tmp_path), "dataDir": str(tmp_path / "data")}}
    )

    assert init["name"] == "workspace"
    assert init["tools"][0]["name"] == "ask_user_choice"
    assert {subscription["event"] for subscription in init["subscriptions"]} == {"agent.start"}

    result = await harness.execute_tool(
        {
            "name": "ask_user_choice",
            "input": {"question": "Pick", "options": ["A", "B"]},
            "context": {"cwd": str(tmp_path)},
        }
    )
    assert result == {"content": "User responded with: Deny and remember this exact command"}


@pytest.mark.asyncio
async def test_workspace_policy_fixture_tool_and_bash_policy(tmp_path: Path) -> None:
    module = load_module(
        "workspace_policy_fixture",
        ROOT / "tests" / "fixtures" / "workspace_policy_extension.py",
    )

    class FakeRPC:
        def __init__(self) -> None:
            self.requests: list[tuple[str, Any]] = []

        async def request(self, method: str, params: Any | None = None) -> Any:
            self.requests.append((method, params))
            if method == "kodelet.ui.select":
                return {"status": "submitted", "value": "Deny and remember this exact command"}
            return {"status": "submitted"}

    fake_rpc = FakeRPC()
    harness = await create_test_harness(module.ext, fake_rpc)
    init = harness.initialize(
        {"extension": {"id": "workspace", "cwd": str(tmp_path), "dataDir": str(tmp_path / "data")}}
    )

    assert init["name"] == "workspace"
    assert init["tools"][0]["name"] == "ask_user_choice"
    assert {subscription["event"] for subscription in init["subscriptions"]} == {
        "agent.start",
        "tool.call",
    }

    result = await harness.execute_tool(
        {
            "name": "ask_user_choice",
            "input": {"question": "Pick", "options": ["A", "B"]},
            "context": {"cwd": str(tmp_path)},
        }
    )
    assert result == {"content": "User responded with: Deny and remember this exact command"}

    event_result = await harness.handle_event(
        {
            "id": "evt",
            "event": "tool.call",
            "context": {"cwd": str(tmp_path)},
            "payload": {
                "tool": {
                    "name": "bash",
                    "input": {"command": "rm -rf /tmp/example", "description": "cleanup"},
                }
            },
        }
    )

    assert event_result == {
        "block": {
            "reason": "Bash command denied and remembered by workspace policy: rm -rf /tmp/example"
        }
    }
    policy = json.loads((tmp_path / "data" / "bash-policy.json").read_text(encoding="utf-8"))
    assert policy == {"allowed": [], "denied": ["rm -rf /tmp/example"]}

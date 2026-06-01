from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from kodelet_sdk import create_test_harness

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_review_example_registers_recipe_command(tmp_path: Path) -> None:
    module = load_module("review_example", ROOT / "examples" / "review" / "extension.py")
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


@pytest.mark.asyncio
async def test_workspace_example_tool_and_bash_policy(tmp_path: Path) -> None:
    module = load_module("workspace_example", ROOT / "examples" / "workspace" / "extension.py")

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

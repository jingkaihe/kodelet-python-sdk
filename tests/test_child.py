from __future__ import annotations

import asyncio
from typing import Any

import pytest

from kodelet_sdk import (
    BackgroundTaskLease,
    ChildClient,
    ExecutionOptions,
    ExecutionProfile,
    Extension,
)


def test_presets_are_strict_isolated_and_pinned() -> None:
    one, two = Extension(), Extension()
    profile = ExecutionProfile(name="code_search", system_prompt_path="search.md",
                               options=ExecutionOptions(allowed_tools=["file_read", "grep_tool",
                                                                        "glob_tool"],
                                                        no_extensions=True, no_skills=True))
    one.register_profile(profile)
    two.register_profile({"name": "code_search"})
    assert profile.options and profile.options.allowed_tools
    profile.options.allowed_tools.append("bash")
    result = one.initialize({"extension": {"id": "one"}})
    assert result["profiles"][0]["options"]["allowedTools"] == [
        "file_read", "grep_tool", "glob_tool",
    ]
    result["profiles"][0]["name"] = "mutated"
    assert one.initialize({})["profiles"][0]["name"] == "code_search"
    with pytest.raises(ValueError, match="Duplicate"):
        one.register_profile(profile)
    for options in ({"api_key": "secret"}, {"allowedTools": None}, {"maxTokens": 0}):
        with pytest.raises(ValueError):
            ExecutionOptions.model_validate(options)


class Host:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any, bool]] = []
        self.result = {"conversationId": "child", "runId": "child-run", "done": False}

    async def request(self, method: str, params: Any = None) -> Any:
        self.calls.append((method, params, False))
        return self.result

    async def request_persistent(self, method: str, params: Any = None) -> Any:
        self.calls.append((method, params, True))
        return {**self.result, "done": True, "output": "found",
                "events": [{"sequence": 1, "kind": "tool-call", "toolName": "grep_tool"}]}


@pytest.mark.asyncio
async def test_children_use_scoped_rpc_and_explicit_retained_lease() -> None:
    host = Host()
    client = ChildClient(host)
    child = await client.start(profile="search", message="query", request_id="once",
                               lease=BackgroundTaskLease(host, "lease"))
    events: list[dict[str, Any]] = []
    assert (await child.wait(on_event=events.append))["output"] == "found"
    assert child.run_id == "child-run"
    assert [event["kind"] for event in events] == ["tool-call"]
    assert not host.calls[0][2]
    assert host.calls[1][2]
    await child.cancel()
    assert host.calls[-1] == ("kodelet.child.cancel", {"childId": "child", "leaseId": "lease"},
                              True)
    with pytest.raises(RuntimeError, match="no local fallback"):
        await ChildClient(None).start(profile="search", message="query")
    with pytest.raises(RuntimeError, match="real runner background lease"):
        await client.start(profile="search", message="query", lease=BackgroundTaskLease(None, None))


@pytest.mark.asyncio
async def test_wait_cancellation_targets_only_child() -> None:
    host = Host()
    child = await ChildClient(host).start(profile="search", message="query")
    task = asyncio.create_task(child.wait())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert host.calls[-1] == ("kodelet.child.cancel", {"childId": "child"}, False)

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
    profile = ExecutionProfile(
        name="code_search",
        system_prompt_path="search.md",
        options=ExecutionOptions(
            allowed_tools=["file_read", "grep_tool", "glob_tool"],
            no_extensions=True,
            no_skills=True,
        ),
    )
    one.register_profile(profile)
    two.register_profile({"name": "code_search"})
    assert profile.options and profile.options.allowed_tools
    profile.options.allowed_tools.append("bash")
    result = one.initialize({"extension": {"id": "one"}})
    assert result["profiles"][0]["options"]["allowedTools"] == [
        "file_read",
        "grep_tool",
        "glob_tool",
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
        self.result: dict[str, Any] = {
            "conversationId": "child",
            "runId": "child-run",
            "done": False,
        }
        self.steer_result: Any = {"outcome": "injected"}

    async def request(self, method: str, params: Any = None) -> Any:
        self.calls.append((method, params, False))
        if method == "kodelet.child.steer":
            return self.steer_result
        return self.result

    async def request_persistent(self, method: str, params: Any = None) -> Any:
        self.calls.append((method, params, True))
        if method == "kodelet.child.steer":
            return self.steer_result
        return {
            **self.result,
            "done": True,
            "output": "found",
            "events": [{"sequence": 1, "kind": "tool-call", "toolName": "grep_tool"}],
        }


@pytest.mark.asyncio
async def test_children_use_scoped_rpc_and_explicit_retained_lease() -> None:
    host = Host()
    client = ChildClient(host)
    child = await client.start(
        profile="search",
        message="query",
        request_id="once",
        lease=BackgroundTaskLease(host, "lease"),
    )
    events: list[dict[str, Any]] = []
    assert (await child.wait(on_event=events.append))["output"] == "found"
    assert child.run_id == "child-run"
    assert [event["kind"] for event in events] == ["tool-call"]
    assert not host.calls[0][2]
    assert host.calls[1][2]
    await child.cancel()
    assert host.calls[-1] == (
        "kodelet.child.cancel",
        {"childId": "child", "childRunId": "child-run", "leaseId": "lease"},
        True,
    )
    with pytest.raises(RuntimeError, match="no local fallback"):
        await ChildClient(None).start(profile="search", message="query")
    with pytest.raises(RuntimeError, match="real runner background lease"):
        await client.start(profile="search", message="query", lease=BackgroundTaskLease(None, None))


@pytest.mark.asyncio
async def test_repeated_retained_fork_still_requires_active_tool_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = Host()
    client = ChildClient(host)
    lease = BackgroundTaskLease(host, "lease")
    await client.start(profile="search", message="first", lease=lease)
    await client.start(profile="search", message="fork again", context_mode="fork", lease=lease)
    assert not host.calls[-1][2]

    async def ended(method: str, params: Any = None) -> Any:
        del method, params
        raise RuntimeError("originating tool ended")

    monkeypatch.setattr(host, "request", ended)
    with pytest.raises(RuntimeError, match="originating tool ended"):
        await client.start(profile="search", message="late fork", context_mode="fork", lease=lease)
    # Failure cannot silently turn the fork into a retained fresh submission.
    assert len(host.calls) == 2


@pytest.mark.asyncio
async def test_wait_cancellation_targets_only_child() -> None:
    host = Host()
    child = await ChildClient(host).start(profile="search", message="query")
    task = asyncio.create_task(child.wait())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert host.calls[-1] == (
        "kodelet.child.cancel",
        {"childId": "child", "childRunId": "child-run"},
        False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [{}, {"context_mode": "fresh"}, {"context_mode": "fork"}])
async def test_child_context_selection_preserves_existing_options(mode: dict[str, Any]) -> None:
    host = Host()
    child = await ChildClient(host).start(
        profile="search",
        message="query",
        request_id="first",
        **mode,
        options=ExecutionOptions(no_tools=False, allowed_commands=[], max_turns=0),
        cwd="/runner/project",
        system_prompt="runner-owned prompt",
    )
    expected = {
        "profile": "search",
        "message": "query",
        "requestId": "first",
        "options": {"noTools": False, "allowedCommands": [], "maxTurns": 0},
        "cwd": "/runner/project",
        "systemPrompt": "runner-owned prompt",
    }
    if mode:
        expected["contextMode"] = mode["context_mode"]
    assert host.calls == [("kodelet.child.start", expected, False)]
    assert child.conversation_id == "child"
    await child.read()
    assert host.calls[-1] == (
        "kodelet.child.read",
        {"childId": "child", "childRunId": "child-run", "after": 0},
        False,
    )


@pytest.mark.asyncio
async def test_retained_followup_reuses_conversation_not_run_or_request() -> None:
    host = Host()
    client = ChildClient(host)
    lease = BackgroundTaskLease(host, "lease")
    first = await client.start(profile="search", message="first", lease=lease, context_mode="fork")
    host.result = {**host.result, "runId": "second-run"}
    second = await client.start(
        profile="search", message="next", lease=lease, resume=first.conversation_id
    )
    assert second.conversation_id == first.conversation_id
    assert second.run_id != first.run_id
    assert host.calls[0][1]["requestId"] != host.calls[1][1]["requestId"]
    assert not host.calls[0][2] and host.calls[1][2]
    assert host.calls[1][1]["resume"] == "child"
    with pytest.raises(RuntimeError, match="does not match"):
        await first.read()
    await first.cancel()
    assert host.calls[-1][1] == {
        "childId": "child",
        "childRunId": "child-run",
        "leaseId": "lease",
    }
    await second.cancel()
    assert host.calls[-1][1]["childRunId"] == "second-run"
    # A new extension/tool context must establish authority again, not inherit
    # the first client's in-memory retained registration.
    await ChildClient(host).start(
        profile="search", message="later", lease=lease, resume=second.conversation_id
    )
    assert not host.calls[-1][2]


@pytest.mark.asyncio
async def test_malformed_or_wrong_resume_identity_never_creates_a_handle() -> None:
    host = Host()
    for values in ({"conversationId": None}, {"runId": 123}, {"runId": " "}, {"done": None}):
        host.result = {"conversationId": "child", "runId": "child-run", "done": False, **values}
        with pytest.raises(RuntimeError, match="Invalid central child execution"):
            await ChildClient(host).start(profile="search", message="query")
    host.result = {"conversationId": "other-child", "runId": "other-run", "done": False}
    with pytest.raises(RuntimeError, match="resumed conversation"):
        await ChildClient(host).start(profile="search", message="next", resume="child")
    assert len(host.calls) == 5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [
        {"profile": ""},
        {"profile": None},
        {"message": "  "},
        {"message": None},
        {"request_id": ""},
        {"request_id": None},
        {"request_id": "x" * 129},
        {"resume": ""},
        {"resume": "  "},
        {"resume": None},
        {"resume": 12},
        {"resume": "bad\x00id"},
        {"context_mode": None},
        {"context_mode": "other"},
        {"context_mode": False},
        {"context_mode": "fork", "resume": "child"},
        {"cwd": "  "},
        {"options": {"apiKey": "not-allowed"}},
    ],
)
async def test_invalid_child_start_is_rejected_before_rpc(invalid: dict[str, Any]) -> None:
    host = Host()
    request: dict[str, Any] = {"profile": "search", "message": "query", **invalid}
    with pytest.raises(ValueError):
        await ChildClient(host).start(**request)
    assert host.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("retained", [False, True])
async def test_steer_is_exact_idempotent_and_never_starts_an_idle_turn(retained: bool) -> None:
    host = Host()
    lease = BackgroundTaskLease(host, "lease") if retained else None
    child = await ChildClient(host).start(profile="search", message="query", lease=lease)
    for _ in range(2):
        assert await child.steer("Check errors", request_id="guidance-1") == {"outcome": "injected"}
    assert host.calls[-1] == host.calls[-2]
    assert host.calls[-1] == (
        "kodelet.child.steer",
        {
            "childId": "child",
            "childRunId": "child-run",
            "message": "Check errors",
            "requestId": "guidance-1",
            **({"leaseId": "lease"} if retained else {}),
        },
        retained,
    )
    host.steer_result = {"outcome": "promptRequired", "reason": "noRunningTurn"}
    assert (await child.steer("A new task"))["outcome"] == "promptRequired"
    assert host.calls[-1][1]["requestId"] != "guidance-1"
    assert [call[0] for call in host.calls].count("kodelet.child.start") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [
        {"message": ""},
        {"message": " \n "},
        {"message": None},
        {"message": "x" * (512 * 1024 + 1)},
        {"request_id": None},
        {"request_id": ""},
        {"request_id": "bad\x00id"},
        {"request_id": "x" * 129},
    ],
)
async def test_invalid_steer_is_rejected_before_rpc(invalid: dict[str, Any]) -> None:
    host = Host()
    child = await ChildClient(host).start(profile="search", message="query")
    with pytest.raises(ValueError):
        await child.steer(**{"message": "guidance", **invalid})
    assert len(host.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response", [None, {}, {"outcome": "started"}, {"outcome": "injected", "reason": None}]
)
async def test_malformed_steer_response_fails_without_retry(response: Any) -> None:
    host = Host()
    host.steer_result = response
    child = await ChildClient(host).start(profile="search", message="query")
    with pytest.raises(RuntimeError, match="steering response"):
        await child.steer("guidance")
    assert len(host.calls) == 2


@pytest.mark.asyncio
async def test_stale_authority_does_not_retry_or_switch_child_run() -> None:
    host = Host()
    child = await ChildClient(host).start(profile="search", message="query")

    async def stale(method: str, params: Any) -> Any:
        assert params["childRunId"] == "child-run"
        host.calls.append((method, params, False))
        raise RuntimeError("stale child run")

    child._call = stale
    for operation in (child.read, child.cancel, lambda: child.steer("guidance")):
        with pytest.raises(RuntimeError, match="stale child run"):
            await operation()
    assert len(host.calls) == 4


@pytest.mark.asyncio
async def test_wait_cancellation_while_read_is_pending_cancels_exact_run() -> None:
    host = Host()
    reading = asyncio.Event()
    read_finished = asyncio.Event()

    async def blocked(method: str, params: Any) -> Any:
        if method == "kodelet.child.read":
            reading.set()
            try:
                await asyncio.Future()
            finally:
                read_finished.set()
        return await host.request(method, params)

    child = await ChildClient(host).start(profile="search", message="query")
    child._call = blocked
    task = asyncio.create_task(child.wait())
    await asyncio.wait_for(reading.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)
    assert read_finished.is_set()
    assert host.calls[-1] == (
        "kodelet.child.cancel",
        {"childId": "child", "childRunId": "child-run"},
        False,
    )


@pytest.mark.asyncio
async def test_cancelled_admission_does_not_retry_or_claim_child_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = Host()
    started = asyncio.Event()

    async def uncertain(method: str, params: Any = None) -> Any:
        host.calls.append((method, params, False))
        started.set()
        await asyncio.Future()

    monkeypatch.setattr(host, "request", uncertain)
    task = asyncio.create_task(
        ChildClient(host).start(
            profile="search",
            message="query",
            request_id="reserved",
            lease=BackgroundTaskLease(host, "lease"),
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert host.calls == [
        (
            "kodelet.child.start",
            {
                "profile": "search",
                "message": "query",
                "requestId": "reserved",
                "leaseId": "lease",
            },
            False,
        )
    ]

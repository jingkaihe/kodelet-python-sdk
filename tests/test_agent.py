from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, cast

import pytest

import kodelet_sdk
from kodelet_sdk import (
    AgentUIHandlers,
    BaseModel,
    BridgeTransport,
    Client,
    CreateSessionOptions,
    ExecutionOptions,
    Extension,
    HostRPCError,
    Profile,
    SessionSteerResult,
    ToolContext,
    ToolUpdateData,
)
from kodelet_sdk.agent import SpawnedProcess, SpawnOptions
from kodelet_sdk.agent.rpc import ACPRPCClient

_DEFAULT_RESPONSE = object()


class FakeACPProcess(SpawnedProcess):
    def __init__(
        self,
        *,
        session_id: str = "conv-1",
        on_prompt: Callable[[Mapping[str, Any], FakeACPProcess], Awaitable[None] | None]
        | None = None,
        steer_result: Any = _DEFAULT_RESPONSE,
        steering_supported: bool = True,
        extension_version: Any = None,
        hierarchy_version: Any = None,
    ) -> None:
        self.stdout = _QueueLineReader()
        self.stderr = _QueueLineReader()
        self.stdin = _FakeStdin(self)
        self.requests: list[dict[str, Any]] = []
        self._session_id = session_id
        self._on_prompt = on_prompt
        self._steer_result = (
            {"outcome": "injected"}
            if steer_result is _DEFAULT_RESPONSE
            else steer_result
        )
        self._steering_supported = steering_supported
        self._extension_version = extension_version
        self._hierarchy_version = hierarchy_version
        self._server_id = 0
        self._server_pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self.server_responses: list[dict[str, Any]] = []
        self._extension_pending: dict[
            tuple[str, str, int | str], asyncio.Future[dict[str, Any]]
        ] = {}
        self.extension_frames: list[dict[str, Any]] = []
        self.host_frames: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._closed = asyncio.Event()
        self._returncode = 0
        self._tasks: set[asyncio.Task[Any]] = set()

    def terminate(self) -> None:
        self._close(0)

    def kill(self) -> None:
        self._close(0)

    async def wait(self) -> int:
        await self._closed.wait()
        return self._returncode

    def notify(self, method: str, params: Any | None = None) -> None:
        self.write({"jsonrpc": "2.0", "method": method, "params": params})

    def write(self, message: Mapping[str, Any]) -> None:
        stdout = self.stdout
        assert isinstance(stdout, _QueueLineReader)
        stdout.feed(f"{json.dumps(message)}\n".encode())

    async def extension_frame(
        self,
        message: Mapping[str, Any] | None = None,
        *,
        session_id: str | None = None,
        run_id: str = "run-1",
        extension_id: str = "inline-1",
        close: bool = False,
    ) -> dict[str, Any]:
        """Send a reverse ACP request and wait only for its acceptance ACK."""

        self._server_id += 1
        request_id = self._server_id
        pending: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._server_pending[request_id] = pending
        self.write({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "kodelet/extensionFrame",
            "params": {
                "sessionId": session_id or self._session_id,
                "runId": run_id,
                "extensionId": extension_id,
                **({"close": True} if close else {"message": message}),
            },
        })
        try:
            return await asyncio.wait_for(pending, timeout=2)
        finally:
            self._server_pending.pop(request_id, None)

    async def extension_call(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        request_id: int | str = 1,
        run_id: str = "run-1",
        extension_id: str = "inline-1",
    ) -> dict[str, Any]:
        """Wait for the raw extension response separately from the ACP ACK."""

        key = (run_id, extension_id, request_id)
        pending: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        assert key not in self._extension_pending
        self._extension_pending[key] = pending
        try:
            ack = await self.extension_frame(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}},
                run_id=run_id,
                extension_id=extension_id,
            )
            assert ack.get("result") == {}, ack
            return await asyncio.wait_for(pending, timeout=2)
        finally:
            self._extension_pending.pop(key, None)

    async def respond_to_host(
        self, frame: Mapping[str, Any], result: Any = None, *, error: Any = None
    ) -> None:
        ack = await self.extension_frame(
            {
                "jsonrpc": "2.0",
                "id": frame["message"]["id"],
                **({"error": error} if error is not None else {"result": result}),
            },
            run_id=frame["runId"],
            extension_id=frame["extensionId"],
        )
        assert ack.get("result") == {}, ack

    def handle_input(self, chunk: bytes) -> None:
        for line in chunk.decode("utf-8").splitlines():
            if not line.strip():
                continue
            request = json.loads(line)
            if not isinstance(request, dict):
                continue
            if not request.get("method"):
                self.server_responses.append(request)
                pending = self._server_pending.get(request.get("id"))
                if pending is not None and not pending.done():
                    pending.set_result(request)
                continue
            if request.get("id") is None:
                continue
            self.requests.append(request)
            task = asyncio.create_task(self._handle_request(request))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _handle_request(self, request: Mapping[str, Any]) -> None:
        method = request.get("method")
        request_id = request.get("id")
        if method == "initialize":
            self._respond(
                request_id,
                {
                    "protocolVersion": 1,
                    "agentCapabilities": {},
                    "authMethods": [],
                    "_meta": {
                        **({"steering": {"supported": True}} if self._steering_supported else {}),
                        **({"sessionExtensions": {"version": self._extension_version}}
                           if self._extension_version is not None else {}),
                        **({"conversationHierarchy": {"version": self._hierarchy_version}}
                           if self._hierarchy_version is not None else {}),
                    },
                },
            )
            return
        if method == "session/new":
            self._respond(request_id, {"sessionId": self._session_id})
            return
        if method == "session/load":
            self._session_id = request["params"]["sessionId"]
            self._respond(request_id, {})
            return
        if method == "kodelet/extensionFrame":
            self._respond(request_id, {})
            frame = dict(request["params"])
            self.extension_frames.append(frame)
            message = frame.get("message") or {}
            if message.get("method") or frame.get("close"):
                self.host_frames.put_nowait(frame)
            else:
                key = (frame["runId"], frame["extensionId"], message["id"])
                pending = self._extension_pending.get(key)
                if pending is not None and not pending.done():
                    pending.set_result(message)
            return
        if method == "session/prompt":
            try:
                if self._on_prompt is not None:
                    result = self._on_prompt(request, self)
                    if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
                        await result
                self._respond(request_id, {"stopReason": "end_turn"})
            except Exception as exc:
                self._respond_error(request_id, str(exc))
            return
        if method == "_session/steering":
            self._respond(request_id, self._steer_result)
            return
        self._respond_error(request_id, f"Unexpected method: {method}")

    def _respond(self, request_id: Any, result: Any) -> None:
        self.write({"jsonrpc": "2.0", "id": request_id, "result": result})

    def _respond_error(self, request_id: Any, message: str) -> None:
        self.write(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": message},
            }
        )

    def _close(self, returncode: int) -> None:
        if self._closed.is_set():
            return
        self._returncode = returncode
        stdout = self.stdout
        stderr = self.stderr
        assert isinstance(stdout, _QueueLineReader)
        assert isinstance(stderr, _QueueLineReader)
        stdout.feed_eof()
        stderr.feed_eof()
        self._closed.set()


class _FakeStdin:
    def __init__(self, process: FakeACPProcess) -> None:
        self._process = process

    def write(self, data: bytes) -> object:
        self._process.handle_input(data)
        return None

    async def drain(self) -> None:
        await asyncio.sleep(0)

    def close(self) -> object:
        return None


class _QueueLineReader:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()

    async def readline(self) -> bytes:
        return await self._queue.get()

    def feed(self, data: bytes) -> None:
        self._queue.put_nowait(data)

    def feed_eof(self) -> None:
        self._queue.put_nowait(b"")


def test_agent_package_preserves_public_reexports() -> None:
    import kodelet_sdk.agent as agent
    from kodelet_sdk.agent import client as client_module

    assert agent.Client is Client
    assert agent.Profile is Profile
    assert agent.SessionSteerResult is SessionSteerResult
    assert agent.CreateSessionOptions is CreateSessionOptions
    assert agent.AgentUIHandlers is AgentUIHandlers
    assert agent.BridgeTransport is BridgeTransport
    assert kodelet_sdk.Client is Client
    assert kodelet_sdk.SessionSteerResult is SessionSteerResult
    assert client_module.Client is Client


@pytest.mark.asyncio
async def test_session_preserves_typed_execution_options_with_named_profile() -> None:
    calls: list[list[str]] = []

    def spawn(_command: str, args: Sequence[str], _options: SpawnOptions) -> FakeACPProcess:
        calls.append(list(args))
        return FakeACPProcess()

    client = Client(spawn=spawn)
    try:
        await client.create_session(
            profile="work",
            options=ExecutionOptions(
                max_turns=0,
                no_tools=False,
                allowed_tools=["file_read", "grep_tool", "glob_tool"],
                allowed_commands=[],
                enableFSSearchTools=True,
            ),
        )
        assert calls == [[
            "acp",
            "--max-turns=0",
            "--no-tools=false",
            '--allowed-tools="file_read","grep_tool","glob_tool"',
            "--allowed-commands=",
            "--enable-fs-search-tools=true",
            "--profile=work",
        ]]
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("registration", [
    {
        "name": "code-search",
        "provider": "openai",
        "model": "gpt-5.6-luna",
        "reasoning_effort": "none",
        "openai": {
            "platform": "codex",
            "api_mode": "responses",
            "service_tier": "fast",
        },
        "hidden": True,
    },
    {
        "name": "claude",
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "anthropic": {"platform": "anthropic"},
        "anthropic_api_access": "subscription",
    },
])
async def test_registered_profile_name_reaches_acp_without_expanding_model_options(
    registration: dict[str, Any],
) -> None:
    ext = Extension()
    profile = ext.register_profile(**registration)
    manifest = ext.initialize({"capabilities": {"profiles": {"remote": True}}})
    assert profile == registration["name"]
    assert profile == manifest["profiles"][0]["name"]
    calls: list[list[str]] = []
    process = FakeACPProcess()

    def spawn(_command: str, args: Sequence[str], _options: SpawnOptions) -> FakeACPProcess:
        calls.append(list(args))
        return process

    client = Client(server="http://daemon", runner="runner-one", spawn=spawn)
    try:
        await client.create_session(profile=profile, cwd="/only/on/runner")
        assert calls == [[
            "acp", "--server", "http://daemon", "--runner", "runner-one",
            f"--profile={registration['name']}",
        ]]
        assert process.requests[1]["method"] == "session/new"
        assert process.requests[1]["params"] == {"cwd": "/only/on/runner"}
    finally:
        await client.close()


def test_profile_maps_early_profiler_spelling_and_nested_config() -> None:
    profile = Profile(
        {
            "name": "openai",
            "profiler": "openai",
            "model": "gpt-5.5",
            "max_tokens": 128000,
            "reasoning_effort": "xhigh",
            "weak_model": "gpt-5.4-mini",
            "enable_fs_search_tools": True,
            "openai": {
                "api_mode": "responses",
                "platform": "codex",
                "service_tier": "fast",
            },
        }
    )

    assert profile.to_launch_config() == {
        "args": [],
        "config": {
            "name": "openai",
            "provider": "openai",
            "model": "gpt-5.5",
            "max_tokens": 128000,
            "reasoning_effort": "xhigh",
            "weak_model": "gpt-5.4-mini",
            "enable_fs_search_tools": True,
            "openai": {
                "api_mode": "responses",
                "platform": "codex",
                "service_tier": "fast",
            },
        },
    }
    assert profile.toLaunchConfig() == profile.to_launch_config()


def test_named_profile_preserves_launch_config() -> None:
    profile = Profile.named("work")

    assert profile.to_launch_config() == {"args": ["--profile", "work"]}
    assert profile.toLaunchConfig() == profile.to_launch_config()


@pytest.mark.asyncio
async def test_session_sends_typed_daemon_flags_without_temporary_config() -> None:
    calls: list[dict[str, Any]] = []

    def spawn(_command: str, args: Sequence[str], options: SpawnOptions) -> FakeACPProcess:
        calls.append({"args": list(args), "env": options.get("env")})
        return FakeACPProcess(session_id="conv-profile")

    client = Client(spawn=spawn)
    session = await client.create_session(
        profile={
            "name": "openai",
            "provider": "openai",
            "model": "gpt-5.5",
            "allowed_tools": ["sdk_echo"],
        }
    )

    env = calls[0]["env"]
    assert "KODELET_CONFIG_FILE_MODE" not in env
    assert "KODELET_CONFIG_FILE" not in env
    assert calls[0]["args"] == [
        "acp", "--provider=openai", "--model=gpt-5.5", '--allowed-tools="sdk_echo"',
    ]

    await session.close()


@pytest.mark.asyncio
async def test_inline_options_do_not_rewrite_client_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def spawn(_command: str, _args: Sequence[str], options: SpawnOptions) -> FakeACPProcess:
        calls.append({"env": options.get("env")})
        return FakeACPProcess(session_id="conv-env")

    monkeypatch.setenv("KODELET_MODEL", "ambient-model")
    client = Client(spawn=spawn, env={"KODELET_PROVIDER": "explicit-provider"})
    await client.create_session(profile={"provider": "openai", "model": "inline-model"})

    env = calls[0]["env"]
    assert env["KODELET_MODEL"] == "ambient-model"
    assert env["KODELET_PROVIDER"] == "explicit-provider"
    assert "KODELET_CONFIG_FILE_MODE" not in env
    await client.close()


@pytest.mark.asyncio
async def test_session_rejects_implicit_inherited_context_before_spawn() -> None:
    processes: list[FakeACPProcess] = []

    class InheritedContext:
        async def fork_conversation(self) -> str:
            raise AssertionError("must not fork implicitly")

    def spawn(_command: str, _args: Sequence[str], _options: SpawnOptions) -> FakeACPProcess:
        process = FakeACPProcess()
        processes.append(process)
        return process

    client = Client(spawn=spawn)
    with pytest.raises(ValueError, match=r"ctx\.fork_conversation\(\).*resume"):
        await client.create_session(inherit_context=cast(Any, InheritedContext()))
    assert processes == []
    await client.close()


@pytest.mark.asyncio
async def test_session_cancellation_while_loading_closes_spawned_process() -> None:
    processes: list[FakeACPProcess] = []
    load_started = asyncio.Event()

    class DeferredProcess(FakeACPProcess):
        async def _handle_request(self, request: Mapping[str, Any]) -> None:
            if request.get("method") == "session/load":
                load_started.set()
                return
            await super()._handle_request(request)

    def spawn(_command: str, _args: Sequence[str], _options: SpawnOptions) -> FakeACPProcess:
        process = DeferredProcess()
        processes.append(process)
        return process

    client = Client(spawn=spawn)
    create_task = asyncio.create_task(
        client.create_session(resume="existing")
    )
    await asyncio.wait_for(load_started.wait(), timeout=1)

    create_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await create_task

    require_process = processes[0]
    assert require_process._closed.is_set()


@pytest.mark.asyncio
async def test_acp_close_reports_incomplete_cleanup_and_allows_session_retry() -> None:
    class StubbornProcess(FakeACPProcess):
        def __init__(self) -> None:
            super().__init__()
            self.signals: list[str] = []

        def terminate(self) -> None:
            self.signals.append("TERM")

        def kill(self) -> None:
            self.signals.append("KILL")

    process = StubbornProcess()
    client = Client(spawn=lambda _command, _args, _options: process)
    session = await client.create_session()
    with pytest.raises(RuntimeError, match="cleanup is incomplete"):
        await asyncio.wait_for(session.close(), timeout=3)
    assert process.signals == ["TERM", "KILL"]
    assert session in client._sessions
    assert session._rpc in client._rpcs
    process._close(0)
    await asyncio.wait_for(session.close(), timeout=1)
    assert session not in client._sessions
    assert not client._rpcs
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("as_mapping", [False, True], ids=["kwargs", "mapping"])
@pytest.mark.parametrize(
    "options",
    [
        {"extension_transport": "unix"},
        {"extension_transport": "tcp"},
        {"extension_transport": None},
        {"extensions": [], "extension_transport": "tcp"},
        {"ui": {}},
        {"ui": {"select": lambda _request: pytest.fail("unexpected UI call")}},
    ],
)
async def test_inline_compatibility_options_without_extensions_do_not_require_relay(
    options: CreateSessionOptions, as_mapping: bool
) -> None:
    process = FakeACPProcess()
    client = Client(spawn=lambda _command, _args, _options: process)
    try:
        session = (
            await client.create_session(options)
            if as_mapping else await client.create_session(**options)
        )
        assert session.id == "conv-1"
        assert process.requests[1]["params"] == {"cwd": os.getcwd()}
        assert process.extension_frames == []
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [None, 0, 2, "1", True])
@pytest.mark.parametrize("resume", ["", "saved-session"])
async def test_inline_extensions_require_negotiated_version_before_new_or_load(
    version: Any, resume: str
) -> None:
    process = FakeACPProcess(extension_version=version)
    client = Client(spawn=lambda _command, _args, _options: process)

    def entrypoint(_ext: Extension) -> None:
        pytest.fail("capability failure must not invoke entrypoints")

    try:
        with pytest.raises(RuntimeError, match=r"sessionExtensions.*version 1"):
            await client.create_session(extensions=[entrypoint], resume=resume)
        assert [request["method"] for request in process.requests] == ["initialize"]
        capabilities = process.requests[0]["params"]["clientCapabilities"]
        assert capabilities["_meta"]["sessionExtensions"] == {"version": 1}
        assert process._closed.is_set()
        assert not client._sessions
        assert not client._rpcs
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("as_mapping", [False, True])
async def test_fresh_child_metadata_preserves_inline_extensions(
    inline: bool, as_mapping: bool
) -> None:
    process = FakeACPProcess(extension_version=1, hierarchy_version=1)
    client = Client(spawn=lambda _command, _args, _options: process)
    options: CreateSessionOptions = {
        "cwd": "/runner/workspace",
        "parent_conversation_id": " parent-conversation ",
    }
    if inline:
        options["extensions"] = [Extension()]
    try:
        session = (
            await client.create_session(options)
            if as_mapping else await client.create_session(**options)
        )
        assert session.id == "conv-1"
        assert process.requests[1]["params"] == {
            "cwd": "/runner/workspace",
            "_meta": {
                "conversationHierarchy": {
                    "version": 1,
                    "parentConversationId": "parent-conversation",
                },
                **({"sessionExtensions": {"version": 1, "extensionIds": ["inline-1"]}}
                   if inline else {}),
            },
        }
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [None, 0, 2, "1", True])
async def test_child_requires_hierarchy_before_sending_new_session(version: Any) -> None:
    process = FakeACPProcess(hierarchy_version=version)
    client = Client(spawn=lambda _command, _args, _options: process)
    try:
        with pytest.raises(RuntimeError, match=r"conversationHierarchy version 1.*update Kodelet"):
            await client.create_session(parent_conversation_id="parent")
        assert [request["method"] for request in process.requests] == ["initialize"]
        assert process._closed.is_set()
        assert not client._sessions
        assert not client._rpcs
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [
    {"parent_conversation_id": None},
    {"parent_conversation_id": ""},
    {"parent_conversation_id": " "},
    {"parent_conversation_id": 123},
    {"parent_conversation_id": True},
    {"parent_conversation_id": "parent", "resume": "child"},
])
async def test_invalid_parent_options_fail_before_spawning(options: Any) -> None:
    def spawn(_command: str, _args: Sequence[str], _options: SpawnOptions) -> SpawnedProcess:
        pytest.fail("invalid parent options must not start ACP")

    client = Client(spawn=spawn)
    try:
        with pytest.raises(ValueError, match="parent_conversation_id"):
            await client.create_session(options)
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("resume", ["", "saved-session"])
@pytest.mark.parametrize("transport", ["unix", "tcp"])
async def test_inline_attachment_metadata_is_deterministic_and_callbacks_stay_live(
    resume: str, transport: BridgeTransport
) -> None:
    process = FakeACPProcess(extension_version=1)
    spawned: list[list[str]] = []

    def spawn(_command: str, args: Sequence[str], _options: SpawnOptions) -> FakeACPProcess:
        spawned.append(list(args))
        return process

    client = Client(spawn=spawn, server="https://daemon.example", runner="selected")
    original = Extension(name="live")
    state = {"prefix": "before"}
    entrypoints: list[Extension] = []

    @original.tool("read_state", description="Read the original Python closure")
    async def read_state() -> str:
        return state["prefix"]

    def entrypoint(ext: Extension) -> None:
        entrypoints.append(ext)
        ext.tool("other", description="Second extension")(lambda: "other result")

    try:
        session = await client.create_session({
            "extensions": [original, entrypoint],
            "resume": resume,
            "extension_transport": transport,
            "ui": {},
        })
        assert spawned == [["acp", "--server", "https://daemon.example", "--runner", "selected"]]
        attachment = process.requests[1]
        assert attachment["method"] == ("session/load" if resume else "session/new")
        assert attachment["params"]["_meta"] == {
            "sessionExtensions": {"version": 1, "extensionIds": ["inline-1", "inline-2"]}
        }
        assert entrypoints == []
        for extension_id in ("inline-1", "inline-2"):
            initialized = await process.extension_call(
                "extension.initialize", {"extension": {"id": extension_id}},
                extension_id=extension_id,
            )
            assert len(initialized["result"]["tools"]) == 1
        assert len(entrypoints) == 1
        state["prefix"] = "changed after attachment"
        result = await process.extension_call("extension.tool.execute", {"name": "read_state"})
        assert result["result"] == {"content": "changed after attachment"}
        other = await process.extension_call(
            "extension.tool.execute", {"name": "other"}, extension_id="inline-2"
        )
        assert other["result"] == {"content": "other result"}
        assert session.id == (resume or "conv-1")
        assert original._init_params is None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_inline_prompt_streams_updates_and_nested_host_rpc_with_colliding_ids() -> None:
    ext = Extension(name="streaming")
    results: list[dict[str, Any]] = []

    @ext.tool("stream", description="Publish snapshots and fork on the runner")
    async def stream(_input: Any, ctx: ToolContext) -> str:
        await ctx.update("first", {"count": 1})
        await ctx.update("first and second", {"count": 2})
        return await ctx.fork_conversation("snapshot")

    async def on_prompt(_request: Mapping[str, Any], process: FakeACPProcess) -> None:
        result = await process.extension_call(
            "extension.tool.execute", {"name": "stream"}, request_id=1
        )
        results.append(result)
        process.notify("session/update", {
            "sessionId": "conv-1",
            "update": {"sessionUpdate": "agent_message_chunk",
                       "content": {"type": "text", "text": result["result"]["content"]}},
        })

    process = FakeACPProcess(extension_version=1, on_prompt=on_prompt)
    client = Client(spawn=lambda _command, _args, _options: process)
    try:
        session = await client.create_session(extensions=[ext], streaming=True)
        await process.extension_call("extension.initialize", {
            "capabilities": {"tools": {"updates": True}, "conversations": {"fork": True}}
        })
        snapshots: list[str] = []
        session.on("tool.update", lambda event: snapshots.append(event.data.result))
        run = asyncio.create_task(session.run_and_wait("stream it"))
        for reverse_id, content in enumerate(("first", "first and second"), start=1):
            frame = await asyncio.wait_for(process.host_frames.get(), timeout=2)
            message = frame["message"]
            assert message == {
                "jsonrpc": "2.0", "id": reverse_id, "parentId": 1,
                "method": "kodelet.tool.update",
                "params": {"content": content, "data": {"count": reverse_id}},
            }
            assert frame["sessionId"] == session.id
            assert process.server_responses[-1]["result"] == {}
            assert not run.done()  # ACK was not delayed until the callback completed.
            process.notify("session/update", {
                "sessionId": session.id,
                "update": {
                    "sessionUpdate": "tool_call_update", "toolCallId": "call-1",
                    "status": "in_progress",
                    "content": [{"type": "content", "content": {"type": "text", "text": content}}],
                },
            })
            await process.respond_to_host(frame, {})
        fork = await asyncio.wait_for(process.host_frames.get(), timeout=2)
        assert fork["message"]["method"] == "kodelet.conversation.fork"
        assert fork["message"]["parentId"] == 1
        assert fork["message"]["params"] == {"name": "snapshot"}
        await process.respond_to_host(fork, {"conversationId": "runner-fork"})
        response = await asyncio.wait_for(run, timeout=2)
        assert response.content == "runner-fork"
        assert results == [{"jsonrpc": "2.0", "id": 1, "result": {"content": "runner-fork"}}]
        assert snapshots == ["first", "first and second"]
        assert [event.data.result for event in response.events if event.type == "tool.update"] == [
            "first and second"
        ]
    finally:
        await client.close()


class _RelayInput(BaseModel):
    value: int


@pytest.mark.asyncio
async def test_inline_callback_validation_dispatch_and_host_errors_stay_protocol_errors() -> None:
    ext = Extension()
    calls: list[int] = []

    @ext.tool("fail", description="Validate input then fail", input_schema=_RelayInput)
    async def fail(input: _RelayInput) -> str:
        calls.append(input.value)
        raise ValueError("callback exploded")

    @ext.tool("host_error", description="Preserve host error codes")
    async def host_error(_input: Any, ctx: ToolContext) -> dict[str, Any]:
        try:
            await ctx.fork_conversation()
        except HostRPCError as exc:
            return {"content": str(exc), "data": {"code": exc.code, "data": exc.data}}
        raise AssertionError("expected a host error")

    process = FakeACPProcess(extension_version=1)
    client = Client(spawn=lambda _command, _args, _options: process)
    try:
        await client.create_session(extensions=[ext])
        await process.extension_call("extension.initialize", {
            "capabilities": {"conversations": {"fork": True}}
        })
        cases = [
            ("extension.tool.execute", {"name": "fail", "input": {"value": "bad"}}, "validation"),
            ("extension.tool.execute", {"name": "fail", "input": {"value": 4}}, "exploded"),
            ("extension.tool.execute", {"name": "missing"}, "Unknown extension tool"),
            ("extension.unknown", {}, "Unknown JSON-RPC method"),
        ]
        for method, params, error in cases:
            response = await process.extension_call(method, params)
            assert response["error"]["code"] == -32000
            assert error in response["error"]["message"]
        assert calls == [4]
        calling = asyncio.create_task(process.extension_call(
            "extension.tool.execute", {"name": "host_error"}
        ))
        frame = await asyncio.wait_for(process.host_frames.get(), timeout=2)
        await process.respond_to_host(frame, error={
            "code": -32123, "message": "runner rejected", "data": {"runner": "selected"}
        })
        response = await calling
        assert response["result"] == {
            "content": "runner rejected", "data": {"code": -32123, "data": {"runner": "selected"}}
        }
        assert not process._closed.is_set()
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("dismissed", [False, True])
async def test_inline_ui_handlers_convert_local_results_and_dismissal(dismissed: bool) -> None:
    ext = Extension()
    requests: list[tuple[str, Any]] = []

    async def input_handler(request: Any) -> str | None:
        requests.append(("input", request))
        return None if dismissed else "typed"

    def confirm_handler(request: Any) -> bool:
        requests.append(("confirm", request))
        return not dismissed

    def select_handler(request: Any) -> str | None:
        requests.append(("select", request))
        return None if dismissed else "B"

    def notify_handler(request: Any) -> None:
        requests.append(("notify", request))

    @ext.tool("ask", description="Request user input locally")
    async def ask(_input: Any, ctx: ToolContext) -> dict[str, Any]:
        values = {
            "input": await ctx.ui.input({"title": "Text", "secret": True}),
            "confirm": await ctx.ui.confirm({"title": "Confirm"}),
            "select": await ctx.ui.select({"title": "Select", "options": ["A", "B"]}),
        }
        assert values == {
            "input": None if dismissed else "typed",
            "confirm": not dismissed,
            "select": None if dismissed else "B",
        }
        await ctx.ui.notify("Done")
        return {"content": "asked", "data": values}

    process = FakeACPProcess(extension_version=1)
    client = Client(spawn=lambda _command, _args, _options: process)
    try:
        await client.create_session(extensions=[ext], ui={
            "input": input_handler, "confirm": confirm_handler,
            "select": select_handler, "notify": notify_handler,
        })
        await process.extension_call("extension.initialize")
        response = await process.extension_call("extension.tool.execute", {"name": "ask"})
        assert response["result"] == {
            "content": "asked",
            "data": {"confirm": False} if dismissed else {
                "input": "typed", "confirm": True, "select": "B",
            },
        }
        assert requests == [
            ("input", {"title": "Text", "secret": True}),
            ("confirm", {"title": "Confirm"}),
            ("select", {"title": "Select", "options": ["A", "B"]}),
            ("notify", {"message": "Done"}),
        ]
        assert process.host_frames.empty()
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("provide_input", [False, True])
async def test_inline_ui_overrides_only_provided_capabilities_and_forwards_the_rest(
    provide_input: bool,
) -> None:
    initialized: list[Mapping[str, Any]] = []

    class ObservedExtension(Extension):
        def initialize(self, params: Mapping[str, Any]) -> dict[str, Any]:
            initialized.append(params)
            return super().initialize(params)

    ext = ObservedExtension()

    @ext.tool("ask", description="Use runner UI where no local handler exists")
    async def ask(_input: Any, ctx: ToolContext) -> str | None:
        return await ctx.ui.select({"title": "Runner selection", "options": ["runner"]})

    process = FakeACPProcess(extension_version=1)
    client = Client(spawn=lambda _command, _args, _options: process)
    ui: AgentUIHandlers = {"input": lambda _request: "local"} if provide_input else {}
    try:
        await client.create_session(extensions=[ext], ui=ui)
        capabilities = {
            "ui": {"input": False, "select": False, "surfaces": True},
            "tools": {"updates": False},
        }
        await process.extension_call("extension.initialize", {"capabilities": capabilities})
        assert initialized == [{"capabilities": {
            **capabilities,
            "ui": {**capabilities["ui"], "input": provide_input},
        }}]
        assert capabilities["ui"]["input"] is False
        calling = asyncio.create_task(process.extension_call(
            "extension.tool.execute", {"name": "ask"}, request_id="ui-parent"
        ))
        frame = await asyncio.wait_for(process.host_frames.get(), timeout=2)
        assert frame["message"]["method"] == "kodelet.ui.select"
        assert frame["message"]["parentId"] == "ui-parent"
        await process.respond_to_host(frame, {"status": "submitted", "value": "runner"})
        assert (await calling)["result"] == {"content": "runner"}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_inline_ui_handler_errors_are_returned_without_closing_channel() -> None:
    ext = Extension()
    count = 0

    async def input_handler(_request: Any) -> str:
        nonlocal count
        count += 1
        if count == 1:
            raise ValueError("local UI failed")
        return "recovered"

    @ext.tool("ask", description="Ask UI")
    async def ask(_input: Any, ctx: ToolContext) -> str | None:
        return await ctx.ui.input({"title": "Question"})

    process = FakeACPProcess(extension_version=1)
    client = Client(spawn=lambda _command, _args, _options: process)
    try:
        await client.create_session(extensions=[ext], ui={"input": input_handler})
        await process.extension_call("extension.initialize")
        first = await process.extension_call("extension.tool.execute", {"name": "ask"})
        assert first["error"] == {"code": -32000, "message": "local UI failed"}
        second = await process.extension_call("extension.tool.execute", {"name": "ask"})
        assert second["result"] == {"content": "recovered"}
        assert process.host_frames.empty()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_inline_ui_handler_cancellation_is_a_dismissed_response() -> None:
    ext = Extension()

    async def input_handler(_request: Any) -> str:
        raise asyncio.CancelledError

    @ext.tool("ask", description="Handle dismissed input without cancelling the tool")
    async def ask(_input: Any, ctx: ToolContext) -> str:
        assert await ctx.ui.input({"title": "Dismiss"}) is None
        return "dismissed, tool completed"

    process = FakeACPProcess(extension_version=1)
    client = Client(spawn=lambda _command, _args, _options: process)
    try:
        await client.create_session(extensions=[ext], ui={"input": input_handler})
        await process.extension_call("extension.initialize")
        response = await process.extension_call("extension.tool.execute", {"name": "ask"})
        assert response["result"] == {"content": "dismissed, tool completed"}
        assert process.host_frames.empty()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_inline_ui_caller_timeout_cancels_handler_before_tool_completion() -> None:
    ext = Extension()
    ui_started, ui_cancelled = asyncio.Event(), asyncio.Event()
    timed_out, finish_tool = asyncio.Event(), asyncio.Event()

    async def input_handler(_request: Any) -> str:
        ui_started.set()
        try:
            await asyncio.Event().wait()
            return "unreachable"
        finally:
            ui_cancelled.set()

    @ext.tool("ask", description="Time out UI but continue running the tool")
    async def ask(_input: Any, ctx: ToolContext) -> str:
        try:
            await asyncio.wait_for(ctx.ui.input({"title": "Timeout"}), timeout=0.02)
        except TimeoutError:
            timed_out.set()
        await finish_tool.wait()
        return "continued after timeout"

    process = FakeACPProcess(extension_version=1)
    client = Client(spawn=lambda _command, _args, _options: process)
    calling: asyncio.Task[dict[str, Any]] | None = None
    try:
        await client.create_session(extensions=[ext], ui={"input": input_handler})
        await process.extension_call("extension.initialize")
        calling = asyncio.create_task(process.extension_call(
            "extension.tool.execute", {"name": "ask"}
        ))
        await asyncio.wait_for(ui_started.wait(), timeout=2)
        await asyncio.wait_for(timed_out.wait(), timeout=2)
        await asyncio.wait_for(ui_cancelled.wait(), timeout=1)
        assert not calling.done()
        finish_tool.set()
        assert (await calling)["result"] == {"content": "continued after timeout"}
    finally:
        finish_tool.set()
        await client.close()
        if calling is not None:
            calling.cancel()
            await asyncio.gather(calling, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["channel", "session", "client"])
async def test_inline_close_cancels_tool_before_joining_dependent_ui_cleanup(scope: str) -> None:
    ext = Extension()
    ui_started, tool_finally, ui_finally = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def input_handler(_request: Any) -> str:
        ui_started.set()
        try:
            await asyncio.Event().wait()
            return "unreachable"
        finally:
            await tool_finally.wait()
            ui_finally.set()

    @ext.tool("ask", description="UI cleanup depends on originating tool cancellation")
    async def ask(_input: Any, ctx: ToolContext) -> str | None:
        try:
            return await ctx.ui.input({"title": "Wait for tool cleanup"})
        finally:
            tool_finally.set()

    process = FakeACPProcess(extension_version=1)
    client = Client(spawn=lambda _command, _args, _options: process)
    try:
        session = await client.create_session(extensions=[ext], ui={"input": input_handler})
        await process.extension_call("extension.initialize")
        ack = await process.extension_frame({
            "jsonrpc": "2.0", "id": 42, "method": "extension.tool.execute",
            "params": {"name": "ask"},
        })
        assert ack["result"] == {}
        await asyncio.wait_for(ui_started.wait(), timeout=2)
        assert not tool_finally.is_set()
        relay = session._rpc._extension_relay
        assert relay is not None
        channel = relay._channels[("run-1", "inline-1")]
        if scope == "channel":
            assert (await process.extension_frame(close=True))["result"] == {}
            await asyncio.wait_for(asyncio.gather(*relay._closing), timeout=2)
        elif scope == "session":
            await asyncio.wait_for(session.close(), timeout=2)
        else:
            await asyncio.wait_for(client.close(), timeout=2)
        assert tool_finally.is_set()
        assert ui_finally.is_set()
        assert channel._worker.done()
        assert not channel._ui_tasks
        assert not relay._channels
    finally:
        # Unblock cleanup even if the ordering regresses, so failure is bounded.
        tool_finally.set()
        await asyncio.wait_for(client.close(), timeout=2)


@pytest.mark.asyncio
async def test_inline_reverse_ack_blocked_drain_does_not_block_acp_reader() -> None:
    ext = Extension()

    @ext.tool("fork", description="Nested RPC must complete while reverse ACK drain blocks")
    async def fork(_input: Any, ctx: ToolContext) -> str:
        return await ctx.fork_conversation()

    draining, drain_cancelled, release_drain = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class BlockedACKStdin(_FakeStdin):
        def __init__(self, process: FakeACPProcess) -> None:
            super().__init__(process)
            self._block_next_drain = False

        def write(self, data: bytes) -> object:
            message = json.loads(data)
            if not message.get("method") and not draining.is_set():
                self._block_next_drain = True
            return super().write(data)

        async def drain(self) -> None:
            if self._block_next_drain:
                self._block_next_drain = False
                draining.set()
                try:
                    await release_drain.wait()
                except asyncio.CancelledError:
                    drain_cancelled.set()
                    raise
            else:
                await super().drain()

    process = FakeACPProcess(extension_version=1)
    client = Client(spawn=lambda _command, _args, _options: process)
    calling: asyncio.Task[dict[str, Any]] | None = None
    try:
        session = await client.create_session(extensions=[ext])
        await process.extension_call("extension.initialize", {
            "capabilities": {"conversations": {"fork": True}}
        })
        process.stdin = BlockedACKStdin(process)
        calling = asyncio.create_task(process.extension_call(
            "extension.tool.execute", {"name": "fork"}, request_id=42
        ))
        await asyncio.wait_for(draining.wait(), timeout=2)
        assert process.server_responses[-1]["result"] == {}
        assert not drain_cancelled.is_set()
        frame = await asyncio.wait_for(process.host_frames.get(), timeout=2)
        assert frame["message"]["parentId"] == 42
        await process.respond_to_host(frame, {"conversationId": "nested response arrived"})
        assert (await calling)["result"] == {"content": "nested response arrived"}
        response = await asyncio.wait_for(session.run_and_wait("reader still active"), timeout=2)
        assert response.stop_reason == "end_turn"
        assert not drain_cancelled.is_set()
        assert not release_drain.is_set()
        await asyncio.wait_for(client.close(), timeout=2)
        assert drain_cancelled.is_set()
        assert not session._rpc._background_tasks
    finally:
        release_drain.set()
        await asyncio.wait_for(client.close(), timeout=2)
        if calling is not None:
            calling.cancel()
            await asyncio.gather(calling, return_exceptions=True)


@pytest.mark.asyncio
async def test_inline_outbound_acp_rejection_retires_channel_without_replay_or_restart() -> None:
    ext = Extension()
    invoked: list[str] = []
    tool_finally = asyncio.Event()

    @ext.tool("fork", description="A rejected relay frame must not be retried")
    async def fork(_input: Any, ctx: ToolContext) -> str:
        invoked.append("fork")
        try:
            return await ctx.fork_conversation()
        finally:
            tool_finally.set()

    rejected: list[Mapping[str, Any]] = []

    class RejectingProcess(FakeACPProcess):
        async def _handle_request(self, request: Mapping[str, Any]) -> None:
            params = request.get("params") or {}
            message = params.get("message") or {}
            if request.get("method") == "kodelet/extensionFrame" and message.get("method"):
                rejected.append(request)
                self._respond_error(request["id"], "runner rejected relay frame")
                return
            await super()._handle_request(request)

    process = RejectingProcess(extension_version=1)
    client = Client(spawn=lambda _command, _args, _options: process)
    try:
        session = await client.create_session(extensions=[ext])
        await process.extension_call("extension.initialize", {
            "capabilities": {"conversations": {"fork": True}}
        })
        relay = session._rpc._extension_relay
        assert relay is not None
        channel = relay._channels[("run-1", "inline-1")]
        ack = await process.extension_frame({
            "jsonrpc": "2.0", "id": 42, "method": "extension.tool.execute",
            "params": {"name": "fork"},
        })
        assert ack["result"] == {}
        close_frame = await asyncio.wait_for(process.host_frames.get(), timeout=2)
        assert close_frame == {
            "sessionId": session.id, "runId": "run-1", "extensionId": "inline-1", "close": True,
        }
        assert tool_finally.is_set()
        assert channel.closed and channel._host_client._closed
        assert not channel._host_client._pending
        assert not relay._channels
        for method in ("extension.initialize", "extension.tool.execute"):
            response = await process.extension_frame({
                "jsonrpc": "2.0", "id": 43, "method": method, "params": {"name": "fork"},
            })
            assert "closed" in response["error"]["message"]
        response = await session.run_and_wait("the ACP connection remains usable")
        assert response.stop_reason == "end_turn"
        assert len(rejected) == 1
        assert invoked == ["fork"]
        assert not process._closed.is_set()
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("destination", ["host", "ui"])
@pytest.mark.parametrize("termination", ["cancel", "reuse", "close", "failure"])
async def test_inline_cancellation_and_teardown_cancel_pending_callbacks(
    destination: str, termination: str
) -> None:
    ext = Extension()
    callback_started, callback_cancelled = asyncio.Event(), asyncio.Event()
    ui_started, ui_cancelled = asyncio.Event(), asyncio.Event()

    async def input_handler(_request: Any) -> str:
        ui_started.set()
        try:
            await asyncio.Event().wait()
            return "unreachable"
        except asyncio.CancelledError:
            ui_cancelled.set()
            raise

    @ext.tool("wait", description="Wait for a pending host or local UI RPC")
    async def wait(_input: Any, ctx: ToolContext) -> str | None:
        callback_started.set()
        try:
            if destination == "ui":
                return await ctx.ui.input({"title": "Blocked"})
            return await ctx.fork_conversation()
        except asyncio.CancelledError:
            callback_cancelled.set()
            raise

    @ext.tool("ping", description="Confirm the channel still dispatches")
    async def ping() -> str:
        return "pong"

    process = FakeACPProcess(extension_version=1)
    client = Client(spawn=lambda _command, _args, _options: process)
    try:
        session = await client.create_session(extensions=[ext], ui={"input": input_handler})
        await process.extension_call("extension.initialize", {
            "capabilities": {"conversations": {"fork": True}}
        })
        ack = await process.extension_frame({
            "jsonrpc": "2.0", "id": 42, "method": "extension.tool.execute",
            "params": {"name": "wait"},
        })
        assert ack["result"] == {}
        await asyncio.wait_for(callback_started.wait(), timeout=2)
        host_frame = None
        if destination == "ui":
            await asyncio.wait_for(ui_started.wait(), timeout=2)
        else:
            host_frame = await asyncio.wait_for(process.host_frames.get(), timeout=2)
        relay = session._rpc._extension_relay
        assert relay is not None
        channel = relay._channels[("run-1", "inline-1")]
        if termination == "cancel":
            ack = await process.extension_frame({
                "jsonrpc": "2.0", "method": "$/cancelRequest", "params": {"id": 42}
            })
            assert ack["result"] == {}
        elif termination == "reuse":
            result = await process.extension_call(
                "extension.tool.execute", {"name": "ping"}, request_id=42
            )
            assert result["result"] == {"content": "pong"}
        elif termination == "close":
            ack = await process.extension_frame(close=True)
            assert ack["result"] == {}
        else:
            assert isinstance(process.stdout, _QueueLineReader)
            process.stdout.feed_eof()
        await asyncio.wait_for(callback_cancelled.wait(), timeout=2)
        if destination == "ui":
            await asyncio.wait_for(ui_cancelled.wait(), timeout=2)
        if termination in {"cancel", "reuse"}:
            if host_frame is not None:
                await process.respond_to_host(host_frame, {"conversationId": "late"})
            ping_result = await process.extension_call(
                "extension.tool.execute", {"name": "ping"}, request_id=43
            )
            assert ping_result["result"] == {"content": "pong"}
            terminals = [
                frame["message"] for frame in process.extension_frames
                if frame.get("message", {}).get("id") == 42
                and not frame["message"].get("method")
            ]
            assert terminals == (
                [{"jsonrpc": "2.0", "id": 42, "result": {"content": "pong"}}]
                if termination == "reuse" else []
            )
        await client.close()
        assert channel._host_client._closed
        assert not channel._host_client._pending
        assert not relay._channels
        assert channel._worker.done()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_same_inline_object_has_isolated_initialization_in_concurrent_sessions() -> None:
    ext = Extension()
    seen: list[tuple[str, str | None]] = []

    @ext.tool("context", description="Read channel initialization and runner RPC")
    async def context(_input: Any, ctx: ToolContext) -> str:
        seen.append((ctx.cwd, ctx.conversation_id))
        return f"{ctx.cwd}:{await ctx.fork_conversation()}"

    processes = [FakeACPProcess(session_id=name, extension_version=1) for name in ("one", "two")]
    spawning = iter(processes)
    client = Client(spawn=lambda _command, _args, _options: next(spawning))
    calls: list[asyncio.Task[dict[str, Any]]] = []
    try:
        sessions = await asyncio.gather(
            client.create_session(extensions=[ext]), client.create_session(extensions=[ext])
        )
        for process in processes:
            await process.extension_call("extension.initialize", {
                "extension": {"cwd": f"/runner/{process._session_id}"},
                "capabilities": {"conversations": {"fork": True}},
            })
        for process in processes:
            calls.append(asyncio.create_task(process.extension_call(
                "extension.tool.execute",
                {"name": "context", "context": {"conversationId": process._session_id}},
            )))
        frames = await asyncio.wait_for(
            asyncio.gather(*(process.host_frames.get() for process in processes)), timeout=2
        )
        assert all(frame["message"]["id"] == frame["message"]["parentId"] == 1 for frame in frames)
        await processes[1].respond_to_host(frames[1], {"conversationId": "fork-two"})
        assert (await calls[1])["result"] == {"content": "/runner/two:fork-two"}
        assert not calls[0].done()
        await sessions[1].close()
        await processes[0].respond_to_host(frames[0], {"conversationId": "fork-one"})
        assert (await calls[0])["result"] == {"content": "/runner/one:fork-one"}
        assert sorted(seen) == [("/runner/one", "one"), ("/runner/two", "two")]
        assert ext._init_params is None
    finally:
        await client.close()
        for call in calls:
            call.cancel()
        await asyncio.gather(*calls, return_exceptions=True)


@pytest.mark.asyncio
async def test_run_and_extension_ids_have_independent_host_rpc_state() -> None:
    ext = Extension()

    @ext.tool("fork", description="Use the channel's independent RPC client")
    async def fork(_input: Any, ctx: ToolContext) -> str:
        return await ctx.fork_conversation()

    process = FakeACPProcess(extension_version=1)
    client = Client(spawn=lambda _command, _args, _options: process)
    calls: dict[tuple[str, str], asyncio.Task[dict[str, Any]]] = {}
    try:
        await client.create_session(extensions=[ext, ext])
        for run_id in ("run-1", "run-2"):
            for extension_id in ("inline-1", "inline-2"):
                await process.extension_call(
                    "extension.initialize", {"capabilities": {"conversations": {"fork": True}}},
                    run_id=run_id, extension_id=extension_id,
                )
                calls[(run_id, extension_id)] = asyncio.create_task(process.extension_call(
                    "extension.tool.execute", {"name": "fork"},
                    run_id=run_id, extension_id=extension_id,
                ))
        frames = [await asyncio.wait_for(process.host_frames.get(), timeout=2) for _ in calls]
        assert all(frame["message"]["id"] == frame["message"]["parentId"] == 1 for frame in frames)
        for frame in reversed(frames):
            key = (frame["runId"], frame["extensionId"])
            result = ":".join(key)
            await process.respond_to_host(frame, {"conversationId": result})
            assert (await calls[key])["result"] == {"content": result}
        assert ext._init_params is None
    finally:
        await client.close()
        for call in calls.values():
            call.cancel()
        await asyncio.gather(*calls.values(), return_exceptions=True)


@pytest.mark.asyncio
async def test_inline_unknown_frames_do_not_initialize_and_closed_channels_cannot_restart() -> None:
    created: list[Extension] = []

    def entrypoint(ext: Extension) -> None:
        created.append(ext)
        ext.tool("ping", description="An actual callback")(lambda: "pong")

    process = FakeACPProcess(extension_version=1)
    client = Client(spawn=lambda _command, _args, _options: process)
    initialize = {"jsonrpc": "2.0", "id": 1, "method": "extension.initialize", "params": {}}
    try:
        session = await client.create_session(extensions=[entrypoint])
        frames = [
            {"message": initialize, "session_id": "unknown"},
            {"message": initialize, "extension_id": "unknown"},
            {"message": {"jsonrpc": "2.0", "id": 1, "result": {}}},
            {"message": {"jsonrpc": "2.0", "method": "$/cancelRequest", "params": {"id": 1}}},
            {"message": {"jsonrpc": "2.0", "id": 1, "method": "extension.tool.execute"}},
            {"message": {"jsonrpc": "2.0", "method": "extension.initialize"}},
            {"close": True},
        ]
        for params in frames:
            ack = await process.extension_frame(**cast(Any, params))
            assert ack["error"]["code"] == -32602
        assert created == []
        relay = session._rpc._extension_relay
        assert relay is not None and not relay._channels
        await process.extension_call("extension.initialize")
        assert len(created) == 1
        duplicate = await process.extension_frame(initialize)
        assert duplicate["error"]["code"] == -32602
        for _ in range(2):
            closed = await process.extension_frame(close=True)
            assert closed["result"] == {}
        assert not relay._channels
        for message in (initialize, {"jsonrpc": "2.0", "id": 1, "result": {}}):
            rejected = await process.extension_frame(message)
            assert "closed" in rejected["error"]["message"]
        assert len(created) == 1
        await process.extension_call("extension.initialize", run_id="run-2")
        assert len(created) == 2
        await session.close()
        assert not relay._channels
        assert process._closed.is_set()
        with pytest.raises(ValueError, match="closed"):
            relay.accept({
                "sessionId": session.id, "runId": "run-3", "extensionId": "inline-1",
                "message": initialize,
            })
        assert len(created) == 2
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_session_rejects_resume_with_inherited_context() -> None:
    class InheritedContext:
        async def fork_conversation(self) -> str:
            return "forked-conversation"

    client = Client()
    with pytest.raises(ValueError, match=r"ctx\.fork_conversation\(\).*resume"):
        await client.create_session(
            resume="existing-conversation",
            inherit_context=cast(Any, InheritedContext()),
        )


@pytest.mark.asyncio
async def test_session_rejects_profile_with_inherited_context() -> None:
    class InheritedContext:
        async def fork_conversation(self) -> str:
            return "forked-conversation"

    client = Client()
    with pytest.raises(ValueError, match=r"ctx\.fork_conversation\(\).*resume"):
        await client.create_session(
            profile="other-profile",
            inherit_context=cast(Any, InheritedContext()),
        )


@pytest.mark.asyncio
async def test_acp_rpc_cancellation_discards_late_response_and_keeps_reader_alive() -> None:
    class DeferredLoadProcess(FakeACPProcess):
        def __init__(self) -> None:
            super().__init__()
            self.load_started = asyncio.Event()
            self.load_request: Mapping[str, Any] | None = None

        async def _handle_request(self, request: Mapping[str, Any]) -> None:
            if request.get("method") == "session/load":
                self.load_request = request
                self.load_started.set()
                return
            await super()._handle_request(request)

    process = DeferredLoadProcess()
    rpc = ACPRPCClient(process)
    load_task = asyncio.create_task(rpc.load_session("forked-conversation", "/workspace"))
    await asyncio.wait_for(process.load_started.wait(), timeout=1)

    load_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await load_task
    assert rpc._pending == {}

    assert process.load_request is not None
    process._respond(process.load_request["id"], {})
    assert await rpc.create_session("/workspace") == "conv-1"
    await rpc.close()


@pytest.mark.asyncio
async def test_session_steers_active_run_and_rejects_blank_messages() -> None:
    prompt_started = asyncio.Event()
    release_prompt = asyncio.Event()
    processes: list[FakeACPProcess] = []

    async def on_prompt(_request: Mapping[str, Any], _process: FakeACPProcess) -> None:
        prompt_started.set()
        await release_prompt.wait()

    def spawn(_command: str, _args: Sequence[str], _options: SpawnOptions) -> FakeACPProcess:
        process = FakeACPProcess(on_prompt=on_prompt)
        processes.append(process)
        return process

    client = Client(spawn=spawn)
    session = await client.create_session()
    run_task = asyncio.create_task(session.run_and_wait(message="inspect the change"))
    await asyncio.wait_for(prompt_started.wait(), timeout=1)

    with pytest.raises(ValueError, match="non-empty"):
        await session.steer("   ")
    result: SessionSteerResult = await session.steer("  focus on the race  ")

    assert result == {"outcome": "injected"}
    steer_requests = [
        request
        for request in processes[0].requests
        if request["method"] == "_session/steering"
    ]
    assert len(steer_requests) == 1
    assert steer_requests[0]["params"] == {
        "sessionId": "conv-1",
        "prompt": [{"type": "text", "text": "focus on the race"}],
        "_meta": {"steering": {"idleBehavior": "promptRequired"}},
    }

    release_prompt.set()
    await run_task
    await client.close()


@pytest.mark.asyncio
async def test_session_steer_requires_active_open_run() -> None:
    process = FakeACPProcess()
    client = Client(spawn=lambda *_args: process)
    session = await client.create_session()

    with pytest.raises(RuntimeError, match="without an active run"):
        await session.steer("focus")
    assert not any(
        request["method"] == "_session/steering" for request in process.requests
    )

    await session.close()
    with pytest.raises(RuntimeError, match="closed"):
        await session.steer("focus")


@pytest.mark.asyncio
async def test_acp_rpc_rejects_malformed_steering_responses() -> None:
    process = FakeACPProcess(
        steer_result={"outcome": "unknown"},
    )
    rpc = ACPRPCClient(process)

    await rpc.initialize()

    with pytest.raises(RuntimeError, match="Invalid _session/steering response"):
        await rpc.steer_session("conv-1", "focus")

    await rpc.close()


@pytest.mark.asyncio
async def test_acp_rpc_requires_advertised_steering_capability() -> None:
    process = FakeACPProcess(steering_supported=False)
    rpc = ACPRPCClient(process)
    await rpc.initialize()

    with pytest.raises(RuntimeError, match="does not advertise session steering support"):
        await rpc.steer_session("conv-1", "focus")
    assert not any(
        request["method"] == "_session/steering" for request in process.requests
    )

    await rpc.close()


@pytest.mark.asyncio
async def test_session_runs_kodelet_acp_json_rpc_and_emits_stream_events() -> None:
    calls: list[dict[str, Any]] = []
    processes: list[FakeACPProcess] = []

    def on_prompt(_request: Mapping[str, Any], child: FakeACPProcess) -> None:
        child.notify(
            "session/update",
            {
                "sessionId": "conv-1",
                "update": {
                    "sessionUpdate": "agent_thought_chunk",
                    "content": {"type": "text", "text": "checking"},
                },
            },
        )
        child.notify(
            "session/update",
            {
                "sessionId": "conv-1",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "forty"},
                },
            },
        )
        child.notify(
            "session/update",
            {
                "sessionId": "conv-1",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": " two"},
                },
            },
        )
        child.notify(
            "session/update",
            {
                "sessionId": "conv-1",
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "call-1",
                    "toolName": "file_read",
                    "title": "Read: /tmp/example.txt",
                    "kind": "read",
                    "rawInput": {"file_path": "/tmp/example.txt"},
                },
            },
        )
        child.notify(
            "session/update",
            {
                "sessionId": "conv-1",
                "update": {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "call-1",
                    "status": "in_progress",
                },
            },
        )
        child.notify(
            "session/update",
            {
                "sessionId": "conv-1",
                "update": {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "call-1",
                    "status": "in_progress",
                    "content": [
                        {
                            "type": "content",
                            "content": {"type": "text", "text": "partial file contents"},
                        }
                    ],
                },
            },
        )
        child.notify(
            "session/update",
            {
                "sessionId": "conv-1",
                "update": {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "call-1",
                    "status": "in_progress",
                    "content": [
                        {
                            "type": "content",
                            "content": {
                                "type": "text",
                                "text": "complete partial file contents",
                            },
                        }
                    ],
                },
            },
        )
        child.notify(
            "session/update",
            {
                "sessionId": "conv-1",
                "update": {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "call-1",
                    "status": "completed",
                    "content": [
                        {
                            "type": "content",
                            "content": {
                                "type": "resource",
                                "resource": {
                                    "uri": "file:///tmp/example.txt",
                                    "mimeType": "text/plain",
                                    "text": "1 | hello",
                                },
                            },
                        }
                    ],
                },
            },
        )

    def spawn(command: str, args: Sequence[str], options: SpawnOptions) -> FakeACPProcess:
        calls.append(
            {
                "command": command,
                "args": list(args),
                "env": options.get("env"),
                "cwd": options.get("cwd"),
            }
        )
        process = FakeACPProcess(on_prompt=on_prompt)
        processes.append(process)
        return process

    client = Client({"command": "kodelet-test", "cwd": "/workspace", "spawn": spawn})
    session = await client.create_session(streaming=True, profile="work", max_turns=2)
    deltas: list[str] = []
    thoughts: list[str] = []
    tool_names: list[str] = []
    tool_updates: list[str] = []
    tool_results: list[str] = []
    session.on("assistant.message_delta", lambda event: deltas.append(event.data.deltaContent))
    session.on("assistant.thinking_delta", lambda event: thoughts.append(event.data.deltaContent))
    session.on("tool.call", lambda event: tool_names.append(event.data.toolName))
    session.on("tool.update", lambda event: tool_updates.append(event.data.result))
    session.on("tool.result", lambda event: tool_results.append(event.data.result))

    response = await session.run_and_wait(message="meaning?", images=["diagram.png"], max_turns=2)

    assert response.content == "forty two"
    assert response.conversationId == "conv-1"
    assert deltas == ["forty", " two"]
    assert thoughts == ["checking"]
    assert tool_names == ["file_read"]
    assert tool_updates == ["partial file contents", "complete partial file contents"]
    assert tool_results == ["1 | hello"]
    recorded_tool_updates = [event for event in response.events if event.type == "tool.update"]
    assert len(recorded_tool_updates) == 1
    update_data: ToolUpdateData = recorded_tool_updates[0].data
    assert update_data["result"] == "complete partial file contents"
    assert update_data["toolCallId"] == "call-1"
    assert update_data["status"] == "in_progress"
    assert response.stopReason == "end_turn"
    assert session.id == "conv-1"
    assert calls[0]["command"] == "kodelet-test"
    assert calls[0]["cwd"] == os.getcwd()
    assert calls[0]["args"] == ["acp", "--max-turns=2", "--profile=work"]
    assert calls[0]["env"].get("KODELET_CONFIG_FILE") is None
    assert [request["method"] for request in processes[0].requests] == [
        "initialize",
        "session/new",
        "session/prompt",
    ]
    assert processes[0].requests[1]["params"]["cwd"] == "/workspace"
    assert processes[0].requests[2]["params"]["prompt"] == [
        {"type": "text", "text": "meaning?"},
        {"type": "image", "uri": "diagram.png"},
    ]

    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("as_mapping", [False, True], ids=["kwargs", "mapping"])
@pytest.mark.parametrize(
    ("options", "error_match"),
    [
        pytest.param(
            {"profile": {"openai": {"api_key": "must-not-forward"}}},
            None,
            id="provider-config",
        ),
        pytest.param({"options": None}, None, id="null-options"),
        *[
            pytest.param({key: settings}, None, id=f"{key}-{name}")
            for key in ("profile", "options")
            for name, settings in (
                ("openai", {"openai": {"platform": "codex"}}),
                ("anthropic", {"anthropic": {"platform": "anthropic"}}),
                ("anthropic-access", {"anthropic_api_access": "subscription"}),
            )
        ],
    ],
)
async def test_session_rejects_unsupported_options_before_spawning(
    options: CreateSessionOptions,
    error_match: str | None,
    as_mapping: bool,
) -> None:
    def spawn(_command: str, _args: Sequence[str], _options: SpawnOptions) -> FakeACPProcess:
        pytest.fail("unsupported options must be rejected before spawning")

    client = Client(spawn=spawn)
    with pytest.raises(ValueError, match=error_match):
        if as_mapping:
            await client.create_session(options)
        else:
            await client.create_session(**options)
    assert not client._sessions
    assert not client._rpcs
    await client.close()

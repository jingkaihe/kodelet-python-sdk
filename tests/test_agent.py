from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import pytest

import kodelet_sdk
from kodelet_sdk import (
    BaseModel,
    Client,
    Extension,
    Profile,
    ToolUpdateData,
    define_extension,
)
from kodelet_sdk.agent import BridgeTransport, SpawnedProcess, SpawnOptions
from kodelet_sdk.agent.bridge import _BridgeConnection, _BridgeRequestState


class FakeACPProcess(SpawnedProcess):
    def __init__(
        self,
        *,
        session_id: str = "conv-1",
        on_prompt: Callable[[Mapping[str, Any], FakeACPProcess], Awaitable[None] | None]
        | None = None,
    ) -> None:
        self.stdout = _QueueLineReader()
        self.stderr = _QueueLineReader()
        self.stdin = _FakeStdin(self)
        self.requests: list[dict[str, Any]] = []
        self._session_id = session_id
        self._on_prompt = on_prompt
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

    def handle_input(self, chunk: bytes) -> None:
        for line in chunk.decode("utf-8").splitlines():
            if not line.strip():
                continue
            request = json.loads(line)
            if not isinstance(request, dict):
                continue
            if not request.get("method") or request.get("id") is None:
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
                {"protocolVersion": 1, "agentCapabilities": {}, "authMethods": []},
            )
            return
        if method == "session/new":
            self._respond(request_id, {"sessionId": self._session_id})
            return
        if method == "session/load":
            self._respond(request_id, {})
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
    assert kodelet_sdk.Client is Client
    assert client_module.Client is Client


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


@pytest.mark.asyncio
async def test_session_writes_inline_profile_to_temporary_override_config() -> None:
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
            "openai": {"api_mode": "responses", "service_tier": "fast"},
        }
    )

    env = calls[0]["env"]
    assert env["KODELET_CONFIG_FILE_MODE"] == "isolated"
    assert env.get("KODELET_MODEL") is None
    config_path = env["KODELET_CONFIG_FILE"]
    assert calls[0]["args"] == ["acp"]
    assert json.loads(await _read_text(Path(config_path))) == {
        "name": "openai",
        "provider": "openai",
        "model": "gpt-5.5",
        "allowed_tools": ["sdk_echo"],
        "openai": {"api_mode": "responses", "service_tier": "fast"},
        "profile": "default",
    }

    await session.close()
    assert not await _exists(Path(config_path))


@pytest.mark.asyncio
async def test_inline_profile_isolation_filters_ambient_kodelet_environment(
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
    assert env.get("KODELET_MODEL") is None
    assert env["KODELET_PROVIDER"] == "explicit-provider"
    assert env["KODELET_CONFIG_FILE_MODE"] == "isolated"
    await client.close()


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
    assert calls[0]["cwd"] == "/workspace"
    assert calls[0]["args"] == ["--profile", "work", "acp", "--max-turns", "2"]
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
async def test_session_exposes_in_process_extensions_through_temporary_json_rpc_bridge(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []

    def spawn(_command: str, args: Sequence[str], options: SpawnOptions) -> FakeACPProcess:
        calls.append({"args": list(args), "env": options.get("env")})
        return FakeACPProcess(
            session_id="conv-ext",
            on_prompt=lambda _request, child: child.notify(
                "session/update",
                {
                    "sessionId": "conv-ext",
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": "done"},
                    },
                },
            ),
        )

    ext = Extension(name="workspace")

    class AskInput(BaseModel):
        question: str
        options: list[str]

    @ext.tool("ask_user_question", description="Ask a question", input_schema=AskInput)
    async def ask_user_question(input: AskInput, ctx: Any) -> str:
        selected = await ctx.ui.select({"title": input.question, "options": input.options})
        return selected or "dismissed"

    client = Client(cwd=tmp_path, spawn=spawn)
    session = await client.create_session(
        extensions=[ext],
        ui={"select": lambda request: request["options"][0]},
    )
    await session.run_and_wait(message="hello")

    env = calls[0]["env"]
    assert env["KODELET_CONFIG_FILE_MODE"] == "merge"
    config_path = Path(env["KODELET_CONFIG_FILE"])
    config = json.loads(await _read_text(config_path))
    assert config["extensions"]["enabled"] is True
    extension_root = Path(config["extensions"]["local_dir"])
    assert config["extensions"]["allow"] == [str(extension_root)]
    assert await _is_dir(extension_root)
    extension_executables = [
        entry.name
        for entry in await _iterdir(extension_root)
        if entry.name.startswith("kodelet-extension-")
    ]
    assert len(extension_executables) == 1
    assert calls[0]["args"] == ["acp"]

    await client.close()
    assert not await _exists(config_path)
    assert not await _exists(extension_root)


@pytest.mark.asyncio
@pytest.mark.parametrize("extension_transport", ["unix", "tcp"])
async def test_extension_bridge_routes_local_ui_handlers(
    tmp_path: Path,
    extension_transport: BridgeTransport,
) -> None:
    selected_values: list[str] = []
    extension_root_holder: dict[str, Path] = {}

    def spawn(_command: str, _args: Sequence[str], options: SpawnOptions) -> FakeACPProcess:
        env = options.get("env") or {}
        config = json.loads(Path(env["KODELET_CONFIG_FILE"]).read_text(encoding="utf-8"))
        extension_root_holder["path"] = Path(config["extensions"]["local_dir"])
        return FakeACPProcess(session_id="conv-ui")

    def extension_entrypoint(ext: Extension) -> None:
        ext.set_metadata(name="workspace")

        class AskInput(BaseModel):
            question: str
            options: list[str]

        @ext.tool("ask_user_question", description="Ask a question", input_schema=AskInput)
        async def ask_user_question(input: AskInput, ctx: Any) -> str:
            await ctx.update("Waiting for selection", {"step": 1})
            selected = await ctx.ui.select({"title": input.question, "options": input.options})
            selected_values.append(selected or "")
            return selected or "dismissed"

    client = Client(cwd=tmp_path, spawn=spawn)
    session = await client.create_session(
        extensions=[define_extension(extension_entrypoint)],
        extension_transport=extension_transport,
        ui={"select": lambda request: request["options"][1]},
    )

    executable = next(extension_root_holder["path"].glob("kodelet-extension-*"))
    assert f"'transport': '{extension_transport}'" in await _read_text(executable)
    process = await asyncio.create_subprocess_exec(
        str(executable),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    await _write_frame(
        process.stdin,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "extension.initialize",
            "params": {
                "extension": {"id": "workspace", "cwd": str(tmp_path)},
                "capabilities": {"toolUpdates": True},
            },
        },
    )
    init_response = await _read_frame(process.stdout)
    assert init_response["result"]["name"] == "workspace"

    await _write_frame(
        process.stdin,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "extension.tool.execute",
            "params": {
                "name": "ask_user_question",
                "input": {"question": "Pick", "options": ["A", "B"]},
            },
        },
    )
    update_request = await _read_frame(process.stdout)
    assert update_request == {
        "jsonrpc": "2.0",
        "id": 1,
        "parentId": 2,
        "method": "kodelet.tool.update",
        "params": {"content": "Waiting for selection", "data": {"step": 1}},
    }
    await _write_frame(
        process.stdin,
        {"jsonrpc": "2.0", "id": update_request["id"], "result": {"accepted": True}},
    )
    tool_response = await _read_frame(process.stdout)
    assert tool_response["result"] == {"content": "B"}
    assert selected_values == ["B"]

    process.terminate()
    await process.wait()
    await session.close()


@pytest.mark.asyncio
async def test_extension_bridge_cancellation_is_connection_scoped(tmp_path: Path) -> None:
    extension_root_holder: dict[str, Path] = {}

    def spawn(_command: str, _args: Sequence[str], options: SpawnOptions) -> FakeACPProcess:
        env = options.get("env") or {}
        config = json.loads(Path(env["KODELET_CONFIG_FILE"]).read_text(encoding="utf-8"))
        extension_root_holder["path"] = Path(config["extensions"]["local_dir"])
        return FakeACPProcess(session_id="conv-cancel")

    started = {"cancel": asyncio.Event(), "disconnect": asyncio.Event()}
    aborted = {"cancel": asyncio.Event(), "disconnect": asyncio.Event()}
    stale_blocked = {"cancel": asyncio.Event(), "disconnect": asyncio.Event()}
    ui_started = asyncio.Event()
    ui_cancelled = asyncio.Event()
    notifications: list[str] = []
    detached_tasks: set[asyncio.Task[None]] = set()

    class WaitInput(BaseModel):
        mode: Literal["cancel", "detached", "disconnect", "quick", "ui"]

    def extension_entrypoint(ext: Extension) -> None:
        @ext.tool("wait_for_cancel", description="Wait for cancellation", input_schema=WaitInput)
        async def wait_for_cancel(input: WaitInput, ctx: Any) -> str:
            if input.mode == "quick":
                return "quick result"
            if input.mode == "detached":
                async def notify_later() -> None:
                    await asyncio.sleep(0.02)
                    try:
                        await ctx.ui.notify({"message": "late notification"})
                    except (asyncio.CancelledError, RuntimeError):
                        pass

                task = asyncio.create_task(notify_later())
                detached_tasks.add(task)
                task.add_done_callback(detached_tasks.discard)
                return "detached result"
            if input.mode == "ui":
                await ctx.ui.notify({"message": "blocking notification"})
                return "ui result"
            started[input.mode].set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                aborted[input.mode].set()
                try:
                    await ctx.update(f"stale {input.mode}")
                except (asyncio.CancelledError, RuntimeError):
                    stale_blocked[input.mode].set()
                return f"late {input.mode}"
            raise AssertionError("wait completed without cancellation")

    async def notify(request: Mapping[str, Any]) -> None:
        if request["message"] != "blocking notification":
            notifications.append(str(request["message"]))
            return
        ui_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            ui_cancelled.set()
            raise

    client = Client(cwd=tmp_path, spawn=spawn)
    session = await client.create_session(
        extensions=[define_extension(extension_entrypoint)],
        extension_transport="tcp",
        ui={"notify": notify},
    )
    executable = next(extension_root_holder["path"].glob("kodelet-extension-*"))

    process = await asyncio.create_subprocess_exec(
        str(executable),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    await _write_frame(
        process.stdin,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "extension.initialize",
            "params": {
                "extension": {"id": "cancellable", "cwd": str(tmp_path)},
                "capabilities": {"toolUpdates": True},
            },
        },
    )
    await _read_frame(process.stdout)

    await _write_frame(
        process.stdin,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "extension.tool.execute",
            "params": {"name": "wait_for_cancel", "input": {"mode": "detached"}},
        },
    )
    detached_response = await _read_frame(process.stdout)
    assert detached_response["result"] == {"content": "detached result"}
    await asyncio.sleep(0.05)
    assert notifications == []

    await _write_frame(
        process.stdin,
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "extension.tool.execute",
            "params": {"name": "wait_for_cancel", "input": {"mode": "ui"}},
        },
    )
    await asyncio.wait_for(ui_started.wait(), timeout=2)
    await _write_frame(
        process.stdin,
        {"jsonrpc": "2.0", "method": "$/cancelRequest", "params": {"id": 3}},
    )
    await asyncio.wait_for(ui_cancelled.wait(), timeout=2)

    await _write_frame(
        process.stdin,
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "extension.tool.execute",
            "params": {"name": "wait_for_cancel", "input": {"mode": "cancel"}},
        },
    )
    await asyncio.wait_for(started["cancel"].wait(), timeout=2)
    await _write_frame(
        process.stdin,
        {"jsonrpc": "2.0", "method": "$/cancelRequest", "params": {"id": 4}},
    )
    await asyncio.wait_for(aborted["cancel"].wait(), timeout=2)
    await asyncio.wait_for(stale_blocked["cancel"].wait(), timeout=2)

    await _write_frame(
        process.stdin,
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "extension.tool.execute",
            "params": {"name": "wait_for_cancel", "input": {"mode": "quick"}},
        },
    )
    reused_response = await _read_frame(process.stdout)
    assert reused_response["result"] == {"content": "quick result"}

    await _write_frame(
        process.stdin,
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "extension.tool.execute",
            "params": {"name": "wait_for_cancel", "input": {"mode": "disconnect"}},
        },
    )
    await asyncio.wait_for(started["disconnect"].wait(), timeout=2)
    process.terminate()
    await process.wait()
    await asyncio.wait_for(aborted["disconnect"].wait(), timeout=2)
    await asyncio.wait_for(stale_blocked["disconnect"].wait(), timeout=2)

    replacement = await asyncio.create_subprocess_exec(
        str(executable),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert replacement.stdin is not None
    assert replacement.stdout is not None
    await _write_frame(
        replacement.stdin,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "extension.initialize",
            "params": {
                "extension": {"id": "cancellable", "cwd": str(tmp_path)},
                "capabilities": {"toolUpdates": True},
            },
        },
    )
    await _read_frame(replacement.stdout)
    await _write_frame(
        replacement.stdin,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "extension.tool.execute",
            "params": {"name": "wait_for_cancel", "input": {"mode": "quick"}},
        },
    )
    response = await _read_frame(replacement.stdout)
    assert response["result"] == {"content": "quick result"}

    replacement.terminate()
    await replacement.wait()
    await session.close()


@pytest.mark.asyncio
async def test_extension_bridge_rechecks_terminal_generation_inside_write_lock() -> None:
    writer = _MemoryStreamWriter()
    connection = _BridgeConnection(cast(asyncio.StreamWriter, writer))
    state = _BridgeRequestState(7)
    state.finish()
    connection.requests[7] = state
    await connection.write_lock.acquire()

    task = asyncio.create_task(
        connection.send(
            {"jsonrpc": "2.0", "id": 7, "result": {"content": "old"}},
            state,
            terminal=True,
        )
    )
    await asyncio.sleep(0)
    state.cancel()
    connection.requests[7] = _BridgeRequestState(7)
    connection.write_lock.release()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert writer.buffer == bytearray()


async def _write_frame(writer: asyncio.StreamWriter, message: Mapping[str, Any]) -> None:
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    writer.write(b"Content-Length: " + str(len(payload)).encode("ascii") + b"\r\n\r\n" + payload)
    await writer.drain()


class _MemoryStreamWriter:
    def __init__(self) -> None:
        self.buffer = bytearray()

    def write(self, data: bytes) -> None:
        self.buffer.extend(data)

    async def drain(self) -> None:
        await asyncio.sleep(0)

    def close(self) -> None:
        return None

    async def wait_closed(self) -> None:
        await asyncio.sleep(0)


async def _read_frame(reader: asyncio.StreamReader) -> dict[str, Any]:
    header_lines: list[bytes] = []
    while True:
        line = await reader.readline()
        assert line != b""
        if line in (b"\r\n", b"\n"):
            break
        header_lines.append(line)
    content_length = _content_length(b"".join(header_lines).decode("ascii"))
    payload = await reader.readexactly(content_length)
    return json.loads(payload.decode("utf-8"))


def _content_length(header: str) -> int:
    for line in header.splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() == "content-length":
            return int(value.strip())
    raise AssertionError("missing content length")


async def _read_text(path: Path) -> str:
    return await asyncio.to_thread(path.read_text, encoding="utf-8")


async def _exists(path: Path) -> bool:
    return await asyncio.to_thread(path.exists)


async def _is_dir(path: Path) -> bool:
    return await asyncio.to_thread(path.is_dir)


async def _iterdir(path: Path) -> list[Path]:
    return await asyncio.to_thread(lambda: list(path.iterdir()))

from __future__ import annotations

import asyncio
import inspect
import json
import os
import shutil
import stat
import tempfile
import uuid
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, TypeAlias, TypedDict, Unpack, cast

from ._utils import AttrDict, maybe_await
from .api import Entrypoint, Extension, create_extension_host
from .context import (
    HostRPCClient,
    UIConfirmRequest,
    UIInputRequest,
    UINotifyRequest,
    UISelectRequest,
    run_with_host_rpc_client,
)

ACP_PROTOCOL_VERSION = 1

ProfileInput: TypeAlias = Mapping[str, Any]
BridgeTransport: TypeAlias = str


class AgentUIHandlers(TypedDict, total=False):
    """SDK-provided handlers for extension UI requests in agent sessions."""

    input: Callable[[UIInputRequest], Awaitable[str | None] | str | None]
    confirm: Callable[[UIConfirmRequest], Awaitable[bool] | bool]
    select: Callable[[UISelectRequest], Awaitable[str | None] | str | None]
    notify: Callable[[UINotifyRequest], Awaitable[None] | None]


class CreateSessionOptions(TypedDict, total=False):
    """Options accepted by :meth:`Client.create_session`."""

    profile: str | Profile | ProfileInput
    extensions: Sequence[Entrypoint | Extension]
    streaming: bool
    cwd: str
    resume: str
    max_turns: int
    extension_transport: BridgeTransport
    ui: AgentUIHandlers


class RunOptions(TypedDict, total=False):
    """Options accepted by :meth:`Session.run_and_wait`."""

    message: str
    images: Sequence[str]
    max_turns: int


class AssistantMessageDeltaData(TypedDict):
    deltaContent: str


class AssistantMessageData(TypedDict):
    content: str


class AssistantThinkingDeltaData(TypedDict):
    deltaContent: str


class ToolCallData(TypedDict, total=False):
    toolName: str
    input: Any
    rawInput: str
    toolCallId: str


class ToolResultData(TypedDict, total=False):
    toolName: str
    result: str
    toolCallId: str
    status: str


class SpawnOptions(TypedDict, total=False):
    cwd: str
    env: Mapping[str, str]
    stdio: Sequence[str]


class BinaryLineReader(Protocol):
    async def readline(self) -> bytes: ...


class BinaryWriter(Protocol):
    def write(self, data: bytes, /) -> object: ...

    def close(self) -> object: ...


class DrainableBinaryWriter(BinaryWriter, Protocol):
    async def drain(self) -> None: ...


class SpawnedProcess(Protocol):
    """Minimal subprocess protocol used by the ACP client."""

    stdin: BinaryWriter | None
    stdout: BinaryLineReader | None
    stderr: BinaryLineReader | None

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    async def wait(self) -> int: ...


SpawnFunction: TypeAlias = Callable[
    [str, Sequence[str], SpawnOptions], Awaitable[SpawnedProcess] | SpawnedProcess
]


class ClientOptions(TypedDict, total=False):
    """Options accepted by :class:`Client`."""

    command: str
    cwd: str | os.PathLike[str]
    env: Mapping[str, str | None]
    spawn: SpawnFunction


class AgentStreamEvent(AttrDict):
    """Event emitted by :class:`Session` while an agent run is in progress."""

    @property
    def conversation_id(self) -> str | None:
        return cast(str | None, self.get("conversationId"))


class AgentResponse(AttrDict):
    """Final response returned by :meth:`Session.run_and_wait`."""

    @property
    def conversation_id(self) -> str | None:
        return cast(str | None, self.get("conversationId"))

    @property
    def exit_code(self) -> int:
        return int(self.get("exitCode") or 0)

    @property
    def stop_reason(self) -> str | None:
        return cast(str | None, self.get("stopReason"))


class AgentRunError(RuntimeError):
    """Raised when the launched ``kodelet acp`` process exits unexpectedly."""

    def __init__(
        self,
        message: str,
        *,
        code: int | None,
        signal: str | None,
        stderr: str,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.signal = signal
        self.stderr = stderr


class RPCError(RuntimeError):
    def __init__(self, error: Mapping[str, Any]) -> None:
        super().__init__(str(error.get("message") or "JSON-RPC error"))
        self.code = int(error.get("code") or 0)
        self.data = error.get("data")


class Profile:
    """Named or inline Kodelet profile used when launching agent sessions."""

    def __init__(self, config: str | ProfileInput) -> None:
        if isinstance(config, str):
            self.name: str | None = config
            self.config: dict[str, Any] = {"name": config}
            return

        normalized = dict(config)
        if normalized.get("provider") is None and isinstance(normalized.get("profiler"), str):
            normalized["provider"] = normalized["profiler"]
        normalized.pop("profiler", None)
        self.name = normalized.get("name") if isinstance(normalized.get("name"), str) else None
        self.config = normalized

    @classmethod
    def named(cls, name: str) -> Profile:
        """Create a profile reference that launches Kodelet with ``--profile``."""

        return cls(name)

    def is_named_only(self) -> bool:
        return all(key == "name" for key in self.config)

    def isNamedOnly(self) -> bool:
        """CamelCase alias for :meth:`is_named_only`."""

        return self.is_named_only()

    def to_launch_config(self) -> dict[str, Any]:
        """Return command-line args and optional inline config for this profile."""

        if self.name and self.is_named_only():
            return {"args": ["--profile", self.name]}
        return {"args": [], "config": dict(self.config)}

    def toLaunchConfig(self) -> dict[str, Any]:
        """CamelCase alias for :meth:`to_launch_config`."""

        return self.to_launch_config()


class Client:
    """Launch and manage Kodelet agent sessions over ACP JSON-RPC."""

    def __init__(
        self,
        options: ClientOptions | None = None,
        *,
        command: str | None = None,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str | None] | None = None,
        spawn: SpawnFunction | None = None,
    ) -> None:
        resolved_options: dict[str, Any] = dict(options or {})
        if command is not None:
            resolved_options["command"] = command
        if cwd is not None:
            resolved_options["cwd"] = cwd
        if env is not None:
            resolved_options["env"] = env
        if spawn is not None:
            resolved_options["spawn"] = spawn

        self._command = str(resolved_options.get("command") or "kodelet")
        self._cwd = _resolve_path(str(resolved_options.get("cwd") or os.getcwd()))
        self._env = dict(cast(Mapping[str, str | None], resolved_options.get("env") or {}))
        self._spawn = (
            cast(SpawnFunction | None, resolved_options.get("spawn")) or self._default_spawn
        )
        self._sessions: set[Session] = set()

    async def create_session(
        self,
        options: CreateSessionOptions | None = None,
        **kwargs: Unpack[CreateSessionOptions],
    ) -> Session:
        """Create a new Kodelet ACP session.

        Keyword arguments mirror :class:`CreateSessionOptions`; passing a mapping
        as the first argument is also accepted for parity with the TypeScript SDK.
        """

        merged_options: dict[str, Any] = {**dict(options or {}), **kwargs}
        extensions = cast(
            Sequence[Entrypoint | Extension] | None,
            merged_options.get("extensions"),
        )
        ui = cast(AgentUIHandlers | None, merged_options.get("ui"))
        extension_transport = _normalize_bridge_transport(
            merged_options.get("extension_transport")
        )
        bridge = (
            await InMemoryExtensionBridge.create(
                extensions,
                {"ui": ui, "transport": extension_transport},
            )
            if extensions
            else None
        )
        cwd = _resolve_path(str(merged_options.get("cwd") or self._cwd))
        profile = _normalize_profile(merged_options.get("profile"))
        launch: LaunchConfig | None = None
        rpc: ACPRPCClient | None = None
        try:
            launch = await _build_launch_config(profile, bridge)
            env = _clean_env(
                {
                    **self._base_env(isolate_kodelet_env=launch.config_file_mode == "isolated"),
                    **launch.env,
                }
            )
            args = [*launch.args, "acp", *_acp_server_args(merged_options)]
            process = await self._spawn_process(
                args,
                {"cwd": cwd, "env": env, "stdio": ["pipe"] * 3},
            )
            rpc = ACPRPCClient(process)
            await rpc.initialize()
            resume = merged_options.get("resume")
            session_id = (
                await rpc.load_session(str(resume), cwd)
                if isinstance(resume, str) and resume
                else await rpc.create_session(cwd)
            )
            session = Session(
                self,
                cwd=cwd,
                session_id=session_id,
                rpc=rpc,
                max_turns=cast(int | None, merged_options.get("max_turns")),
                extension_bridge=bridge,
                temp_config=launch.temp_config,
            )
            self._sessions.add(session)
            return session
        except Exception:
            if rpc is not None:
                await rpc.close()
            if bridge is not None:
                await bridge.close()
            if launch is not None and launch.temp_config is not None:
                await launch.temp_config.close()
            raise

    async def createSession(self, *args: Any, **kwargs: Any) -> Session:
        """CamelCase alias for :meth:`create_session`."""

        return await self.create_session(*args, **kwargs)

    async def close(self) -> None:
        """Close all sessions owned by this client."""

        await asyncio.gather(*(session.close() for session in list(self._sessions)))
        self._sessions.clear()

    def _base_env(self, *, isolate_kodelet_env: bool = False) -> dict[str, str | None]:
        env: dict[str, str | None] = dict(os.environ)
        if isolate_kodelet_env:
            for key in list(env):
                if key.startswith("KODELET_"):
                    del env[key]
        env.update(self._env)
        return env

    async def _spawn_process(self, args: Sequence[str], options: SpawnOptions) -> SpawnedProcess:
        result = self._spawn(self._command, args, options)
        if inspect.isawaitable(result):
            return cast(SpawnedProcess, await result)
        return result

    async def _default_spawn(
        self,
        command: str,
        args: Sequence[str],
        options: SpawnOptions,
    ) -> SpawnedProcess:
        process = await asyncio.create_subprocess_exec(
            command,
            *args,
            cwd=options.get("cwd"),
            env=dict(options.get("env") or {}),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return cast(SpawnedProcess, process)

    def _delete_session(self, session: Session) -> None:
        self._sessions.discard(session)


EventListener: TypeAlias = Callable[[AgentStreamEvent], Any]


class Session:
    """A single Kodelet ACP conversation session."""

    def __init__(
        self,
        client: Client,
        *,
        cwd: str,
        session_id: str,
        rpc: ACPRPCClient,
        max_turns: int | None = None,
        extension_bridge: InMemoryExtensionBridge | None = None,
        temp_config: TempConfig | None = None,
    ) -> None:
        self.cwd = cwd
        self._client = client
        self._rpc = rpc
        self._max_turns = max_turns
        self._conversation_id = session_id
        self._extension_bridge = extension_bridge
        self._temp_config = temp_config
        self._closed = False
        self._running = False
        self._listeners: dict[str, list[EventListener]] = {}
        self._listener_tasks: set[asyncio.Task[Any]] = set()

    @property
    def id(self) -> str:
        """Current Kodelet conversation/session id."""

        return self._conversation_id

    def on(self, event_name: str, listener: EventListener) -> Session:
        """Register a listener for SDK stream events."""

        self._listeners.setdefault(event_name, []).append(listener)
        return self

    def once(self, event_name: str, listener: EventListener) -> Session:
        """Register a listener that is removed after its first event."""

        def wrapper(event: AgentStreamEvent) -> Any:
            self.off(event_name, wrapper)
            return listener(event)

        return self.on(event_name, wrapper)

    def off(self, event_name: str, listener: EventListener) -> Session:
        """Remove a previously registered listener."""

        listeners = self._listeners.get(event_name)
        if listeners and listener in listeners:
            listeners.remove(listener)
        return self

    async def run_and_wait(
        self,
        options: RunOptions | str | None = None,
        **kwargs: Unpack[RunOptions],
    ) -> AgentResponse:
        """Run a prompt and wait for the final ACP response."""

        run_options = _normalize_run_options(options, kwargs)
        message = run_options["message"]
        max_turns = _option_int(run_options, "max_turns")
        if self._closed:
            raise RuntimeError("Cannot run a closed Kodelet session")
        if self._running:
            raise RuntimeError("Cannot run a Kodelet session while another run is in progress")
        if max_turns is not None and max_turns != self._max_turns:
            raise RuntimeError(
                "Per-run max_turns is not supported by the RPC transport; "
                "set max_turns in create_session instead"
            )

        self._running = True
        events: list[AgentStreamEvent] = []
        assistant_chunks: list[str] = []
        thinking_active = {"value": False}
        tool_names: dict[str, str] = {}

        def notification_handler(method: str, params: Any) -> None:
            if method == "session/update":
                self._handle_session_update(
                    params,
                    events,
                    assistant_chunks,
                    thinking_active,
                    tool_names,
                )

        unsubscribe = self._rpc.on_notification(notification_handler)
        self._emit_sdk_event("agent.start", {"message": message}, events)
        self._emit_sdk_event("user.message", {"content": message}, events)

        try:
            prompt_blocks = _build_prompt_blocks(run_options)
            result = await self._rpc.prompt(self._conversation_id, prompt_blocks)
            if thinking_active["value"]:
                thinking_active["value"] = False
                self._emit_sdk_event("assistant.thinking_end", {}, events)
            self._emit_sdk_event("assistant.content_end", {}, events)
            content = "".join(assistant_chunks)
            if content:
                self._emit_sdk_event("assistant.message", {"content": content}, events)
            response = AgentResponse(
                {
                    "content": content,
                    "conversationId": self._conversation_id,
                    "events": events,
                    "exitCode": 0,
                    "stopReason": result.get("stopReason"),
                }
            )
            self._emit_sdk_event("agent.end", {**response, "events": [*events]}, events)
            return response
        except asyncio.CancelledError:
            self._rpc.cancel_session(self._conversation_id)
            self._emit_sdk_event("agent.error", {"message": "cancelled"}, events)
            raise
        except Exception as exc:
            self._emit_sdk_event("agent.error", {"message": str(exc)}, events)
            raise
        finally:
            unsubscribe()
            self._running = False

    async def runAndWait(self, *args: Any, **kwargs: Any) -> AgentResponse:
        """CamelCase alias for :meth:`run_and_wait`."""

        return await self.run_and_wait(*args, **kwargs)

    async def close(self) -> None:
        """Close the underlying ACP process and temporary session resources."""

        if self._closed:
            return
        self._closed = True
        await self._rpc.close()
        if self._extension_bridge is not None:
            await self._extension_bridge.close()
        if self._temp_config is not None:
            await self._temp_config.close()
        for task in list(self._listener_tasks):
            task.cancel()
        self._client._delete_session(self)

    def _handle_session_update(
        self,
        params: Any,
        events: list[AgentStreamEvent],
        assistant_chunks: list[str],
        thinking_active: dict[str, bool],
        tool_names: dict[str, str],
    ) -> None:
        if not isinstance(params, Mapping):
            return
        if params.get("sessionId") != self._conversation_id:
            return
        update = params.get("update")
        if not isinstance(update, Mapping):
            return

        session_update = update.get("sessionUpdate")
        if session_update == "agent_message_chunk":
            content = _text_from_acp_content(update.get("content"))
            if content:
                assistant_chunks.append(content)
                self._emit_sdk_event(
                    "assistant.message_delta", {"deltaContent": content}, events, update
                )
            return
        if session_update == "agent_thought_chunk":
            if not thinking_active["value"]:
                thinking_active["value"] = True
                self._emit_sdk_event("assistant.thinking_start", {}, events, update)
            content = _text_from_acp_content(update.get("content"))
            if content:
                self._emit_sdk_event(
                    "assistant.thinking_delta", {"deltaContent": content}, events, update
                )
            return
        if session_update == "tool_call":
            tool_call_id = _string_field(update, "toolCallId")
            tool_name = _tool_name_from_update(update)
            if tool_call_id and tool_name:
                tool_names[tool_call_id] = tool_name
            raw_input = update.get("rawInput")
            raw_input_text = (
                raw_input
                if isinstance(raw_input, str)
                else json.dumps(raw_input if raw_input is not None else None, separators=(",", ":"))
            )
            self._emit_sdk_event(
                "tool.call",
                {
                    "toolName": tool_name,
                    "input": raw_input,
                    "rawInput": raw_input_text,
                    **({"toolCallId": tool_call_id} if tool_call_id else {}),
                },
                events,
                update,
            )
            return
        if session_update == "tool_call_update":
            status_value = _string_field(update, "status")
            if status_value not in {"completed", "failed"}:
                return
            tool_call_id = _string_field(update, "toolCallId")
            self._emit_sdk_event(
                "tool.result",
                {
                    "toolName": (tool_names.get(tool_call_id or "") if tool_call_id else None)
                    or _tool_name_from_update(update),
                    "result": _tool_content_to_text(update.get("content")),
                    **({"toolCallId": tool_call_id} if tool_call_id else {}),
                    "status": status_value,
                },
                events,
                update,
            )
            return

        self._emit_sdk_event("event", dict(update), events, update)

    def _emit_sdk_event(
        self,
        event_type: str,
        data: Any,
        events: list[AgentStreamEvent],
        raw: Any | None = None,
    ) -> AgentStreamEvent:
        event = AgentStreamEvent(
            {
                "type": event_type,
                "data": _to_attr_dict(data),
                "conversationId": self._conversation_id,
                **({"raw": raw} if raw is not None else {}),
            }
        )
        events.append(event)
        self._emit(event_type, event)
        if event_type != "event":
            self._emit("event", event)
        return event

    def _emit(self, event_name: str, event: AgentStreamEvent) -> None:
        for listener in list(self._listeners.get(event_name, [])):
            result = listener(event)
            if inspect.isawaitable(result):
                task = asyncio.create_task(cast(Coroutine[Any, Any, Any], result))
                self._listener_tasks.add(task)
                task.add_done_callback(self._listener_tasks.discard)


class ACPRPCClient:
    def __init__(self, process: SpawnedProcess) -> None:
        if process.stdin is None:
            raise RuntimeError("kodelet acp process did not expose stdin")
        self._process = process
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._stderr_chunks: list[str] = []
        self._notification_handlers: set[Callable[[str, Any], None]] = set()
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._closed = False
        self._stdout_task = asyncio.create_task(self._read_stdout()) if process.stdout else None
        self._stderr_task = asyncio.create_task(self._read_stderr()) if process.stderr else None
        self._wait_task = asyncio.create_task(self._wait_for_process())

    async def initialize(self) -> None:
        await self.request(
            "initialize",
            {
                "protocolVersion": ACP_PROTOCOL_VERSION,
                "clientCapabilities": {
                    "terminal": True,
                    "fs": {"readTextFile": False, "writeTextFile": False},
                },
                "clientInfo": {"name": "kodelet-sdk", "title": "Kodelet SDK"},
            },
        )

    async def create_session(self, cwd: str) -> str:
        result = await self.request("session/new", {"cwd": cwd})
        if not isinstance(result, Mapping) or not isinstance(result.get("sessionId"), str):
            raise RuntimeError("Invalid session/new response from kodelet acp")
        return str(result["sessionId"])

    async def load_session(self, session_id: str, cwd: str) -> str:
        await self.request("session/load", {"sessionId": session_id, "cwd": cwd})
        return session_id

    async def prompt(self, session_id: str, prompt: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        result = await self.request(
            "session/prompt",
            {"sessionId": session_id, "prompt": list(prompt)},
        )
        if not isinstance(result, Mapping):
            return {}
        stop_reason = result.get("stopReason")
        return {"stopReason": stop_reason} if isinstance(stop_reason, str) else {}

    def cancel_session(self, session_id: str) -> None:
        self.notify("session/cancel", {"sessionId": session_id})

    def on_notification(self, handler: Callable[[str, Any], None]) -> Callable[[], None]:
        self._notification_handlers.add(handler)

        def unsubscribe() -> None:
            self._notification_handlers.discard(handler)

        return unsubscribe

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._process.terminate()
        try:
            await asyncio.wait_for(self._process.wait(), timeout=1)
        except TimeoutError:
            self._process.kill()
            await self._process.wait()
        for task in (self._stdout_task, self._stderr_task, self._wait_task):
            if task is not None and not task.done():
                task.cancel()
        for task in list(self._background_tasks):
            task.cancel()

    async def request(self, method: str, params: Any | None = None) -> Any:
        if self._closed:
            raise RuntimeError("kodelet acp process is closed")
        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._write(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            )
        except Exception:
            self._pending.pop(request_id, None)
            raise
        return await future

    def notify(self, method: str, params: Any | None = None) -> None:
        if not self._closed:
            task = asyncio.create_task(
                self._write({"jsonrpc": "2.0", "method": method, "params": params})
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def _write(self, message: Mapping[str, Any]) -> None:
        stdin = self._process.stdin
        if stdin is None:
            raise RuntimeError("kodelet acp process stdin is closed")
        payload = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
        stdin.write(payload)
        drain = getattr(stdin, "drain", None)
        if drain is not None:
            result = drain()
            if inspect.isawaitable(result):
                await result

    async def _read_stdout(self) -> None:
        stdout = self._process.stdout
        if stdout is None:
            return
        while True:
            line = await stdout.readline()
            if not line:
                return
            await self._handle_line(line.decode("utf-8", errors="replace").rstrip("\r\n"))

    async def _read_stderr(self) -> None:
        stderr = self._process.stderr
        if stderr is None:
            return
        while True:
            line = await stderr.readline()
            if not line:
                return
            self._stderr_chunks.append(line.decode("utf-8", errors="replace"))

    async def _wait_for_process(self) -> None:
        code = await self._process.wait()
        if self._closed and not self._pending:
            return
        self._closed = True
        stderr = "".join(self._stderr_chunks)
        status = code if code is not None else "unknown"
        message = stderr.strip() or f"kodelet acp exited with status {status}"
        self._reject_pending(AgentRunError(message, code=code, signal=None, stderr=stderr))

    async def _handle_line(self, line: str) -> None:
        trimmed = line.strip()
        if not trimmed:
            return
        try:
            message = json.loads(trimmed)
        except json.JSONDecodeError:
            self._notify_handlers("$/stdout", {"line": line})
            return
        if not isinstance(message, Mapping):
            return

        method = message.get("method")
        message_id = message.get("id")
        if isinstance(method, str) and message_id is not None:
            await self._respond_to_server_request(message)
            return
        if isinstance(method, str):
            self._notify_handlers(method, message.get("params"))
            return
        if not isinstance(message_id, int):
            return
        pending = self._pending.pop(message_id, None)
        if pending is None:
            return
        error = message.get("error")
        if isinstance(error, Mapping):
            pending.set_exception(RPCError(error))
        else:
            pending.set_result(message.get("result"))

    async def _respond_to_server_request(self, message: Mapping[str, Any]) -> None:
        await self._write(
            {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {
                    "code": -32601,
                    "message": f"Unsupported client RPC method: {message.get('method')}",
                },
            }
        )

    def _notify_handlers(self, method: str, params: Any) -> None:
        for handler in list(self._notification_handlers):
            handler(method, params)

    def _reject_pending(self, error: Exception) -> None:
        for pending in self._pending.values():
            pending.set_exception(error)
        self._pending.clear()


class BridgeEndpoint:
    def __init__(self, transport: BridgeTransport, *, path: str | None = None) -> None:
        self.transport = transport
        self.path = path
        self.host: str | None = None
        self.port: int | None = None


class InMemoryExtensionBridge:
    def __init__(self, root_dir: str, servers: Sequence[ExtensionSocketServer]) -> None:
        self._root_dir = root_dir
        self._servers = list(servers)

    @classmethod
    async def create(
        cls,
        entrypoints: Sequence[Entrypoint | Extension],
        options: Mapping[str, Any] | None = None,
    ) -> InMemoryExtensionBridge:
        root_dir = tempfile.mkdtemp(prefix="kodelet-sdk-extensions-")
        bridge_id = uuid.uuid4().hex[:16]
        servers: list[ExtensionSocketServer] = []
        ui = cast(AgentUIHandlers | None, (options or {}).get("ui"))
        transport = _normalize_bridge_transport((options or {}).get("transport"))
        try:
            for index, entrypoint in enumerate(entrypoints, start=1):
                extension_id = f"sdk-{bridge_id}-{index}"
                endpoint = _extension_bridge_endpoint(root_dir, extension_id, transport)
                host = await create_extension_host(entrypoint)
                server = ExtensionSocketServer(host, endpoint, ui)
                await server.listen()
                servers.append(server)

                executable_path = Path(root_dir) / f"kodelet-extension-{extension_id}"
                await asyncio.to_thread(
                    _write_executable,
                    executable_path,
                    _extension_bridge_executable(server.endpoint),
                )
        except Exception:
            await asyncio.gather(*(server.close() for server in servers), return_exceptions=True)
            shutil.rmtree(root_dir, ignore_errors=True)
            raise
        return cls(root_dir, servers)

    def config(self) -> dict[str, Any]:
        return {"enabled": True, "local_dir": self._root_dir, "allow": [self._root_dir]}

    async def close(self) -> None:
        await asyncio.gather(*(server.close() for server in self._servers), return_exceptions=True)
        shutil.rmtree(self._root_dir, ignore_errors=True)


class TempConfig:
    def __init__(self, root_dir: str, path: str) -> None:
        self._root_dir = root_dir
        self.path = path

    @classmethod
    async def create(cls, config: Mapping[str, Any]) -> TempConfig:
        root_dir = tempfile.mkdtemp(prefix="kodelet-sdk-config-")
        config_path = str(Path(root_dir) / "kodelet-config.json")
        await asyncio.to_thread(
            Path(config_path).write_text,
            f"{json.dumps(config, indent=2)}\n",
            encoding="utf-8",
        )
        return cls(root_dir, config_path)

    async def close(self) -> None:
        await asyncio.to_thread(shutil.rmtree, self._root_dir, True)


class ExtensionSocketServer(HostRPCClient):
    def __init__(
        self,
        host: Extension,
        endpoint: BridgeEndpoint,
        ui: AgentUIHandlers | None = None,
    ) -> None:
        self._host = host
        self.endpoint = endpoint
        self._ui = ui or {}
        self._server: asyncio.AbstractServer | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    async def listen(self) -> None:
        if self.endpoint.transport == "unix":
            if self.endpoint.path is None:
                raise RuntimeError("Unix extension bridge endpoint is missing a socket path")
            await asyncio.to_thread(_unlink_missing, self.endpoint.path)
            self._server = await asyncio.start_unix_server(
                self._handle_client,
                path=self.endpoint.path,
            )
            return

        self._server = await asyncio.start_server(self._handle_client, host="127.0.0.1", port=0)
        sock = self._server.sockets[0] if self._server.sockets else None
        if sock is None:
            raise RuntimeError("TCP extension bridge server did not expose a listening socket")
        host, port = sock.getsockname()[:2]
        self.endpoint.host = str(host)
        self.endpoint.port = int(port)

    async def close(self) -> None:
        for pending in self._pending.values():
            pending.set_exception(RuntimeError("Extension bridge closed"))
        self._pending.clear()
        for task in list(self._tasks):
            task.cancel()
        if self._writer is not None:
            self._writer.close()
            await self._writer.wait_closed()
            self._writer = None
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self.endpoint.transport == "unix" and self.endpoint.path is not None:
            await asyncio.to_thread(_unlink_missing, self.endpoint.path)

    async def request(self, method: str, params: Any | None = None) -> Any:
        local = await self._try_handle_local_ui_request(method, params)
        if local.get("handled"):
            return local.get("result")

        if self._writer is None:
            raise RuntimeError("Extension bridge is not connected")
        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        await _write_frame(
            self._writer,
            json.dumps(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
                separators=(",", ":"),
            ).encode("utf-8"),
        )
        return await future

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._writer = writer
        try:
            while True:
                payload = await _read_frame(reader)
                if payload is None:
                    return
                task = asyncio.create_task(self._handle_payload(payload, writer))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
        finally:
            if self._writer is writer:
                self._writer = None
            writer.close()
            await writer.wait_closed()

    async def _handle_payload(self, payload: bytes, writer: asyncio.StreamWriter) -> None:
        try:
            message = json.loads(payload.decode("utf-8"))
        except Exception as exc:
            await _write_json_frame(
                writer,
                {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}},
            )
            return
        if not isinstance(message, Mapping):
            return

        if not message.get("method") and message.get("id") is not None:
            self._handle_response(message)
            return

        method = message.get("method")
        message_id = message.get("id")
        if not isinstance(method, str) or message_id is None:
            return
        try:
            result = await run_with_host_rpc_client(
                self,
                lambda: self._dispatch(method, message.get("params")),
            )
            await _write_json_frame(writer, {"jsonrpc": "2.0", "id": message_id, "result": result})
        except Exception as exc:
            await _write_json_frame(
                writer,
                {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "error": {"code": -32000, "message": str(exc)},
                },
            )

    def _handle_response(self, response: Mapping[str, Any]) -> None:
        response_id = response.get("id")
        if not isinstance(response_id, int):
            return
        pending = self._pending.pop(response_id, None)
        if pending is None:
            return
        error = response.get("error")
        if isinstance(error, Mapping):
            pending.set_exception(RuntimeError(str(error.get("message") or "JSON-RPC error")))
        else:
            pending.set_result(response.get("result"))

    async def _dispatch(self, method: str, params: Any) -> Any:
        request_params = params if isinstance(params, Mapping) else {}
        if method == "extension.initialize":
            return self._host.initialize(request_params)
        if method == "extension.tool.execute":
            return await self._host.execute_tool(request_params)
        if method == "extension.command.execute":
            return await self._host.execute_command(request_params)
        if method == "extension.event.handle":
            return await self._host.handle_event(request_params)
        if method == "$/cancelRequest":
            return None
        raise RuntimeError(f"Unknown JSON-RPC method: {method}")

    async def _try_handle_local_ui_request(self, method: str, params: Any) -> dict[str, Any]:
        if method == "kodelet.ui.input":
            handler = self._ui.get("input")
            if handler is None:
                return {"handled": True, "result": _unavailable_ui("ui input is not available")}
            value = await maybe_await(handler(cast(UIInputRequest, params)))
            return {
                "handled": True,
                "result": _dismissed_ui()
                if value is None
                else {"status": "submitted", "value": value},
            }
        if method == "kodelet.ui.confirm":
            handler = self._ui.get("confirm")
            if handler is None:
                return {"handled": True, "result": _unavailable_ui("ui confirm is not available")}
            confirmed = await maybe_await(handler(cast(UIConfirmRequest, params)))
            return {
                "handled": True,
                "result": {"status": "submitted", "confirmed": bool(confirmed)},
            }
        if method == "kodelet.ui.select":
            handler = self._ui.get("select")
            if handler is None:
                return {"handled": True, "result": _unavailable_ui("ui select is not available")}
            value = await maybe_await(handler(cast(UISelectRequest, params)))
            return {
                "handled": True,
                "result": _dismissed_ui()
                if value is None
                else {"status": "submitted", "value": value},
            }
        if method == "kodelet.ui.notify":
            handler = self._ui.get("notify")
            if handler is None:
                return {"handled": True, "result": _unavailable_ui("ui notify is not available")}
            await maybe_await(handler(cast(UINotifyRequest, params)))
            return {"handled": True, "result": {"status": "submitted"}}
        return {"handled": False}


class LaunchConfig:
    def __init__(
        self,
        *,
        args: Sequence[str],
        env: Mapping[str, str],
        temp_config: TempConfig | None = None,
        config_file_mode: str | None = None,
    ) -> None:
        self.args = list(args)
        self.env = dict(env)
        self.temp_config = temp_config
        self.config_file_mode = config_file_mode


def _normalize_profile(profile: Any) -> Profile | None:
    if profile is None:
        return None
    if isinstance(profile, Profile):
        return profile
    if isinstance(profile, str) or isinstance(profile, Mapping):
        return Profile(profile)
    raise TypeError("profile must be a profile name, Profile, or mapping")


def _normalize_bridge_transport(value: Any) -> BridgeTransport:
    if value is None:
        return "unix"
    if value in {"unix", "tcp"}:
        return cast(BridgeTransport, value)
    raise ValueError("extension_transport must be 'unix' or 'tcp'")


def _normalize_run_options(
    options: RunOptions | str | None,
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(options, str):
        run_options = {"message": options}
    else:
        run_options = dict(options or {})
    run_options.update(kwargs)
    message = run_options.get("message")
    if not isinstance(message, str) or not message:
        raise ValueError("run_and_wait requires a non-empty message")
    return run_options


def _option_int(options: Mapping[str, Any], key: str) -> int | None:
    value = options.get(key)
    return value if isinstance(value, int) else None


def _acp_server_args(options: Mapping[str, Any]) -> list[str]:
    max_turns = _option_int(options, "max_turns")
    if max_turns is not None and max_turns > 0:
        return ["--max-turns", str(max_turns)]
    return []


async def _build_launch_config(
    profile: Profile | None,
    bridge: InMemoryExtensionBridge | None,
) -> LaunchConfig:
    resolved = profile.to_launch_config() if profile is not None else None
    profile_config = resolved.get("config") if resolved else None
    config: dict[str, Any] = {}
    if isinstance(profile_config, Mapping):
        config.update(profile_config)
        if profile_config.get("profile") is None:
            config["profile"] = "default"
    if bridge is not None:
        config["extensions"] = bridge.config()
    config = _prune_none(config)
    if not config:
        return LaunchConfig(args=cast(Sequence[str], (resolved or {}).get("args") or []), env={})

    temp_config = await TempConfig.create(config)
    config_file_mode = "isolated" if isinstance(profile_config, Mapping) else "merge"
    return LaunchConfig(
        args=cast(Sequence[str], (resolved or {}).get("args") or []),
        env={"KODELET_CONFIG_FILE": temp_config.path, "KODELET_CONFIG_FILE_MODE": config_file_mode},
        temp_config=temp_config,
        config_file_mode=config_file_mode,
    )


def _build_prompt_blocks(options: Mapping[str, Any]) -> list[dict[str, Any]]:
    prompt = [{"type": "text", "text": str(options["message"])}]
    images = options.get("images")
    if isinstance(images, Sequence) and not isinstance(images, str | bytes):
        for image in images:
            if isinstance(image, str):
                prompt.append(_image_to_content_block(image))
    return prompt


def _image_to_content_block(image: str) -> dict[str, Any]:
    prefix = "data:"
    marker = ";base64,"
    if image.startswith(prefix) and marker in image:
        mime_type, data = image[len(prefix) :].split(marker, 1)
        return {"type": "image", "mimeType": mime_type, "data": data}
    return {"type": "image", "uri": image}


def _text_from_acp_content(content: Any) -> str:
    if not isinstance(content, Mapping):
        return ""
    text = content.get("text")
    if isinstance(text, str):
        return text
    resource = content.get("resource")
    if isinstance(resource, Mapping) and isinstance(resource.get("text"), str):
        return str(resource["text"])
    return ""


def _tool_name_from_update(update: Mapping[str, Any]) -> str:
    tool_name = update.get("toolName")
    return tool_name.strip() if isinstance(tool_name, str) and tool_name.strip() else ""


def _tool_content_to_text(content: Any) -> str:
    if not isinstance(content, Sequence) or isinstance(content, str | bytes):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, Mapping):
            continue
        if item.get("type") == "content" and isinstance(item.get("content"), Mapping):
            text = _text_from_acp_content(item.get("content"))
            if text:
                parts.append(text)
            continue
        path = item.get("path")
        if isinstance(path, str):
            parts.append(path)
        new_text = item.get("newText")
        if isinstance(new_text, str):
            parts.append(new_text)
    return "\n".join(parts)


def _string_field(record: Mapping[str, Any], key: str) -> str | None:
    value = record.get(key)
    return value if isinstance(value, str) else None


def _prune_none(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if item is None:
            continue
        if isinstance(item, Mapping):
            result[str(key)] = _prune_none(item)
        else:
            result[str(key)] = item
    return result


def _clean_env(env: Mapping[str, str | None]) -> dict[str, str]:
    return {key: str(value) for key, value in env.items() if value is not None}


def _resolve_path(path: str) -> str:
    return str(Path(path).resolve(strict=False))


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _unlink_missing(path: str) -> None:
    Path(path).unlink(missing_ok=True)


def _to_attr_dict(value: Any) -> Any:
    if isinstance(value, Mapping):
        return AttrDict({str(key): _to_attr_dict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_attr_dict(item) for item in value]
    return value


async def _read_frame(reader: asyncio.StreamReader) -> bytes | None:
    header_lines: list[bytes] = []
    while True:
        line = await reader.readline()
        if line == b"":
            return None
        if line in (b"\r\n", b"\n"):
            break
        header_lines.append(line)
    header = b"".join(header_lines).decode("ascii", errors="replace")
    content_length = _parse_content_length(header)
    try:
        return await reader.readexactly(content_length)
    except asyncio.IncompleteReadError:
        return None


def _parse_content_length(header: str) -> int:
    for line in header.splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() == "content-length":
            content_length = int(value.strip())
            if content_length >= 0:
                return content_length
    raise ValueError("Missing Content-Length header")


async def _write_json_frame(writer: asyncio.StreamWriter, message: Mapping[str, Any]) -> None:
    await _write_frame(writer, json.dumps(message, separators=(",", ":")).encode("utf-8"))


async def _write_frame(writer: asyncio.StreamWriter, payload: bytes) -> None:
    writer.write(b"Content-Length: " + str(len(payload)).encode("ascii") + b"\r\n\r\n" + payload)
    await writer.drain()


def _extension_bridge_endpoint(
    root_dir: str,
    extension_id: str,
    transport: BridgeTransport,
) -> BridgeEndpoint:
    if transport == "unix":
        return BridgeEndpoint(transport, path=str(Path(root_dir) / f"{extension_id}.sock"))
    return BridgeEndpoint(transport)


def _extension_bridge_executable(endpoint: BridgeEndpoint) -> str:
    if endpoint.transport == "unix":
        if endpoint.path is None:
            raise RuntimeError("Unix extension bridge endpoint is missing a socket path")
        endpoint_config: dict[str, Any] = {"transport": "unix", "path": endpoint.path}
    else:
        if endpoint.host is None or endpoint.port is None:
            raise RuntimeError("TCP extension bridge endpoint is missing host/port")
        endpoint_config = {"transport": "tcp", "host": endpoint.host, "port": endpoint.port}

    return f'''#!/usr/bin/env python3
from __future__ import annotations

import json
import socket
import sys
import threading

ENDPOINT = {endpoint_config!r}


def read_frame(stream):
    header = bytearray()
    while True:
        ch = stream.read(1)
        if ch == b"":
            return None
        header += ch
        if header.endswith(b"\\r\\n\\r\\n") or header.endswith(b"\\n\\n"):
            break
    header_text = header.decode("ascii", errors="replace")
    content_length = parse_content_length(header_text)
    payload = stream.read(content_length)
    if len(payload) != content_length:
        return None
    return payload


def parse_content_length(header):
    for line in header.splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() == "content-length":
            parsed = int(value.strip())
            if parsed >= 0:
                return parsed
    raise RuntimeError("Missing Content-Length header")


def framed(payload):
    return b"Content-Length: " + str(len(payload)).encode("ascii") + b"\\r\\n\\r\\n" + payload


stdout_lock = threading.Lock()


def stdin_to_socket(sock):
    while True:
        payload = read_frame(sys.stdin.buffer)
        if payload is None:
            try:
                sock.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            return
        sock.sendall(framed(payload))


def socket_to_stdout(sock_file):
    while True:
        payload = read_frame(sock_file)
        if payload is None:
            return
        with stdout_lock:
            sys.stdout.buffer.write(framed(payload))
            sys.stdout.buffer.flush()


def main():
    if ENDPOINT["transport"] == "unix":
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        address = ENDPOINT["path"]
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        address = (ENDPOINT["host"], ENDPOINT["port"])
    try:
        sock.connect(address)
        sock_file = sock.makefile("rb")
        thread = threading.Thread(target=stdin_to_socket, args=(sock,), daemon=True)
        thread.start()
        socket_to_stdout(sock_file)
    except Exception as exc:
        error_payload = {{
            "level": "error",
            "message": "kodelet SDK extension bridge failed",
            "error": str(exc),
        }}
        sys.stderr.write(json.dumps(error_payload) + "\\n")
        sys.stderr.flush()
        return 1
    finally:
        try:
            sock.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _unavailable_ui(reason: str) -> dict[str, str]:
    return {"status": "unavailable", "reason": reason}


def _dismissed_ui() -> dict[str, str]:
    return {"status": "dismissed"}


__all__ = [
    "AgentResponse",
    "AgentRunError",
    "AgentStreamEvent",
    "AgentUIHandlers",
    "AssistantMessageData",
    "AssistantMessageDeltaData",
    "AssistantThinkingDeltaData",
    "BridgeTransport",
    "Client",
    "ClientOptions",
    "CreateSessionOptions",
    "Profile",
    "ProfileInput",
    "RunOptions",
    "Session",
    "SpawnFunction",
    "SpawnOptions",
    "SpawnedProcess",
    "ToolCallData",
    "ToolResultData",
]

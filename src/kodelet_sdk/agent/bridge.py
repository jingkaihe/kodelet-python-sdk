from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from .._utils import maybe_await
from ..api import Entrypoint, Extension, create_extension_host
from ..context import (
    HostRPCClient,
    UIConfirmRequest,
    UIInputRequest,
    UINotifyRequest,
    UISelectRequest,
    _release_persistent_ui_state,
    run_with_host_rpc_client,
)
from .transport import (
    BridgeEndpoint,
    _extension_bridge_endpoint,
    _extension_bridge_executable,
    _normalize_bridge_transport,
    _read_frame,
    _unlink_missing,
    _write_executable,
    _write_json_frame,
)
from .types import AgentUIHandlers


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
        root_dir = tempfile.mkdtemp(
            prefix="kodelet-sdk-extensions-",
            dir=None if os.name == "nt" else "/tmp",
        )
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


class _BridgeRequestState:
    def __init__(self, request_id: int | str) -> None:
        self.request_id = request_id
        self.task: asyncio.Task[None] | None = None
        self.active = True
        self.terminal_valid = True
        self.cancelled = False
        self.terminal = asyncio.Event()

    def cancel(self) -> None:
        self.active = False
        self.terminal_valid = False
        self.cancelled = True
        self.terminal.set()
        if self.task is not None and not self.task.done():
            self.task.cancel()

    def finish(self) -> None:
        self.active = False
        self.terminal.set()

    def finish_terminal(self) -> None:
        self.terminal_valid = False


class _BridgeConnection:
    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self.writer = writer
        self.write_lock = asyncio.Lock()
        self.next_id = 0
        self.pending: dict[int, tuple[_BridgeRequestState | None, asyncio.Future[Any]]] = {}
        self.requests: dict[int | str, _BridgeRequestState] = {}
        self.notification_handlers: set[Callable[[str, Any], None]] = set()
        self.persistent_client: _PersistentConnectionHostRPCClient | None = None
        self.closed = False

    async def send(
        self,
        message: Mapping[str, Any],
        state: _BridgeRequestState | None = None,
        *,
        terminal: bool = False,
    ) -> None:
        if self.closed:
            raise RuntimeError("Extension bridge connection is closed")
        async with self.write_lock:
            if self.closed:
                raise RuntimeError("Extension bridge connection is closed")
            if state is not None:
                valid = state.terminal_valid if terminal else state.active
                if not valid or self.requests.get(state.request_id) is not state:
                    raise asyncio.CancelledError
            await _write_json_frame(self.writer, message)

    def cancel_request(self, request_id: int | str) -> None:
        state = self.requests.get(request_id)
        if state is not None:
            state.cancel()
            for reverse_id, (pending_state, future) in list(self.pending.items()):
                if pending_state is not state:
                    continue
                self.pending.pop(reverse_id, None)
                if not future.done():
                    future.set_exception(asyncio.CancelledError())

    def finish_request(self, state: _BridgeRequestState) -> None:
        state.finish()
        for reverse_id, (pending_state, future) in list(self.pending.items()):
            if pending_state is not state:
                continue
            self.pending.pop(reverse_id, None)
            if not future.done():
                future.set_exception(RuntimeError("Extension request completed"))

    def finish_terminal(self, state: _BridgeRequestState) -> None:
        state.finish_terminal()
        if self.requests.get(state.request_id) is state:
            self.requests.pop(state.request_id, None)

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.persistent_client is not None:
            _release_persistent_ui_state(self.persistent_client)
        for _, future in self.pending.values():
            if not future.done():
                future.set_exception(RuntimeError("Extension bridge connection closed"))
        self.pending.clear()
        states = list(self.requests.values())
        for state in states:
            state.cancel()
        tasks = [state.task for state in states if state.task is not None]
        if tasks:
            await asyncio.wait(tasks, timeout=1)
        self.requests.clear()
        self.notification_handlers.clear()
        self.persistent_client = None
        self.writer.close()
        try:
            await self.writer.wait_closed()
        except (ConnectionError, RuntimeError):
            pass

    def handle_notification(self, method: str, params: Any) -> None:
        if self.closed:
            return
        for handler in list(self.notification_handlers):
            handler(method, params)


class _ConnectionHostRPCClient(HostRPCClient):
    def __init__(
        self,
        server: ExtensionSocketServer,
        connection: _BridgeConnection,
        parent_id: int | str,
        state: _BridgeRequestState,
    ) -> None:
        self._server = server
        self._connection = connection
        self._parent_id = parent_id
        self._state = state
        if connection.persistent_client is None:
            connection.persistent_client = _PersistentConnectionHostRPCClient(server, connection)
        self.persistent: HostRPCClient = connection.persistent_client

    async def request(self, method: str, params: Any | None = None) -> Any:
        return await self._server._request(
            self._connection,
            self._parent_id,
            self._state,
            method,
            params,
        )

    async def request_persistent(self, method: str, params: Any | None = None) -> Any:
        if self._state.active:
            return await self._server._request(
                self._connection,
                self._parent_id,
                self._state,
                method,
                params,
            )
        return await self._server._request_persistent(self._connection, method, params)


class _PersistentConnectionHostRPCClient(HostRPCClient):
    def __init__(
        self,
        server: ExtensionSocketServer,
        connection: _BridgeConnection,
    ) -> None:
        self._server = server
        self._connection = connection

    async def request(self, method: str, params: Any | None = None) -> Any:
        return await self._server._request_persistent(self._connection, method, params)

    async def notify(self, method: str, params: Any | None = None) -> None:
        await self._connection.send({"jsonrpc": "2.0", "method": method, "params": params})

    def on_notification(
        self,
        handler: Callable[[str, Any], None],
    ) -> Callable[[], None]:
        self._connection.notification_handlers.add(handler)

        def unsubscribe() -> None:
            self._connection.notification_handlers.discard(handler)

        return unsubscribe


class ExtensionSocketServer:
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
        self._connections: set[_BridgeConnection] = set()

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
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        await asyncio.gather(
            *(connection.close() for connection in list(self._connections)),
            return_exceptions=True,
        )
        self._connections.clear()
        if self.endpoint.transport == "unix" and self.endpoint.path is not None:
            await asyncio.to_thread(_unlink_missing, self.endpoint.path)

    async def _request(
        self,
        connection: _BridgeConnection,
        parent_id: int | str,
        state: _BridgeRequestState,
        method: str,
        params: Any | None = None,
    ) -> Any:
        self._ensure_active(connection, state)
        local = await self._try_handle_active_local_request(state, method, params)
        if local.get("handled"):
            self._ensure_active(connection, state)
            return local.get("result")

        self._ensure_active(connection, state)
        connection.next_id += 1
        request_id = connection.next_id
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        connection.pending[request_id] = (state, future)
        message: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "parentId": parent_id,
            "method": method,
            "params": params,
        }
        try:
            await connection.send(message, state)
            return await future
        finally:
            connection.pending.pop(request_id, None)

    async def _request_persistent(
        self,
        connection: _BridgeConnection,
        method: str,
        params: Any | None = None,
    ) -> Any:
        if connection.closed:
            raise RuntimeError("Extension bridge connection is closed")
        connection.next_id += 1
        request_id = connection.next_id
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        connection.pending[request_id] = (None, future)
        try:
            await connection.send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params,
                }
            )
            return await future
        finally:
            connection.pending.pop(request_id, None)

    @staticmethod
    def _ensure_active(
        connection: _BridgeConnection,
        state: _BridgeRequestState,
    ) -> None:
        if state.cancelled:
            raise asyncio.CancelledError
        if (
            connection.closed
            or not state.active
            or connection.requests.get(state.request_id) is not state
        ):
            raise RuntimeError("Extension request is no longer active")

    async def _try_handle_active_local_request(
        self,
        state: _BridgeRequestState,
        method: str,
        params: Any,
    ) -> dict[str, Any]:
        local_task = asyncio.create_task(self._try_handle_local_ui_request(method, params))
        terminal_task = asyncio.create_task(state.terminal.wait())
        try:
            done, _ = await asyncio.wait(
                {local_task, terminal_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if terminal_task in done:
                if state.cancelled:
                    raise asyncio.CancelledError
                raise RuntimeError("Extension request is no longer active")
            return await local_task
        finally:
            for task in (local_task, terminal_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(local_task, terminal_task, return_exceptions=True)

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        connection = _BridgeConnection(writer)
        self._connections.add(connection)
        try:
            while True:
                payload = await _read_frame(reader)
                if payload is None:
                    return
                await self._route_payload(payload, connection)
        finally:
            await connection.close()
            self._connections.discard(connection)

    async def _route_payload(self, payload: bytes, connection: _BridgeConnection) -> None:
        try:
            message = json.loads(payload.decode("utf-8"))
        except Exception as exc:
            await connection.send(
                {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}},
            )
            return
        if not isinstance(message, Mapping):
            return

        if not message.get("method") and message.get("id") is not None:
            self._handle_response(connection, message)
            return

        method = message.get("method")
        message_id = message.get("id")
        if method == "$/cancelRequest" and message_id is None:
            params = message.get("params")
            if isinstance(params, Mapping) and isinstance(params.get("id"), int | str):
                connection.cancel_request(params["id"])
            return
        if isinstance(method, str) and message_id is None:
            connection.handle_notification(method, message.get("params"))
            return
        if not isinstance(method, str) or not isinstance(message_id, int | str):
            return

        previous = connection.requests.get(message_id)
        if previous is not None:
            previous.cancel()
        state = _BridgeRequestState(message_id)
        task = asyncio.create_task(
            self._dispatch_request(
                connection,
                message_id,
                state,
                method,
                message.get("params"),
            )
        )
        state.task = task
        connection.requests[message_id] = state

        def remove_request(completed: asyncio.Task[None]) -> None:
            current = connection.requests.get(message_id)
            if current is state and current.task is completed:
                connection.requests.pop(message_id, None)

        task.add_done_callback(remove_request)

    async def _dispatch_request(
        self,
        connection: _BridgeConnection,
        message_id: int | str,
        state: _BridgeRequestState,
        method: str,
        params: Any,
    ) -> None:
        client = _ConnectionHostRPCClient(self, connection, message_id, state)
        try:
            result = await run_with_host_rpc_client(
                client,
                lambda: self._dispatch(method, params),
            )
            should_respond = not connection.closed and state.active
            connection.finish_request(state)
            if should_respond:
                await connection.send(
                    {"jsonrpc": "2.0", "id": message_id, "result": result},
                    state,
                    terminal=True,
                )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            should_respond = not connection.closed and state.active
            connection.finish_request(state)
            if should_respond:
                await connection.send(
                    {
                        "jsonrpc": "2.0",
                        "id": message_id,
                        "error": {"code": -32000, "message": str(exc)},
                    },
                    state,
                    terminal=True,
                )
        finally:
            connection.finish_request(state)
            connection.finish_terminal(state)

    def _handle_response(
        self,
        connection: _BridgeConnection,
        response: Mapping[str, Any],
    ) -> None:
        response_id = response.get("id")
        if not isinstance(response_id, int):
            return
        pending_entry = connection.pending.pop(response_id, None)
        if pending_entry is None:
            return
        _, pending = pending_entry
        if pending.done():
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


def _unavailable_ui(reason: str) -> dict[str, str]:
    return {"status": "unavailable", "reason": reason}


def _dismissed_ui() -> dict[str, str]:
    return {"status": "dismissed"}


__all__ = [
    "BridgeEndpoint",
    "ExtensionSocketServer",
    "InMemoryExtensionBridge",
    "TempConfig",
]

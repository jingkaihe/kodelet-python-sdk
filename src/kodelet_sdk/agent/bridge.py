from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import uuid
from collections.abc import Mapping, Sequence
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
    _write_frame,
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

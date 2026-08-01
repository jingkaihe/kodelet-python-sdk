from __future__ import annotations

import asyncio
import json
import sys
import threading
from collections.abc import Callable, Mapping
from typing import Any, Protocol, cast

from .api import Entrypoint, Extension, create_extension_host
from .context import HostRPCClient, _release_persistent_ui_state, run_with_host_rpc_client


class _StdioRequestState:
    def __init__(self, request_id: int | str) -> None:
        self.request_id = request_id
        self.task: asyncio.Task[None] | None = None
        self.active = True
        self.terminal_valid = True
        self._write_guard = threading.Lock()

    async def cancel(self) -> None:
        await asyncio.to_thread(self._mark_cancelled)
        if self.task is not None:
            self.task.cancel()

    def _mark_cancelled(self) -> None:
        with self._write_guard:
            self.active = False
            self.terminal_valid = False

    async def finish(self) -> None:
        await asyncio.to_thread(self._mark_finished)

    def _mark_finished(self) -> None:
        with self._write_guard:
            self.active = False

    async def finish_terminal(self) -> None:
        await asyncio.to_thread(self._mark_terminal_finished)

    def _mark_terminal_finished(self) -> None:
        with self._write_guard:
            self.terminal_valid = False


class BinaryReader(Protocol):
    """Minimal binary reader protocol used by the stdio runtime."""

    def read(self, size: int = -1, /) -> bytes: ...

    def readline(self, size: int = -1, /) -> bytes: ...


class BinaryWriter(Protocol):
    """Minimal binary writer protocol used by the stdio runtime."""

    def write(self, data: bytes, /) -> object: ...

    def flush(self) -> object: ...


class StdioHostRPCClient(HostRPCClient):
    """Reverse-RPC client that sends extension-initiated requests to stdout."""

    def __init__(self, writer: BinaryWriter) -> None:
        self._writer = writer
        self._write_lock = asyncio.Lock()
        self._next_id = 0
        self._pending: dict[int, tuple[_StdioRequestState | None, asyncio.Future[Any]]] = {}
        self._notification_handlers: set[Callable[[str, Any], None]] = set()
        self._closed = False

    async def send(
        self,
        message: Mapping[str, Any],
        state: _StdioRequestState | None = None,
        *,
        terminal: bool = False,
    ) -> None:
        """Write one framed message without interleaving concurrent writes."""

        if self._closed:
            raise RuntimeError("Extension host connection is closed")
        async with self._write_lock:
            if self._closed:
                raise RuntimeError("Extension host connection is closed")
            await write_message(self._writer, message, state, terminal=terminal)

    async def request(self, method: str, params: Any | None = None) -> Any:
        """Send a connection-scoped JSON-RPC request without a parent ID.

        Args:
            method: Host method name, for example ``kodelet.ui.input``.
            params: Optional JSON-serializable parameters.

        Returns:
            The host response ``result`` value.
        """

        return await self._request(None, method, params)

    async def request_for(
        self,
        state: _StdioRequestState,
        method: str,
        params: Any | None = None,
    ) -> Any:
        """Send a reverse-RPC request scoped to one active host request."""

        return await self._request(state, method, params)

    async def _request(
        self,
        state: _StdioRequestState | None,
        method: str,
        params: Any | None,
    ) -> Any:
        if self._closed:
            raise RuntimeError("Extension host connection is closed")
        if state is not None and not state.active:
            raise asyncio.CancelledError
        self._next_id += 1
        request_id = self._next_id
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[request_id] = (state, future)
        message: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        if state is not None:
            message["parentId"] = state.request_id
        try:
            await self.send(message, state)
            return await future
        finally:
            self._pending.pop(request_id, None)

    async def notify(self, method: str, params: Any | None = None) -> None:
        """Send a connection-scoped JSON-RPC notification."""

        await self.send({"jsonrpc": "2.0", "method": method, "params": params})

    def on_notification(
        self,
        handler: Callable[[str, Any], None],
    ) -> Callable[[], None]:
        """Subscribe to parentless notifications received from the host."""

        self._notification_handlers.add(handler)

        def unsubscribe() -> None:
            self._notification_handlers.discard(handler)

        return unsubscribe

    def handle_notification(self, method: str, params: Any) -> None:
        """Route one parentless host notification to subscribers."""

        if self._closed:
            return
        for handler in list(self._notification_handlers):
            handler(method, params)

    async def close(self, error: BaseException | None = None) -> None:
        """Reject all pending reverse RPC and stop notification delivery."""

        if self._closed:
            return
        self._closed = True
        _release_persistent_ui_state(self)
        terminal_error = error or RuntimeError("Extension host connection closed")
        for _, future in self._pending.values():
            if not future.done():
                future.set_exception(terminal_error)
        self._pending.clear()
        self._notification_handlers.clear()

    async def finish_request(
        self,
        state: _StdioRequestState,
        error: BaseException | None = None,
    ) -> None:
        """Invalidate one request generation and reject its outstanding reverse RPC."""

        await state.finish()
        terminal_error = error or RuntimeError("Extension request completed")
        for reverse_id, (pending_state, future) in list(self._pending.items()):
            if pending_state is not state:
                continue
            self._pending.pop(reverse_id, None)
            if not future.done():
                future.set_exception(terminal_error)

    def handle_response(self, response: Mapping[str, Any]) -> bool:
        """Resolve a pending reverse-RPC request from a host response.

        Args:
            response: Decoded JSON-RPC response.

        Returns:
            ``True`` when the response matched a pending request.
        """

        response_id = response.get("id")
        if not isinstance(response_id, int):
            return False
        pending = self._pending.pop(response_id, None)
        if pending is None:
            return False
        _, future = pending
        if future.done():
            return True
        if error := response.get("error"):
            if isinstance(error, Mapping):
                future.set_exception(RuntimeError(str(error.get("message") or "JSON-RPC error")))
            else:
                future.set_exception(RuntimeError("JSON-RPC error"))
        else:
            future.set_result(response.get("result"))
        return True


class _RequestScopedHostRPCClient:
    def __init__(self, client: StdioHostRPCClient, state: _StdioRequestState) -> None:
        self.persistent: HostRPCClient = client
        self._client = client
        self._state = state

    async def request(self, method: str, params: Any | None = None) -> Any:
        return await self._client.request_for(self._state, method, params)

    async def request_persistent(self, method: str, params: Any | None = None) -> Any:
        if self._state.active:
            return await self._client.request_for(self._state, method, params)
        return await self._client.request(method, params)


async def run_extension(entrypoint: Extension | Entrypoint) -> None:
    """Run an extension entrypoint over stdio.

    Args:
        entrypoint: Existing :class:`kodelet_sdk.Extension` or callable that
            registers behavior on a new extension.
    """

    host = await create_extension_host(entrypoint)
    await run_stdio_server(host)


async def run_stdio_server(
    host: Extension,
    reader: BinaryReader | None = None,
    writer: BinaryWriter | None = None,
) -> None:
    """Serve the Kodelet JSON-RPC protocol on framed binary streams.

    Args:
        host: Extension host to dispatch requests to.
        reader: Binary input stream. Defaults to ``sys.stdin.buffer``.
        writer: Binary output stream. Defaults to ``sys.stdout.buffer``.
    """

    resolved_reader: BinaryReader = reader or cast(BinaryReader, sys.stdin.buffer)
    resolved_writer: BinaryWriter = writer or cast(BinaryWriter, sys.stdout.buffer)
    host_client = StdioHostRPCClient(resolved_writer)
    pending_tasks: set[asyncio.Task[None]] = set()
    request_states: dict[int | str, _StdioRequestState] = {}
    try:
        while True:
            payload = await asyncio.to_thread(read_frame, resolved_reader)
            if payload is None:
                break

            try:
                message = json.loads(payload.decode("utf-8"))
            except Exception as exc:
                await host_client.send(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": str(exc)},
                    },
                )
                continue
            if not isinstance(message, Mapping):
                continue
            if not message.get("method") and host_client.handle_response(message):
                continue

            method = message.get("method")
            request_id = message.get("id")
            if method == "$/cancelRequest" and request_id is None:
                params = message.get("params")
                if isinstance(params, Mapping) and isinstance(params.get("id"), int | str):
                    cancelled_id = params["id"]
                    state = request_states.get(cancelled_id)
                    if state is not None:
                        await state.cancel()
                        await host_client.finish_request(state, asyncio.CancelledError())
                continue
            if isinstance(method, str) and request_id is None:
                host_client.handle_notification(method, message.get("params"))
                continue
            if not isinstance(request_id, int | str):
                continue

            previous = request_states.get(request_id)
            if previous is not None:
                await previous.cancel()
                await host_client.finish_request(
                    previous,
                    RuntimeError("Extension request id was reused"),
                )
            state = _StdioRequestState(request_id)
            task = asyncio.create_task(_dispatch_request(host, host_client, message, state))
            state.task = task
            request_states[request_id] = state
            pending_tasks.add(task)
            task.add_done_callback(pending_tasks.discard)

            def remove_request(
                completed: asyncio.Task[None],
                *,
                active_id: int | str = request_id,
                active_state: _StdioRequestState = state,
            ) -> None:
                current = request_states.get(active_id)
                if current is active_state and current.task is completed:
                    request_states.pop(active_id, None)

            task.add_done_callback(remove_request)
    finally:
        for state in list(request_states.values()):
            await state.cancel()
            await host_client.finish_request(state, asyncio.CancelledError())
        await host_client.close()
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)


async def _dispatch_request(
    host: Extension,
    host_client: StdioHostRPCClient,
    message: Mapping[str, Any],
    state: _StdioRequestState,
) -> None:
    request_client = _RequestScopedHostRPCClient(host_client, state)
    try:
        result = await run_with_host_rpc_client(
            request_client,
            lambda: _dispatch(host, message),
        )
        should_respond = state.active
        await host_client.finish_request(state)
        if should_respond:
            await host_client.send(
                {"jsonrpc": "2.0", "id": state.request_id, "result": result},
                state,
                terminal=True,
            )
    except asyncio.CancelledError:
        return
    except Exception as exc:
        should_respond = state.active
        await host_client.finish_request(state)
        if should_respond:
            await host_client.send(
                {
                    "jsonrpc": "2.0",
                    "id": state.request_id,
                    "error": {"code": -32000, "message": str(exc)},
                },
                state,
                terminal=True,
            )
    finally:
        await host_client.finish_request(state)
        await state.finish_terminal()


async def _dispatch(host: Extension, request: Mapping[str, Any]) -> Any:
    method = request.get("method")
    params = request.get("params")
    if not isinstance(params, Mapping):
        params = {}
    if method == "extension.initialize":
        return host.initialize(params)
    if method == "extension.tool.execute":
        return await host.execute_tool(params)
    if method == "extension.command.execute":
        return await host.execute_command(params)
    if method == "extension.event.handle":
        return await host.handle_event(params)
    raise ValueError(f"Unknown JSON-RPC method: {method}")


def _try_read_frame(buffer: bytes) -> tuple[bytes, bytes] | None:
    header_end = buffer.find(b"\r\n\r\n")
    separator_length = 4
    if header_end == -1:
        header_end = buffer.find(b"\n\n")
        separator_length = 2
    if header_end == -1:
        return None
    header = buffer[:header_end].decode("ascii", errors="replace")
    content_length = _parse_content_length(header)
    payload_start = header_end + separator_length
    payload_end = payload_start + content_length
    if len(buffer) < payload_end:
        return None
    return buffer[payload_start:payload_end], buffer[payload_end:]


def _parse_content_length(header: str) -> int:
    for line in header.splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() == "content-length":
            content_length = int(value.strip())
            if content_length >= 0:
                return content_length
    raise ValueError("Missing Content-Length header")


def read_frame(reader: BinaryReader) -> bytes | None:
    """Read one LSP-style framed JSON-RPC payload from a blocking stream.

    Unlike ``reader.read(4096)``, this returns as soon as one complete frame has
    arrived and does not wait for EOF or for the pipe buffer to fill.
    """

    header_lines: list[bytes] = []
    while True:
        line = reader.readline()
        if line == b"":
            return None
        if line in (b"\r\n", b"\n"):
            break
        header_lines.append(line)

    header = b"".join(header_lines).decode("ascii", errors="replace")
    content_length = _parse_content_length(header)
    payload = reader.read(content_length)
    if len(payload) != content_length:
        return None
    return payload


async def write_message(
    writer: BinaryWriter,
    message: Mapping[str, Any],
    state: _StdioRequestState | None = None,
    *,
    terminal: bool = False,
) -> None:
    """Write one LSP-style ``Content-Length`` framed JSON-RPC message.

    Args:
        writer: Binary output stream.
        message: JSON-serializable message mapping.
    """

    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    frame = b"Content-Length: " + str(len(payload)).encode("ascii") + b"\r\n\r\n" + payload
    if state is None:
        await asyncio.to_thread(_write_and_flush, writer, frame)
        return
    await asyncio.to_thread(_write_and_flush_for_request, writer, frame, state, terminal)


def _write_and_flush(writer: BinaryWriter, frame: bytes) -> None:
    writer.write(frame)
    writer.flush()


def _write_and_flush_for_request(
    writer: BinaryWriter,
    frame: bytes,
    state: _StdioRequestState,
    terminal: bool,
) -> None:
    with state._write_guard:
        valid = state.terminal_valid if terminal else state.active
        if not valid:
            raise asyncio.CancelledError
        writer.write(frame)
        writer.flush()

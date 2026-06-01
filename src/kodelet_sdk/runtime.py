from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Mapping
from typing import Any, Protocol, cast

from .api import Entrypoint, Extension, create_extension_host
from .context import HostRPCClient, set_active_host_rpc_client


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
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}

    async def request(self, method: str, params: Any | None = None) -> Any:
        """Send a JSON-RPC request to the Kodelet host and await the response.

        Args:
            method: Host method name, for example ``kodelet.ui.input``.
            params: Optional JSON-serializable parameters.

        Returns:
            The host response ``result`` value.
        """

        self._next_id += 1
        request_id = self._next_id
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[request_id] = future
        await write_message(
            self._writer,
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
        )
        return await future

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
        future = self._pending.pop(response_id, None)
        if future is None:
            return False
        if error := response.get("error"):
            if isinstance(error, Mapping):
                future.set_exception(RuntimeError(str(error.get("message") or "JSON-RPC error")))
            else:
                future.set_exception(RuntimeError("JSON-RPC error"))
        else:
            future.set_result(response.get("result"))
        return True


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
    set_active_host_rpc_client(host_client)
    pending_tasks: set[asyncio.Task[None]] = set()
    while True:
        payload = await asyncio.to_thread(read_frame, resolved_reader)
        if payload is None:
            if pending_tasks:
                await asyncio.gather(*pending_tasks)
            return
        task = asyncio.create_task(_handle_message(host, host_client, resolved_writer, payload))
        pending_tasks.add(task)
        task.add_done_callback(pending_tasks.discard)


async def _handle_message(
    host: Extension,
    host_client: StdioHostRPCClient,
    writer: BinaryWriter,
    payload: bytes,
) -> None:
    try:
        message = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        await write_message(
            writer,
            {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}},
        )
        return

    if (
        isinstance(message, Mapping)
        and not message.get("method")
        and host_client.handle_response(message)
    ):
        return
    if not isinstance(message, Mapping):
        return
    request_id = message.get("id")
    if request_id is None:
        return
    try:
        result = await _dispatch(host, message)
        await write_message(writer, {"jsonrpc": "2.0", "id": request_id, "result": result})
    except Exception as exc:
        await write_message(
            writer,
            {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32000, "message": str(exc)}},
        )


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
    if method == "$/cancelRequest":
        return None
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


async def write_message(writer: BinaryWriter, message: Mapping[str, Any]) -> None:
    """Write one LSP-style ``Content-Length`` framed JSON-RPC message.

    Args:
        writer: Binary output stream.
        message: JSON-serializable message mapping.
    """

    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    frame = b"Content-Length: " + str(len(payload)).encode("ascii") + b"\r\n\r\n" + payload
    await asyncio.to_thread(_write_and_flush, writer, frame)


def _write_and_flush(writer: BinaryWriter, frame: bytes) -> None:
    writer.write(frame)
    writer.flush()

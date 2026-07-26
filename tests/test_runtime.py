from __future__ import annotations

import asyncio
import json
import os
import queue
from collections.abc import Awaitable, Mapping
from typing import Any

import pytest

from kodelet_sdk import (
    BaseModel,
    CommandContext,
    CommandResult,
    Extension,
    Field,
    ToolContext,
    UIContext,
    UISurfaceInputEvent,
    UISurfaceResizeEvent,
)
from kodelet_sdk.runtime import StdioHostRPCClient, _StdioRequestState, run_stdio_server


@pytest.mark.asyncio
async def test_stdio_client_close_disconnects_open_surface_handles() -> None:
    writer = MemoryWriter()
    client = StdioHostRPCClient(writer)
    ui = UIContext(
        {"capabilities": {"ui": {"surfaces": True}}},
        client,
    )
    open_task = asyncio.create_task(ui.open_surface({"id": "game"}))
    open_request = await writer.read_frame()
    client.handle_response(
        {
            "jsonrpc": "2.0",
            "id": open_request["id"],
            "result": {"accepted": True},
        }
    )
    surface = await open_task
    input_events: list[UISurfaceInputEvent] = []
    surface.on_input(input_events.append)

    await client.close()
    surface.update(["after close"])
    client.handle_notification(
        "extension.ui.surface.input",
        {"id": "game", "sequence": 1, "kind": "key", "key": "q"},
    )
    await _settle_event_loop()
    await surface.close()

    assert input_events == []
    assert writer._buffer == bytearray()


@pytest.mark.asyncio
async def test_runtime_serves_json_rpc_and_reverse_host_rpc() -> None:
    ext = Extension(name="rpc")

    class EchoInput(BaseModel):
        text: str = Field(min_length=1)

    @ext.tool("echo", description="Echo text", input_schema=EchoInput)
    async def echo(input: EchoInput, ctx: Any) -> dict[str, str]:
        await ctx.update("Working", {"step": 1})
        answer = await ctx.ui.input({"title": "Choose"})
        return {"content": f"{input.text.upper()}:{answer}"}

    server_reader = MemoryReader()
    server_writer = MemoryWriter()
    task = asyncio.create_task(run_stdio_server(ext, server_reader, server_writer))
    client = RpcTestClient(server_reader, server_writer)

    init = await client.call(
        "extension.initialize",
        {
            "protocolVersion": "2026-05-30",
            "kodelet": {"version": "test"},
            "extension": {"id": "rpc", "cwd": os.getcwd(), "dataDir": ""},
            "capabilities": {"toolUpdates": True, "ui": {"input": True}},
        },
    )
    assert init["name"] == "rpc"
    assert init["tools"][0]["name"] == "echo"

    result = await client.call(
        "extension.tool.execute",
        {"name": "echo", "input": {"text": "hello"}, "context": {"cwd": os.getcwd()}},
    )
    assert result == {"content": "HELLO:from-host"}
    assert [request["method"] for request in client.host_requests] == [
        "kodelet.tool.update",
        "kodelet.ui.input",
    ]
    assert client.host_requests[0]["params"] == {
        "content": "Working",
        "data": {"step": 1},
    }
    assert [request["parentId"] for request in client.host_requests] == [2, 2]

    server_reader.close()
    await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_runtime_keeps_interactive_surfaces_alive_after_command_returns() -> None:
    ext = Extension(name="persistent-ui-rpc")
    background_tasks: set[asyncio.Future[Any]] = set()

    def track(awaitable: Awaitable[Any]) -> None:
        task = asyncio.ensure_future(awaitable)
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

    @ext.command("game", description="Open a persistent surface")
    async def game(_input: Any, ctx: CommandContext) -> CommandResult:
        surface = await ctx.ui.open_surface(
            {"id": "game", "initialLines": ["loading"], "width": "50%"}
        )

        def resize(event: UISurfaceResizeEvent) -> None:
            surface.update([f"size={event['width']}x{event['height']}"])
            track(
                ctx.ui.append_transcript(
                    {
                        "title": "Resized",
                        "message": f"{event['width']}x{event['height']}",
                    }
                )
            )

        def input_event(event: UISurfaceInputEvent) -> None:
            size = surface.size
            surface.update(
                [
                    f"key={event.get('key')};size="
                    f"{size['width'] if size else None}x{size['height'] if size else None}"
                ]
            )
            if event.get("key") == "q":
                async def close_later() -> None:
                    await asyncio.sleep(0)
                    await surface.close()

                track(close_later())

        surface.on_resize(resize)
        surface.on_input(input_event)
        return {"action": "respond", "response": "opened"}

    server_reader = MemoryReader()
    server_writer = MemoryWriter()
    task = asyncio.create_task(run_stdio_server(ext, server_reader, server_writer))
    client = RpcTestClient(server_reader, server_writer)
    await client.call(
        "extension.initialize",
        {
            "protocolVersion": "2026-05-30",
            "extension": {"id": "surface", "cwd": os.getcwd(), "dataDir": ""},
            "capabilities": {"ui": {"surfaces": True, "transcript": True}},
        },
    )
    result = await client.call(
        "extension.command.execute",
        {
            "name": "game",
            "input": {},
            "invocation": {"raw": "/game", "commandName": "game", "args": [], "flags": {}},
        },
    )
    assert result == {"action": "respond", "response": "opened"}
    open_request = next(
        request
        for request in client.host_requests
        if request["method"] == "kodelet.ui.surface.open"
    )
    assert "parentId" not in open_request
    assert open_request["params"] == {
        "id": "game",
        "options": {"width": "50%"},
        "frame": {"sequence": 1, "lines": ["loading"]},
    }

    client.notify(
        "extension.ui.surface.resize",
        {"id": "game", "sequence": 1, "width": 60, "height": 18},
    )
    resize_messages = await client.read_host_messages(2)
    frame_notification = next(
        message
        for message in resize_messages
        if message["method"] == "kodelet.ui.surface.frame"
    )
    transcript_request = next(
        message
        for message in resize_messages
        if message["method"] == "kodelet.ui.transcript.append"
    )
    assert frame_notification == {
        "jsonrpc": "2.0",
        "method": "kodelet.ui.surface.frame",
        "params": {"id": "game", "frame": {"sequence": 2, "lines": ["size=60x18"]}},
    }
    assert "parentId" not in transcript_request
    assert transcript_request["params"] == {"title": "Resized", "message": "60x18"}

    client.notify(
        "extension.ui.surface.input",
        {"id": "game", "sequence": 2, "kind": "key", "key": "q", "text": "q"},
    )
    input_messages = await client.read_host_messages(2)
    input_frame = next(
        message
        for message in input_messages
        if message["method"] == "kodelet.ui.surface.frame"
    )
    close_request = next(
        message
        for message in input_messages
        if message["method"] == "kodelet.ui.surface.close"
    )
    assert input_frame == {
        "jsonrpc": "2.0",
        "method": "kodelet.ui.surface.frame",
        "params": {
            "id": "game",
            "frame": {"sequence": 3, "lines": ["key=q;size=60x18"]},
        },
    }
    assert "parentId" not in close_request
    assert close_request["params"] == {"id": "game", "sequence": 4}

    if background_tasks:
        await asyncio.gather(*background_tasks)
    server_reader.close()
    await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_runtime_correlates_concurrent_reverse_rpc_requests() -> None:
    ext = Extension(name="concurrent-rpc")
    both_started = asyncio.Event()
    started = 0

    @ext.tool("ask", description="Ask concurrently", input_schema={"type": "object"})
    async def ask(input: Any, ctx: ToolContext) -> str:
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=1)
        answer = await ctx.ui.input({"title": input["label"]})
        return f"{input['label']}:{answer}"

    server_reader = MemoryReader()
    server_writer = MemoryWriter()
    task = asyncio.create_task(run_stdio_server(ext, server_reader, server_writer))
    client = RpcTestClient(server_reader, server_writer)

    await client.call(
        "extension.initialize",
        {
            "protocolVersion": "2026-05-30",
            "extension": {"id": "concurrent-rpc", "cwd": os.getcwd(), "dataDir": ""},
        },
    )

    for request_id, label in ((2, "first"), (3, "second")):
        server_reader.feed(
            _frame(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "extension.tool.execute",
                    "params": {"name": "ask", "input": {"label": label}},
                }
            )
        )

    reverse_requests = [
        await asyncio.wait_for(server_writer.read_frame(), timeout=1),
        await asyncio.wait_for(server_writer.read_frame(), timeout=1),
    ]
    reverse_by_title = {request["params"]["title"]: request for request in reverse_requests}
    assert reverse_by_title["first"]["parentId"] == 2
    assert reverse_by_title["second"]["parentId"] == 3

    for label in ("second", "first"):
        request = reverse_by_title[label]
        server_reader.feed(
            _frame(
                {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": {"status": "submitted", "value": f"{label}-answer"},
                }
            )
        )

    responses = [
        await asyncio.wait_for(server_writer.read_frame(), timeout=1),
        await asyncio.wait_for(server_writer.read_frame(), timeout=1),
    ]
    response_by_id = {response["id"]: response for response in responses}
    assert response_by_id[2]["result"] == {"content": "first:first-answer"}
    assert response_by_id[3]["result"] == {"content": "second:second-answer"}

    server_reader.close()
    await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_runtime_cancels_requests_and_blocks_late_reverse_rpc() -> None:
    ext = Extension(name="cancellable-rpc")
    started = asyncio.Event()
    cancelled = asyncio.Event()
    stale_blocked = asyncio.Event()

    @ext.tool("wait", description="Wait for cancellation", input_schema={"type": "object"})
    async def wait(input: Any, ctx: ToolContext) -> str:
        if input.get("quick"):
            return "quick result"
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            try:
                await ctx.update("stale update")
            except asyncio.CancelledError:
                stale_blocked.set()
            return "late result"
        raise AssertionError("wait completed without cancellation")

    server_reader = MemoryReader()
    server_writer = MemoryWriter()
    task = asyncio.create_task(run_stdio_server(ext, server_reader, server_writer))
    client = RpcTestClient(server_reader, server_writer)
    await client.call(
        "extension.initialize",
        {
            "protocolVersion": "2026-05-30",
            "extension": {"id": "cancellable-rpc", "cwd": os.getcwd(), "dataDir": ""},
            "capabilities": {"toolUpdates": True},
        },
    )

    server_reader.feed(
        _frame(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "extension.tool.execute",
                "params": {"name": "wait", "input": {}},
            }
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    server_reader.feed(
        _frame({"jsonrpc": "2.0", "method": "$/cancelRequest", "params": {"id": 2}})
    )
    await asyncio.wait_for(cancelled.wait(), timeout=1)
    await asyncio.wait_for(stale_blocked.wait(), timeout=1)

    server_reader.feed(
        _frame(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "extension.tool.execute",
                "params": {"name": "wait", "input": {"quick": True}},
            }
        )
    )
    response = await asyncio.wait_for(server_writer.read_frame(), timeout=1)
    assert response == {"jsonrpc": "2.0", "id": 2, "result": {"content": "quick result"}}

    server_reader.close()
    await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_runtime_shutdown_rejects_persistent_rpc_started_during_cancellation() -> None:
    ext = Extension(name="shutdown-rpc")
    started = asyncio.Event()
    cleanup_unblocked = asyncio.Event()

    @ext.tool("wait", description="Wait for connection shutdown", input_schema={})
    async def wait(_input: Any, ctx: ToolContext) -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            try:
                await ctx.ui.append_transcript("cleanup")
            except RuntimeError:
                cleanup_unblocked.set()
        return "done"

    server_reader = MemoryReader()
    server_writer = MemoryWriter()
    task = asyncio.create_task(run_stdio_server(ext, server_reader, server_writer))
    client = RpcTestClient(server_reader, server_writer)
    await client.call(
        "extension.initialize",
        {
            "protocolVersion": "2026-05-30",
            "extension": {"id": "shutdown", "cwd": os.getcwd(), "dataDir": ""},
            "capabilities": {"ui": {"transcript": True}},
        },
    )
    server_reader.feed(
        _frame(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "extension.tool.execute",
                "params": {"name": "wait", "input": {}},
            }
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    server_reader.close()

    await asyncio.wait_for(task, timeout=1)
    assert cleanup_unblocked.is_set()


@pytest.mark.asyncio
async def test_runtime_rechecks_request_generation_inside_write_lock() -> None:
    writer = MemoryWriter()
    client = StdioHostRPCClient(writer)
    state = _StdioRequestState(7)
    await client._write_lock.acquire()

    async def send_update() -> Any:
        return await client.request_for(
            state,
            "kodelet.tool.update",
            {"content": "stale"},
        )

    task = asyncio.create_task(send_update())
    await asyncio.sleep(0)
    await state.cancel()
    await client.finish_request(state, asyncio.CancelledError())
    client._write_lock.release()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert writer._buffer == bytearray()


@pytest.mark.asyncio
async def test_runtime_rechecks_terminal_generation_inside_write_lock() -> None:
    writer = MemoryWriter()
    client = StdioHostRPCClient(writer)
    state = _StdioRequestState(7)
    await state.finish()
    await client._write_lock.acquire()

    task = asyncio.create_task(
        client.send(
            {"jsonrpc": "2.0", "id": 7, "result": {"content": "old"}},
            state,
            terminal=True,
        )
    )
    await asyncio.sleep(0)
    await state.cancel()
    client._write_lock.release()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert writer._buffer == bytearray()


class MemoryReader:
    def __init__(self) -> None:
        self._queue: queue.Queue[bytes] = queue.Queue()
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> None:
        self._queue.put(chunk)

    def close(self) -> None:
        self.feed(b"")

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            chunk = self._queue.get(timeout=5)
            return chunk
        while len(self._buffer) < size:
            chunk = self._queue.get(timeout=5)
            if chunk == b"":
                break
            self._buffer.extend(chunk)
        data = bytes(self._buffer[:size])
        del self._buffer[:size]
        return data

    def readline(self, size: int = -1) -> bytes:
        while True:
            newline_index = self._buffer.find(b"\n")
            if newline_index != -1:
                end = newline_index + 1
                if size >= 0:
                    end = min(end, size)
                data = bytes(self._buffer[:end])
                del self._buffer[:end]
                return data
            chunk = self._queue.get(timeout=5)
            if chunk == b"":
                if not self._buffer:
                    return b""
                data = bytes(self._buffer)
                self._buffer.clear()
                return data
            self._buffer.extend(chunk)


class MemoryWriter:
    def __init__(self) -> None:
        self._buffer = bytearray()
        self._condition = queue.Queue()

    def write(self, data: bytes) -> int:
        self._buffer.extend(data)
        self._condition.put(None)
        return len(data)

    def flush(self) -> None:
        return None

    async def read_frame(self) -> dict[str, Any]:
        return await asyncio.to_thread(self.read_frame_blocking)

    def read_frame_blocking(self) -> dict[str, Any]:
        while True:
            frame = _try_read_frame_from_bytearray(self._buffer)
            if frame is not None:
                return json.loads(frame.decode("utf-8"))
            self._condition.get(timeout=5)


class RpcTestClient:
    def __init__(self, server_reader: MemoryReader, server_writer: MemoryWriter) -> None:
        self._server_reader = server_reader
        self._server_writer = server_writer
        self._next_id = 0
        self.host_requests: list[dict[str, Any]] = []
        self.host_notifications: list[dict[str, Any]] = []

    async def call(self, method: str, params: Any) -> Any:
        self._next_id += 1
        request_id = self._next_id
        self._server_reader.feed(
            _frame({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        )
        while True:
            response = await self._server_writer.read_frame()
            if response.get("method"):
                self._handle_host_message(response)
                continue
            if response.get("id") != request_id:
                continue
            if "error" in response:
                raise RuntimeError(response["error"]["message"])
            return response.get("result")

    def notify(self, method: str, params: Any | None = None) -> None:
        self._server_reader.feed(
            _frame({"jsonrpc": "2.0", "method": method, "params": params})
        )

    async def read_host_messages(self, count: int) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        while len(messages) < count:
            message = await self._server_writer.read_frame()
            if not message.get("method"):
                continue
            self._handle_host_message(message)
            messages.append(message)
        return messages

    def _handle_host_message(self, message: dict[str, Any]) -> None:
        if message.get("id") is None:
            self.host_notifications.append(message)
            return
        self.host_requests.append(message)
        method = message.get("method")
        if method in {
            "kodelet.ui.widget.set",
            "kodelet.ui.widget.remove",
            "kodelet.ui.surface.open",
            "kodelet.ui.surface.close",
        }:
            params = message.get("params")
            sequence = 0
            if isinstance(params, Mapping):
                frame = params.get("frame")
                if isinstance(params.get("sequence"), int):
                    sequence = params["sequence"]
                elif isinstance(frame, Mapping) and isinstance(frame.get("sequence"), int):
                    sequence = frame["sequence"]
            result: Any = {"accepted": True, "latestSequence": sequence}
        elif method == "kodelet.ui.transcript.append":
            result = {"accepted": True}
        else:
            result = {"status": "submitted", "value": "from-host"}
        self._server_reader.feed(
            _frame({"jsonrpc": "2.0", "id": message["id"], "result": result})
        )


async def _settle_event_loop() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


def _frame(message: dict[str, Any]) -> bytes:
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    return b"Content-Length: " + str(len(payload)).encode("ascii") + b"\r\n\r\n" + payload


def _try_read_frame_from_bytearray(buffer: bytearray) -> bytes | None:
    header_end = buffer.find(b"\r\n\r\n")
    if header_end == -1:
        return None
    header = buffer[:header_end].decode("ascii")
    length = None
    for line in header.splitlines():
        key, _, value = line.partition(":")
        if key.lower() == "content-length":
            length = int(value.strip())
    assert length is not None
    start = header_end + 4
    end = start + length
    if len(buffer) < end:
        return None
    payload = bytes(buffer[start:end])
    del buffer[:end]
    return payload

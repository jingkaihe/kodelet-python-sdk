from __future__ import annotations

import asyncio
import gc
import json
import os
import queue
from collections.abc import Awaitable, Mapping
from typing import Any, cast

import pytest

from kodelet_sdk import (
    BackgroundTaskLease,
    BaseModel,
    CommandContext,
    CommandResult,
    Extension,
    Field,
    HostRPCError,
    ShortcutContext,
    ShortcutResult,
    ToolContext,
    ToolExecutionResult,
    ToolPresentation,
    UIContext,
    UISurfaceInputEvent,
    UISurfaceResizeEvent,
)
from kodelet_sdk.runtime import (
    ExtensionMessageDispatcher,
    StdioHostRPCClient,
    _RequestScopedHostRPCClient,
    _StdioRequestState,
    run_stdio_server,
)


@pytest.mark.asyncio
async def test_stdio_client_preserves_host_rpc_error_code() -> None:
    writer = MemoryWriter()
    client = StdioHostRPCClient(writer)
    task = asyncio.create_task(client.request("kodelet.conversation.fork"))
    request = await writer.read_frame()

    assert client.handle_response(
        {
            "jsonrpc": "2.0",
            "id": request["id"],
            "error": {"code": -32004, "message": "fork unavailable"},
        }
    )
    with pytest.raises(HostRPCError, match="fork unavailable") as exc_info:
        await task
    assert exc_info.value.code == -32004


@pytest.mark.asyncio
async def test_native_surface_revoked_while_opening_cannot_activate() -> None:
    writer = MemoryWriter()
    client = StdioHostRPCClient(writer)
    ui = UIContext({"capabilities": {"ui": {"surfaces": True}}}, client, "conversation")
    opening = asyncio.create_task(ui.open_surface({"id": "canvas"}))
    request = await writer.read_frame()
    client.handle_notification(
        "extension.ui.surface.closed",
        {
            "scopeId": "conversation",
            "id": "canvas",
            "openSequence": request["params"]["frame"]["sequence"],
        },
    )
    client.handle_response({"jsonrpc": "2.0", "id": request["id"], "result": {"accepted": True}})
    with pytest.raises(RuntimeError, match=r"closed.*while.*opening"):
        await opening
    await client.close()


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
    shortcut_contexts: list[tuple[str | None, str | None]] = []

    class EchoInput(BaseModel):
        text: str = Field(min_length=1)

    @ext.tool("echo", description="Echo text", input_schema=EchoInput)
    async def echo(input: EchoInput, ctx: Any) -> ToolExecutionResult:
        await ctx.update("Working", {"step": 1})
        answer = await ctx.ui.input({"title": "Choose"})
        presentation: ToolPresentation = {
            "summary": "Echo complete",
            "body": f"Returned `{input.text.upper()}`.",
            "format": "markdown",
        }
        return {
            "content": f"{input.text.upper()}:{answer}",
            "data": {"presentation": presentation},
        }

    @ext.shortcut("ctrl+alt+r", description="Refresh project context")
    async def refresh(ctx: ShortcutContext) -> ShortcutResult:
        shortcut_contexts.append((ctx.conversation_id, ctx.recipe_name))
        await ctx.ui.notify("Refreshed")
        return {"action": "submit", "message": "/refresh"}

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
            "capabilities": {
                "toolUpdates": True,
                "shortcuts": {"submit": True},
                "ui": {"input": True},
            },
        },
    )
    assert init["name"] == "rpc"
    assert init["tools"][0]["name"] == "echo"
    assert init["shortcuts"] == [{"key": "ctrl+alt+r", "description": "Refresh project context"}]

    result = await client.call(
        "extension.tool.execute",
        {"name": "echo", "input": {"text": "hello"}, "context": {"cwd": os.getcwd()}},
    )
    assert result == {
        "content": "HELLO:from-host",
        "data": {
            "presentation": {
                "summary": "Echo complete",
                "body": "Returned `HELLO`.",
                "format": "markdown",
            }
        },
    }
    shortcut_result = await client.call(
        "extension.shortcut.execute",
        {
            "key": "alt+control+r",
            "context": {"conversationId": "conv-shortcut", "recipeName": "review"},
        },
    )
    assert shortcut_result == {"action": "submit", "message": "/refresh"}
    assert shortcut_contexts == [("conv-shortcut", "review")]
    assert [request["method"] for request in client.host_requests] == [
        "kodelet.tool.update",
        "kodelet.ui.input",
        "kodelet.ui.notify",
    ]
    assert client.host_requests[0]["params"] == {
        "content": "Working",
        "data": {"step": 1},
    }
    assert [request["parentId"] for request in client.host_requests] == [2, 2, 3]

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
            "context": {"uiScopeId": "conversation-a"},
            "invocation": {"raw": "/game", "commandName": "game", "args": [], "flags": {}},
        },
    )
    assert result == {"action": "respond", "response": "opened"}
    open_request = next(
        request
        for request in client.host_requests
        if request["method"] == "kodelet.ui.surface.open"
    )
    assert open_request["parentId"] == 2
    assert open_request["params"] == {
        "id": "game",
        "scopeId": "conversation-a",
        "options": {"width": "50%"},
        "frame": {"sequence": 1, "lines": ["loading"]},
    }

    client.notify(
        "extension.ui.surface.resize",
        {
            "id": "game",
            "scopeId": "conversation-a",
            "sequence": 1,
            "width": 60,
            "height": 18,
        },
    )
    resize_messages = await client.read_host_messages(2)
    frame_notification = next(
        message for message in resize_messages if message["method"] == "kodelet.ui.surface.frame"
    )
    transcript_request = next(
        message
        for message in resize_messages
        if message["method"] == "kodelet.ui.transcript.append"
    )
    assert frame_notification == {
        "jsonrpc": "2.0",
        "method": "kodelet.ui.surface.frame",
        "params": {
            "id": "game",
            "scopeId": "conversation-a",
            "frame": {"sequence": 2, "lines": ["size=60x18"]},
        },
    }
    assert "parentId" not in transcript_request
    assert transcript_request["params"] == {
        "title": "Resized",
        "message": "60x18",
        "scopeId": "conversation-a",
    }

    client.notify(
        "extension.ui.surface.input",
        {
            "id": "game",
            "scopeId": "conversation-a",
            "sequence": 2,
            "kind": "key",
            "key": "q",
            "text": "q",
        },
    )
    input_messages = await client.read_host_messages(2)
    input_frame = next(
        message for message in input_messages if message["method"] == "kodelet.ui.surface.frame"
    )
    close_request = next(
        message for message in input_messages if message["method"] == "kodelet.ui.surface.close"
    )
    assert input_frame == {
        "jsonrpc": "2.0",
        "method": "kodelet.ui.surface.frame",
        "params": {
            "id": "game",
            "scopeId": "conversation-a",
            "frame": {"sequence": 3, "lines": ["key=q;size=60x18"]},
        },
    }
    assert "parentId" not in close_request
    assert close_request["params"] == {
        "id": "game",
        "sequence": 4,
        "scopeId": "conversation-a",
    }

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
async def test_runtime_correlates_concurrent_persistent_widget_requests() -> None:
    ext = Extension(name="concurrent-widgets")
    both_started = asyncio.Event()
    started = 0

    @ext.tool("widget", description="Update a persistent widget", input_schema={"type": "object"})
    async def widget(input: Any, ctx: ToolContext) -> str:
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=1)
        await ctx.ui.set_widget("todo-progress", [input["label"]])
        return input["label"]

    server_reader = MemoryReader()
    server_writer = MemoryWriter()
    task = asyncio.create_task(run_stdio_server(ext, server_reader, server_writer))
    client = RpcTestClient(server_reader, server_writer)

    await client.call(
        "extension.initialize",
        {
            "protocolVersion": "2026-05-30",
            "extension": {"id": "concurrent-widgets", "cwd": os.getcwd(), "dataDir": ""},
            "capabilities": {"ui": {"widgets": True}},
        },
    )

    for request_id, label, scope_id in (
        (2, "first", "conversation-a"),
        (3, "second", "conversation-b"),
    ):
        server_reader.feed(
            _frame(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "extension.tool.execute",
                    "params": {
                        "name": "widget",
                        "input": {"label": label},
                        "context": {"uiScopeId": scope_id},
                    },
                }
            )
        )

    widget_requests = [
        await asyncio.wait_for(server_writer.read_frame(), timeout=1),
        await asyncio.wait_for(server_writer.read_frame(), timeout=1),
    ]
    widget_by_label = {
        request["params"]["frame"]["lines"][0]: request for request in widget_requests
    }
    assert widget_by_label["first"]["parentId"] == 2
    assert widget_by_label["second"]["parentId"] == 3
    assert widget_by_label["first"]["params"]["scopeId"] == "conversation-a"
    assert widget_by_label["second"]["params"]["scopeId"] == "conversation-b"
    assert [request["params"]["frame"]["sequence"] for request in widget_requests] == [1, 1]

    for request in widget_requests:
        sequence = request["params"]["frame"]["sequence"]
        server_reader.feed(
            _frame(
                {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": {"accepted": True, "latestSequence": sequence},
                }
            )
        )

    responses = [
        await asyncio.wait_for(server_writer.read_frame(), timeout=1),
        await asyncio.wait_for(server_writer.read_frame(), timeout=1),
    ]
    response_by_id = {response["id"]: response for response in responses}
    assert response_by_id[2]["result"] == {"content": "first"}
    assert response_by_id[3]["result"] == {"content": "second"}

    server_reader.close()
    await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_state", ["cancelled", "completed"])
async def test_stdio_persistent_request_is_not_replayed_after_request_ends(
    terminal_state: str,
) -> None:
    calls: list[str] = []
    state = _StdioRequestState(7)

    class FakeClient:
        async def request_for(
            self,
            request_state: _StdioRequestState,
            method: str,
            params: Any | None = None,
        ) -> Any:
            del method, params
            assert request_state is state
            calls.append("parented")
            if terminal_state == "cancelled":
                await state.cancel()
                raise asyncio.CancelledError
            await state.finish()
            raise RuntimeError("Extension request completed")

        async def request(self, method: str, params: Any | None = None) -> Any:
            del method, params
            calls.append("parentless")
            return {"accepted": True}

    client = _RequestScopedHostRPCClient(cast(StdioHostRPCClient, FakeClient()), state)
    if terminal_state == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await client.request_persistent("kodelet.ui.transcript.append", {"message": "one"})
    else:
        with pytest.raises(RuntimeError, match="completed"):
            await client.request_persistent("kodelet.ui.transcript.append", {"message": "one"})

    assert calls == ["parented"]


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_state", ["completed", "cancelled"])
async def test_background_lease_can_be_released_after_originating_request_ends(
    terminal_state: str,
) -> None:
    writer = MemoryWriter()
    transport = StdioHostRPCClient(writer)
    state = _StdioRequestState(7)
    scoped = _RequestScopedHostRPCClient(transport, state)
    lease = BackgroundTaskLease(scoped, "lease")
    if terminal_state == "cancelled":
        await state.cancel()
    else:
        await transport.finish_request(state)

    pending = asyncio.create_task(lease.close())
    request = await writer.read_frame()
    assert not state.active
    assert "parentId" not in request
    assert request["method"] == "kodelet.runtime.background.release"
    assert request["params"] == {"leaseId": "lease"}
    assert transport.handle_response({"jsonrpc": "2.0", "id": request["id"], "result": {}})
    await asyncio.wait_for(pending, timeout=1)
    await asyncio.wait_for(lease.close(), timeout=1)
    await transport.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "late_response", [{"result": {}}, {"error": {"code": -1, "message": "late"}}]
)
async def test_late_reverse_response_cannot_cancel_colliding_forward_request(
    late_response: dict[str, Any],
) -> None:
    ext = Extension(name="duplex-ids")
    update_cancelled = asyncio.Event()
    release = asyncio.Event()
    driver_cancelled = asyncio.Event()
    updating: asyncio.Task[None] | None = None

    @ext.tool("driver", description="Retain a handler across a late response", input_schema={})
    async def driver(_input: Any, ctx: ToolContext) -> str:
        nonlocal updating
        try:
            await ctx.update("starting")
            updating = asyncio.create_task(ctx.update("waiting"))
            try:
                await updating
            except asyncio.CancelledError:
                update_cancelled.set()
            await release.wait()
            return "driver remained active"
        except asyncio.CancelledError:
            driver_cancelled.set()
            raise

    @ext.tool("release", description="Barrier after processing a late response", input_schema={})
    async def release_driver(_input: Any, _ctx: ToolContext) -> str:
        release.set()
        return "released"

    reader, writer = MemoryReader(), MemoryWriter()
    server = asyncio.create_task(run_stdio_server(ext, reader, writer))
    try:
        await RpcTestClient(reader, writer).call(
            "extension.initialize", {"capabilities": {"tools": {"updates": True}}}
        )
        reader.feed(
            _frame(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "extension.tool.execute",
                    "params": {"name": "driver", "input": {}},
                }
            )
        )
        start = await asyncio.wait_for(writer.read_frame(), timeout=1)
        assert start["method"] == "kodelet.tool.update"
        reader.feed(
            _frame(
                {
                    "jsonrpc": "2.0",
                    "id": start["id"],
                    "result": {},
                }
            )
        )
        request = await asyncio.wait_for(writer.read_frame(), timeout=1)
        assert request["method"] == "kodelet.tool.update"
        assert request["id"] == 2, "opposite directions may use the same numeric ID"
        assert updating is not None
        updating.cancel()
        await asyncio.wait_for(update_cancelled.wait(), timeout=1)
        reader.feed(_frame({"jsonrpc": "2.0", "id": 2, **late_response}))
        reader.feed(
            _frame(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "extension.tool.execute",
                    "params": {"name": "release", "input": {}},
                }
            )
        )
        responses = [await asyncio.wait_for(writer.read_frame(), timeout=1) for _ in range(2)]
        assert not driver_cancelled.is_set()
        assert {response["id"]: response.get("result") for response in responses} == {
            2: {"content": "driver remained active"},
            3: {"content": "released"},
        }
        assert all("error" not in response for response in responses)
    finally:
        reader.close()
        await asyncio.wait_for(server, timeout=1)


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
    server_reader.feed(_frame({"jsonrpc": "2.0", "method": "$/cancelRequest", "params": {"id": 2}}))
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
async def test_runtime_cancels_shortcut_handlers() -> None:
    ext = Extension(name="cancellable-shortcut")
    started = asyncio.Event()
    cancelled = asyncio.Event()

    @ext.shortcut("ctrl+r", description="Wait for cancellation")
    async def wait(ctx: ShortcutContext) -> None:
        if ctx.profile == "quick":
            return
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    server_reader = MemoryReader()
    server_writer = MemoryWriter()
    task = asyncio.create_task(run_stdio_server(ext, server_reader, server_writer))
    client = RpcTestClient(server_reader, server_writer)
    await client.call(
        "extension.initialize",
        {
            "protocolVersion": "2026-05-30",
            "extension": {"id": "cancellable-shortcut", "cwd": os.getcwd(), "dataDir": ""},
        },
    )

    server_reader.feed(
        _frame(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "extension.shortcut.execute",
                "params": {"key": "ctrl+r"},
            }
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    server_reader.feed(_frame({"jsonrpc": "2.0", "method": "$/cancelRequest", "params": {"id": 2}}))
    await asyncio.wait_for(cancelled.wait(), timeout=1)

    server_reader.feed(
        _frame(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "extension.shortcut.execute",
                "params": {"key": "ctrl+r", "context": {"profile": "quick"}},
            }
        )
    )
    response = await asyncio.wait_for(server_writer.read_frame(), timeout=1)
    assert response == {"jsonrpc": "2.0", "id": 3, "result": None}

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
@pytest.mark.parametrize("raw_transport", [False, True], ids=["stdio", "raw"])
async def test_runtime_rechecks_request_generation_inside_write_lock(raw_transport: bool) -> None:
    writer = MemoryWriter()
    messages: asyncio.Queue[Mapping[str, Any]] = asyncio.Queue()
    client = (
        StdioHostRPCClient(send_message=messages.put)
        if raw_transport
        else StdioHostRPCClient(writer)
    )
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
    assert messages.empty()
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_transport", [False, True], ids=["stdio", "raw"])
async def test_runtime_rechecks_terminal_generation_inside_write_lock(raw_transport: bool) -> None:
    writer = MemoryWriter()
    messages: asyncio.Queue[Mapping[str, Any]] = asyncio.Queue()
    client = (
        StdioHostRPCClient(send_message=messages.put)
        if raw_transport
        else StdioHostRPCClient(writer)
    )
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
    assert messages.empty()
    await client.close()


def test_stdio_client_requires_exactly_one_transport() -> None:
    messages: asyncio.Queue[Mapping[str, Any]] = asyncio.Queue()
    with pytest.raises(ValueError, match="exactly one"):
        StdioHostRPCClient()
    with pytest.raises(ValueError, match="exactly one"):
        StdioHostRPCClient(MemoryWriter(), send_message=messages.put)


@pytest.mark.asyncio
async def test_async_transport_preserves_raw_messages_and_serializes_sends() -> None:
    messages: list[Mapping[str, Any]] = []
    entered = asyncio.Event()
    release = asyncio.Event()

    async def send(message: Mapping[str, Any]) -> None:
        messages.append(message)
        entered.set()
        await release.wait()

    client = StdioHostRPCClient(send_message=send)
    raw = {
        "jsonrpc": "2.0",
        "id": "raw-id",
        "parentId": "parent-id",
        "method": "custom.request",
        "params": {"nested": [1, None, "π"]},
        "_meta": {"futureField": True},
    }
    first = asyncio.create_task(client.send(raw))
    second: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        second = asyncio.create_task(client.notify("custom.notification", {"value": 2}))
        await _settle_event_loop()
        assert messages == [raw]
        assert messages[0] is raw
        assert not second.done()
        release.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
        assert messages[1] == {
            "jsonrpc": "2.0", "method": "custom.notification", "params": {"value": 2}
        }
    finally:
        release.set()
        await asyncio.gather(first, *([second] if second is not None else []))
        await client.close()


@pytest.mark.asyncio
async def test_async_transport_rechecks_closed_connection_inside_write_lock() -> None:
    messages: asyncio.Queue[Mapping[str, Any]] = asyncio.Queue()
    client = StdioHostRPCClient(send_message=messages.put)
    await client._write_lock.acquire()
    pending = asyncio.create_task(client.notify("custom.notification"))
    await _settle_event_loop()
    await client.close()
    client._write_lock.release()
    with pytest.raises(RuntimeError, match="connection is closed"):
        await pending
    assert messages.empty()


@pytest.mark.asyncio
async def test_message_dispatcher_round_trip_with_updates_and_reverse_rpc() -> None:
    ext = Extension(name="raw-rpc")

    @ext.tool("echo", description="Exercise reverse RPC", input_schema={})
    async def echo(input: Any, ctx: ToolContext) -> str:
        await ctx.update("working", {"text": input["text"]})
        answer = await ctx.ui.input({"title": input["text"]})
        return f"{input['text']}:{answer}"

    messages: asyncio.Queue[Mapping[str, Any]] = asyncio.Queue()
    errors: list[Exception] = []
    notifications: list[tuple[str, Any]] = []
    client = StdioHostRPCClient(send_message=messages.put)
    client.on_notification(lambda method, params: notifications.append((method, params)))
    dispatcher = ExtensionMessageDispatcher(ext, client, on_error=errors.append)
    try:
        await dispatcher.handle_message({
            "jsonrpc": "2.0", "id": "init", "method": "extension.initialize",
            "params": {"capabilities": {"tools": {"updates": True}}},
        })
        init = await asyncio.wait_for(messages.get(), timeout=1)
        assert init["result"]["name"] == "raw-rpc"
        assert init["result"]["tools"][0]["name"] == "echo"
        await asyncio.wait_for(dispatcher.handle_message({
            "jsonrpc": "2.0", "id": 1, "method": "extension.tool.execute",
            "params": {"name": "echo", "input": {"text": "local"}},
        }), timeout=1)
        update = await asyncio.wait_for(messages.get(), timeout=1)
        assert update == {
            "jsonrpc": "2.0", "id": 1, "parentId": 1,
            "method": "kodelet.tool.update",
            "params": {"content": "working", "data": {"text": "local"}},
        }
        await dispatcher.handle_message({"jsonrpc": "2.0", "id": 1, "result": {}})
        reverse = await asyncio.wait_for(messages.get(), timeout=1)
        assert reverse["method"] == "kodelet.ui.input"
        assert reverse["parentId"] == 1
        await dispatcher.handle_message({
            "jsonrpc": "2.0", "method": "custom.notification", "params": {"raw": True}
        })
        assert notifications == [("custom.notification", {"raw": True})]
        await dispatcher.handle_message({
            "jsonrpc": "2.0", "id": reverse["id"],
            "result": {"status": "submitted", "value": "host"},
        })
        assert await asyncio.wait_for(messages.get(), timeout=1) == {
            "jsonrpc": "2.0", "id": 1, "result": {"content": "local:host"}
        }
        assert errors == []
    finally:
        await dispatcher.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["cancel", "reuse", "close"])
async def test_message_dispatcher_cancels_pending_callbacks_without_replay(operation: str) -> None:
    ext = Extension(name="cancel-raw-rpc")
    cancelled = asyncio.Event()

    @ext.tool("wait", description="Wait for reverse RPC", input_schema={})
    async def wait(input: Any, ctx: ToolContext) -> str:
        if input.get("quick"):
            return "fresh"
        try:
            return str(await ctx.ui.input({"title": "blocked"}))
        except asyncio.CancelledError:
            cancelled.set()
            return "stale"

    messages: asyncio.Queue[Mapping[str, Any]] = asyncio.Queue()
    errors: list[Exception] = []
    client = StdioHostRPCClient(send_message=messages.put)
    dispatcher = ExtensionMessageDispatcher(ext, client, on_error=errors.append)
    request = {
        "jsonrpc": "2.0", "id": "tool", "method": "extension.tool.execute",
        "params": {"name": "wait", "input": {}},
    }
    try:
        await dispatcher.handle_message(request)
        reverse = await asyncio.wait_for(messages.get(), timeout=1)
        assert reverse["parentId"] == "tool"
        assert client._pending
        if operation == "close":
            await asyncio.wait_for(dispatcher.close(), timeout=1)
        else:
            if operation == "cancel":
                await dispatcher.handle_message({
                    "jsonrpc": "2.0", "method": "$/cancelRequest", "params": {"id": "tool"}
                })
            await dispatcher.handle_message({
                **request, "params": {"name": "wait", "input": {"quick": True}}
            })
            await dispatcher.handle_message({
                "jsonrpc": "2.0", "id": reverse["id"],
                "result": {"status": "submitted", "value": "late"},
            })
            assert await asyncio.wait_for(messages.get(), timeout=1) == {
                "jsonrpc": "2.0", "id": "tool", "result": {"content": "fresh"}
            }
        await asyncio.wait_for(cancelled.wait(), timeout=1)
    finally:
        await dispatcher.close()
    assert messages.empty()
    assert not client._pending
    assert not dispatcher._pending_tasks
    assert not dispatcher._request_states
    assert errors == []
    with pytest.raises(RuntimeError, match="dispatcher is closed"):
        await dispatcher.handle_message(request)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["cancel", "close"])
async def test_message_dispatcher_consumes_failed_future_during_blocked_send(
    operation: str,
) -> None:
    ext = Extension(name="blocked-send")
    sending = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()

    @ext.tool("wait", description="Wait for transport acknowledgement", input_schema={})
    async def wait(_input: Any, ctx: ToolContext) -> str:
        return str(await ctx.ui.input({"title": "blocked"}))

    async def send(_message: Mapping[str, Any]) -> None:
        sending.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()

    loop = asyncio.get_running_loop()
    loop_errors: list[dict[str, Any]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
    client = StdioHostRPCClient(send_message=send)
    dispatcher = ExtensionMessageDispatcher(ext, client)
    stopping: asyncio.Task[None] | None = None
    try:
        await dispatcher.handle_message({
            "jsonrpc": "2.0", "id": "tool", "method": "extension.tool.execute",
            "params": {"name": "wait", "input": {}},
        })
        await asyncio.wait_for(sending.wait(), timeout=1)
        future = next(iter(client._pending.values()))[1]
        future_done = asyncio.Event()
        future.add_done_callback(lambda _future: future_done.set())
        stopping = asyncio.create_task(
            dispatcher.close() if operation == "close" else dispatcher.handle_message({
                "jsonrpc": "2.0", "method": "$/cancelRequest", "params": {"id": "tool"}
            })
        )
        await asyncio.wait_for(cleaning.wait(), timeout=1)
        await asyncio.wait_for(future_done.wait(), timeout=1)
        assert not future.cancelled(), "finish_request rejected the future while send was yielding"
        release.set()
        await asyncio.wait_for(stopping, timeout=1)
        await asyncio.wait_for(dispatcher.close(), timeout=1)
        assert not client._pending
        del future
        gc.collect()
        assert loop_errors == []
    finally:
        release.set()
        await asyncio.wait_for(dispatcher.close(), timeout=1)
        if stopping is not None:
            await asyncio.gather(stopping, return_exceptions=True)
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
@pytest.mark.parametrize("response_fails", [False, True], ids=["unanswered", "closed"])
async def test_async_transport_failure_keeps_send_error_and_cleans_response_future(
    response_fails: bool,
) -> None:
    sending = asyncio.Event()
    release = asyncio.Event()
    failure = OSError("transport acknowledgement failed")

    async def send(_message: Mapping[str, Any]) -> None:
        sending.set()
        await release.wait()
        raise failure

    client = StdioHostRPCClient(send_message=send)
    pending = asyncio.create_task(client.request("kodelet.ui.input"))
    try:
        await asyncio.wait_for(sending.wait(), timeout=1)
        future = next(iter(client._pending.values()))[1]
        if response_fails:
            await client.close()
        release.set()
        with pytest.raises(OSError, match="transport acknowledgement failed") as caught:
            await asyncio.wait_for(pending, timeout=1)
        assert caught.value is failure
        assert not client._pending
        if response_fails:
            # Checking the logging flag does not itself retrieve the exception.
            assert not cast(Any, future)._log_traceback
        else:
            assert future.cancelled()
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        await client.close()


@pytest.mark.asyncio
async def test_message_dispatcher_close_survives_caller_cancellation() -> None:
    ext = Extension(name="close-raw-rpc")
    started = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()

    @ext.tool("wait", description="Wait during shutdown", input_schema={})
    async def wait(_input: Any, _ctx: ToolContext) -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
        return "done"

    messages: asyncio.Queue[Mapping[str, Any]] = asyncio.Queue()
    client = StdioHostRPCClient(send_message=messages.put)
    dispatcher = ExtensionMessageDispatcher(ext, client)
    await dispatcher.handle_message({
        "jsonrpc": "2.0", "id": 1, "method": "extension.tool.execute",
        "params": {"name": "wait", "input": {}},
    })
    await asyncio.wait_for(started.wait(), timeout=1)
    pending = asyncio.create_task(client.request("persistent.request"))
    assert "parentId" not in await asyncio.wait_for(messages.get(), timeout=1)
    closing = asyncio.create_task(dispatcher.close())
    try:
        await asyncio.wait_for(cleaning.wait(), timeout=1)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        with pytest.raises(RuntimeError, match="connection closed"):
            await asyncio.wait_for(pending, timeout=1)
        assert dispatcher._close_task is not None
        assert not dispatcher._close_task.done()
    finally:
        release.set()
        await asyncio.wait_for(dispatcher.close(), timeout=1)
        await asyncio.gather(pending, closing, return_exceptions=True)
    assert not dispatcher._pending_tasks
    assert not dispatcher._request_states
    assert not client._pending
    assert messages.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("handler_fails", [False, True], ids=["result", "error"])
async def test_message_dispatcher_reports_send_failure_without_retry(handler_fails: bool) -> None:
    ext = Extension(name="broken-raw-rpc")

    @ext.tool("test", description="Return or raise", input_schema={})
    async def execute(_input: Any, _ctx: ToolContext) -> str:
        if handler_fails:
            raise ValueError("handler failed")
        return "done"

    sent: list[Mapping[str, Any]] = []
    failure = OSError("transport failed")
    reported: asyncio.Queue[Exception] = asyncio.Queue()

    async def send(message: Mapping[str, Any]) -> None:
        sent.append(message)
        if len(sent) == 1:
            raise failure

    client = StdioHostRPCClient(send_message=send)
    dispatcher = ExtensionMessageDispatcher(ext, client, on_error=reported.put_nowait)
    try:
        await dispatcher.handle_message({
            "jsonrpc": "2.0", "id": "request", "method": "extension.tool.execute",
            "params": {"name": "test", "input": {}},
        })
        assert await asyncio.wait_for(reported.get(), timeout=1) is failure
        assert len(sent) == 1
        assert sent[0]["id"] == "request"
        if handler_fails:
            assert sent[0]["error"] == {"code": -32000, "message": "handler failed"}
        else:
            assert sent[0]["result"] == {"content": "done"}
        assert not dispatcher._pending_tasks
        assert not dispatcher._request_states
        assert reported.empty()
    finally:
        await dispatcher.close()


@pytest.mark.asyncio
async def test_message_dispatcher_returns_handler_errors_as_protocol_responses() -> None:
    messages: asyncio.Queue[Mapping[str, Any]] = asyncio.Queue()
    errors: list[Exception] = []
    client = StdioHostRPCClient(send_message=messages.put)
    dispatcher = ExtensionMessageDispatcher(Extension(), client, on_error=errors.append)
    try:
        await dispatcher.handle_message({
            "jsonrpc": "2.0", "id": "unknown", "method": "extension.unknown", "params": {}
        })
        assert await asyncio.wait_for(messages.get(), timeout=1) == {
            "jsonrpc": "2.0", "id": "unknown",
            "error": {"code": -32000, "message": "Unknown JSON-RPC method: extension.unknown"},
        }
    finally:
        await dispatcher.close()
    assert errors == []


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
        self._server_reader.feed(_frame({"jsonrpc": "2.0", "method": method, "params": params}))

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
        self._server_reader.feed(_frame({"jsonrpc": "2.0", "id": message["id"], "result": result}))


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

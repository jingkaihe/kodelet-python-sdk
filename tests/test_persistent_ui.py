from __future__ import annotations

import asyncio
import gc
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import pytest

from kodelet_sdk import (
    CommandContext,
    CommandResult,
    Extension,
    UIContext,
    UISurface,
    UISurfaceInputEvent,
    UISurfaceResizeEvent,
    create_test_harness,
    set_active_host_rpc_client,
)


@pytest.mark.asyncio
async def test_widgets_use_sequences_and_surfaces_route_host_events() -> None:
    opened_surface: UISurface | None = None
    input_events: list[UISurfaceInputEvent] = []
    resize_events: list[UISurfaceResizeEvent] = []
    requests: list[tuple[str, Any]] = []
    notification_handlers: set[Callable[[str, Any], None]] = set()

    class FakeRPC:
        async def request(self, method: str, params: Any | None = None) -> Any:
            requests.append((method, params))
            return {"accepted": True, "latestSequence": 1}

        def on_notification(
            self,
            handler: Callable[[str, Any], None],
        ) -> Callable[[], None]:
            notification_handlers.add(handler)
            return lambda: notification_handlers.discard(handler)

    ext = Extension()

    @ext.command("ui", description="Open extension UI")
    async def open_ui(_input: Any, ctx: CommandContext) -> CommandResult:
        nonlocal opened_surface
        await ctx.ui.set_widget(
            "status",
            [
                "ready",
                {
                    "spans": [
                        {
                            "text": "green",
                            "style": {"foreground": "#00ff00", "bold": True},
                        }
                    ]
                },
            ],
        )
        await ctx.ui.set_widget("status", ["updated"], {"placement": "belowComposer"})
        await ctx.ui.set_widget("status", None)
        await ctx.ui.append_transcript({"title": "Saved", "message": "./drawing.png"})
        opened_surface = await ctx.ui.open_surface(
            {
                "id": "game",
                "initialLines": ["loading"],
                "width": "75%",
                "maxHeight": "95%",
                "anchor": "center",
            }
        )
        opened_surface.on_input(input_events.append)
        opened_surface.on_resize(resize_events.append)
        return {"action": "respond", "response": "opened"}

    harness = await create_test_harness(ext, FakeRPC())
    harness.initialize(
        {"capabilities": {"ui": {"widgets": True, "surfaces": True, "transcript": True}}}
    )
    await harness.execute_command(
        {
            "name": "ui",
            "context": {"uiScopeId": "conversation-a"},
            "invocation": {"raw": "/ui", "commandName": "ui", "args": [], "flags": {}},
        }
    )

    assert requests[:5] == [
        (
            "kodelet.ui.widget.set",
            {
                "id": "status",
                "scopeId": "conversation-a",
                "placement": "aboveComposer",
                "frame": {
                    "sequence": 1,
                    "lines": [
                        "ready",
                        {
                            "spans": [
                                {
                                    "text": "green",
                                    "style": {"foreground": "#00ff00", "bold": True},
                                }
                            ]
                        },
                    ],
                },
            },
        ),
        (
            "kodelet.ui.widget.set",
            {
                "id": "status",
                "scopeId": "conversation-a",
                "placement": "belowComposer",
                "frame": {"sequence": 2, "lines": ["updated"]},
            },
        ),
        (
            "kodelet.ui.widget.remove",
            {"id": "status", "sequence": 3, "scopeId": "conversation-a"},
        ),
        (
            "kodelet.ui.transcript.append",
            {
                "title": "Saved",
                "message": "./drawing.png",
                "scopeId": "conversation-a",
            },
        ),
        (
            "kodelet.ui.surface.open",
            {
                "id": "game",
                "scopeId": "conversation-a",
                "options": {"width": "75%", "maxHeight": "95%", "anchor": "center"},
                "frame": {"sequence": 1, "lines": ["loading"]},
            },
        ),
    ]

    assert opened_surface is not None
    for handler in list(notification_handlers):
        handler(
            "extension.ui.surface.unknown",
            {"id": "game", "scopeId": "conversation-a", "sequence": 100},
        )
        handler(
            "extension.ui.surface.resize",
            {
                "id": "game",
                "scopeId": "conversation-a",
                "sequence": 99,
                "width": "invalid",
                "height": 1,
            },
        )
        handler(
            "extension.ui.surface.resize",
            {
                "id": "game",
                "scopeId": "conversation-a",
                "sequence": 1,
                "width": 80,
                "height": 20,
            },
        )
        handler(
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
        handler(
            "extension.ui.surface.resize",
            {
                "id": "game",
                "scopeId": "conversation-a",
                "sequence": 1,
                "width": 1,
                "height": 1,
            },
        )

    assert opened_surface.size == {"width": 80, "height": 20}
    assert resize_events == [
        {
            "sequence": 1,
            "scopeId": "conversation-a",
            "width": 80,
            "height": 20,
        }
    ]
    assert input_events == [
        {
            "id": "game",
            "scopeId": "conversation-a",
            "sequence": 2,
            "kind": "key",
            "key": "q",
            "text": "q",
        }
    ]
    await opened_surface.close()
    assert requests[-1] == (
        "kodelet.ui.surface.close",
        {"id": "game", "sequence": 2, "scopeId": "conversation-a"},
    )


@pytest.mark.asyncio
async def test_same_surface_id_isolated_by_ui_scope_and_routes_scoped_events() -> None:
    surfaces: dict[str, UISurface] = {}
    input_events: dict[str, list[UISurfaceInputEvent]] = {
        "conversation-a": [],
        "conversation-b": [],
    }
    resize_events: dict[str, list[UISurfaceResizeEvent]] = {
        "conversation-a": [],
        "conversation-b": [],
    }
    requests: list[tuple[str, Any]] = []
    notification_handlers: set[Callable[[str, Any], None]] = set()

    class FakeRPC:
        async def request(self, method: str, params: Any | None = None) -> Any:
            requests.append((method, params))
            return {"accepted": True}

        def on_notification(
            self,
            handler: Callable[[str, Any], None],
        ) -> Callable[[], None]:
            notification_handlers.add(handler)
            return lambda: notification_handlers.discard(handler)

    ext = Extension()

    @ext.command("open", description="Open one scoped surface")
    async def open_surface(_input: Any, ctx: CommandContext) -> CommandResult:
        scope_id = ctx.ui_scope_id
        assert scope_id is not None
        surface = await ctx.ui.open_surface({"id": "shared"})
        surfaces[scope_id] = surface
        surface.on_input(input_events[scope_id].append)
        surface.on_resize(resize_events[scope_id].append)
        return {"action": "respond", "response": scope_id}

    harness = await create_test_harness(ext, FakeRPC())
    harness.initialize({"capabilities": {"ui": {"surfaces": True}}})
    for scope_id in ("conversation-a", "conversation-b"):
        await harness.execute_command(
            {
                "name": "open",
                "context": {"uiScopeId": scope_id},
                "invocation": {
                    "raw": "/open",
                    "commandName": "open",
                    "args": [],
                    "flags": {},
                },
            }
        )

    assert requests == [
        (
            "kodelet.ui.surface.open",
            {
                "id": "shared",
                "scopeId": "conversation-a",
                "options": {},
                "frame": {"sequence": 1, "lines": []},
            },
        ),
        (
            "kodelet.ui.surface.open",
            {
                "id": "shared",
                "scopeId": "conversation-b",
                "options": {},
                "frame": {"sequence": 1, "lines": []},
            },
        ),
    ]

    for handler in list(notification_handlers):
        handler(
            "extension.ui.surface.input",
            {
                "id": "shared",
                "scopeId": "conversation-a",
                "sequence": 1,
                "kind": "key",
                "key": "a",
            },
        )
        handler(
            "extension.ui.surface.resize",
            {
                "id": "shared",
                "scopeId": "conversation-b",
                "sequence": 1,
                "width": 80,
                "height": 20,
            },
        )
        handler(
            "extension.ui.surface.input",
            {
                "id": "shared",
                "scopeId": "conversation-c",
                "sequence": 1,
                "kind": "key",
                "key": "ignored",
            },
        )

    assert input_events == {
        "conversation-a": [
            {
                "id": "shared",
                "scopeId": "conversation-a",
                "sequence": 1,
                "kind": "key",
                "key": "a",
            }
        ],
        "conversation-b": [],
    }
    assert resize_events == {
        "conversation-a": [],
        "conversation-b": [
            {
                "scopeId": "conversation-b",
                "sequence": 1,
                "width": 80,
                "height": 20,
            }
        ],
    }

    await surfaces["conversation-a"].close()
    await surfaces["conversation-b"].close()
    assert requests[-2:] == [
        (
            "kodelet.ui.surface.close",
            {"id": "shared", "sequence": 2, "scopeId": "conversation-a"},
        ),
        (
            "kodelet.ui.surface.close",
            {"id": "shared", "sequence": 2, "scopeId": "conversation-b"},
        ),
    ]


@pytest.mark.asyncio
async def test_surface_ids_remain_exclusive_until_close_finishes() -> None:
    opened_surface: UISurface | None = None
    requests: list[tuple[str, Any]] = []
    loop = asyncio.get_running_loop()
    first_open_response: asyncio.Future[dict[str, bool]] = loop.create_future()
    first_close_response: asyncio.Future[dict[str, bool]] = loop.create_future()
    open_requests = 0
    close_requests = 0

    class FakeRPC:
        async def request(self, method: str, params: Any | None = None) -> Any:
            nonlocal open_requests, close_requests
            requests.append((method, params))
            if method == "kodelet.ui.surface.open":
                open_requests += 1
                if open_requests == 1:
                    return await first_open_response
            if method == "kodelet.ui.surface.close":
                close_requests += 1
                if close_requests == 1:
                    return await first_close_response
            return {"accepted": True}

    ext = Extension()

    @ext.command("exclusive", description="Open one surface at a time")
    async def exclusive(_input: Any, ctx: CommandContext) -> CommandResult:
        nonlocal opened_surface
        first_open = asyncio.create_task(ctx.ui.open_surface({"id": "singleton"}))
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="already open, opening, or closing"):
            await ctx.ui.open_surface({"id": "singleton"})
        first_open_response.set_result({"accepted": True})
        opened_surface = await first_open
        with pytest.raises(RuntimeError, match="already open, opening, or closing"):
            await ctx.ui.open_surface({"id": "singleton"})
        first_close = asyncio.create_task(opened_surface.close())
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="already open, opening, or closing"):
            await ctx.ui.open_surface({"id": "singleton"})
        first_close_response.set_result({"accepted": True})
        await first_close
        opened_surface = await ctx.ui.open_surface({"id": "singleton"})
        return {"action": "respond", "response": "opened"}

    harness = await create_test_harness(ext, FakeRPC())
    harness.initialize({"capabilities": {"ui": {"surfaces": True}}})
    await harness.execute_command(
        {
            "name": "exclusive",
            "invocation": {
                "raw": "/exclusive",
                "commandName": "exclusive",
                "args": [],
                "flags": {},
            },
        }
    )

    assert [method for method, _ in requests] == [
        "kodelet.ui.surface.open",
        "kodelet.ui.surface.close",
        "kodelet.ui.surface.open",
    ]
    assert [
        params.get("sequence", params.get("frame", {}).get("sequence")) for _, params in requests
    ] == [1, 2, 3]
    assert opened_surface is not None
    await opened_surface.close()
    assert requests[-1] == (
        "kodelet.ui.surface.close",
        {"id": "singleton", "sequence": 4, "scopeId": ""},
    )


@pytest.mark.asyncio
async def test_failed_surface_close_keeps_ownership_and_can_be_retried() -> None:
    requests: list[tuple[str, Any]] = []
    close_attempts = 0

    class FakeRPC:
        async def request(self, method: str, params: Any | None = None) -> Any:
            nonlocal close_attempts
            requests.append((method, params))
            if method == "kodelet.ui.surface.close":
                close_attempts += 1
                if close_attempts == 1:
                    raise RuntimeError("close failed")
            return {"accepted": True}

    ext = Extension()

    @ext.command("retry-close", description="Retry a failed surface close")
    async def retry_close(_input: Any, ctx: CommandContext) -> CommandResult:
        surface = await ctx.ui.open_surface({"id": "retryable"})
        with pytest.raises(RuntimeError, match="close failed"):
            await surface.close()
        with pytest.raises(RuntimeError, match="already open, opening, or closing"):
            await ctx.ui.open_surface({"id": "retryable"})
        await surface.close()
        replacement = await ctx.ui.open_surface({"id": "retryable"})
        await replacement.close()
        return {"action": "respond", "response": "closed"}

    harness = await create_test_harness(ext, FakeRPC())
    harness.initialize({"capabilities": {"ui": {"surfaces": True}}})
    await harness.execute_command(
        {
            "name": "retry-close",
            "invocation": {
                "raw": "/retry-close",
                "commandName": "retry-close",
                "args": [],
                "flags": {},
            },
        }
    )

    assert [method for method, _ in requests] == [
        "kodelet.ui.surface.open",
        "kodelet.ui.surface.close",
        "kodelet.ui.surface.close",
        "kodelet.ui.surface.open",
        "kodelet.ui.surface.close",
    ]
    assert [
        params.get("sequence", params.get("frame", {}).get("sequence")) for _, params in requests
    ] == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_surface_update_during_pending_close_does_not_overtake_close() -> None:
    requests: list[tuple[str, Any]] = []
    notifications: list[tuple[str, Any]] = []
    close_response: asyncio.Future[dict[str, bool]] = asyncio.get_running_loop().create_future()
    close_started = asyncio.Event()

    class FakeRPC:
        async def request(self, method: str, params: Any | None = None) -> Any:
            requests.append((method, params))
            if method == "kodelet.ui.surface.close":
                close_started.set()
                return await close_response
            return {"accepted": True}

        async def notify(self, method: str, params: Any | None = None) -> None:
            notifications.append((method, params))

    opened_surface: UISurface | None = None
    ext = Extension()

    @ext.command("close-order", description="Keep close ordering stable")
    async def close_order(_input: Any, ctx: CommandContext) -> CommandResult:
        nonlocal opened_surface
        opened_surface = await ctx.ui.open_surface({"id": "ordered"})
        return {"action": "respond", "response": "opened"}

    harness = await create_test_harness(ext, FakeRPC())
    harness.initialize({"capabilities": {"ui": {"surfaces": True}}})
    await harness.execute_command(
        {
            "name": "close-order",
            "invocation": {
                "raw": "/close-order",
                "commandName": "close-order",
                "args": [],
                "flags": {},
            },
        }
    )

    assert opened_surface is not None
    close_task = asyncio.create_task(opened_surface.close())
    await close_started.wait()
    opened_surface.update(["must not overtake close"])
    await _settle_event_loop()

    assert notifications == []
    assert requests[-1] == (
        "kodelet.ui.surface.close",
        {"id": "ordered", "sequence": 2, "scopeId": ""},
    )

    close_response.set_result({"accepted": True})
    await close_task
    await _settle_event_loop()
    assert notifications == []


@pytest.mark.asyncio
async def test_surface_frames_keep_one_write_in_flight_and_latest_pending() -> None:
    opened_surface: UISurface | None = None
    notifications: list[tuple[str, Any]] = []
    release_notifications: list[asyncio.Future[None]] = []

    class FakeRPC:
        async def request(self, method: str, params: Any | None = None) -> Any:
            del method, params
            return {"accepted": True}

        async def notify(self, method: str, params: Any | None = None) -> None:
            notifications.append((method, params))
            release = asyncio.get_running_loop().create_future()
            release_notifications.append(release)
            await release

    ext = Extension()

    @ext.command("bounded", description="Open a bounded surface")
    async def bounded(_input: Any, ctx: CommandContext) -> CommandResult:
        nonlocal opened_surface
        opened_surface = await ctx.ui.open_surface({"id": "bounded"})
        return {"action": "respond", "response": "opened"}

    harness = await create_test_harness(ext, FakeRPC())
    harness.initialize({"capabilities": {"ui": {"surfaces": True}}})
    await harness.execute_command(
        {
            "name": "bounded",
            "invocation": {
                "raw": "/bounded",
                "commandName": "bounded",
                "args": [],
                "flags": {},
            },
        }
    )

    assert opened_surface is not None
    opened_surface.update(["frame 1"])
    await _settle_event_loop()
    assert len(notifications) == 1

    opened_surface.update(["frame 2"])
    opened_surface.update(["frame 3"])
    await _settle_event_loop()
    assert len(notifications) == 1

    release_notifications.pop(0).set_result(None)
    await _settle_event_loop()
    assert notifications == [
        (
            "kodelet.ui.surface.frame",
            {
                "id": "bounded",
                "frame": {"sequence": 2, "lines": ["frame 1"]},
                "scopeId": "",
            },
        ),
        (
            "kodelet.ui.surface.frame",
            {
                "id": "bounded",
                "frame": {"sequence": 3, "lines": ["frame 3"]},
                "scopeId": "",
            },
        ),
    ]

    release_notifications.pop(0).set_result(None)
    await _settle_event_loop()
    await opened_surface.close()


@pytest.mark.asyncio
async def test_surface_routing_retains_only_initial_resize_and_focus_events() -> None:
    opened_surface: UISurface | None = None
    input_events: list[UISurfaceInputEvent] = []
    resize_events: list[UISurfaceResizeEvent] = []
    notification_handlers: set[Callable[[str, Any], None]] = set()
    unsubscribers: list[Callable[[], None]] = []

    class FakeRPC:
        async def request(self, method: str, params: Any | None = None) -> Any:
            del params
            if method == "kodelet.ui.surface.open":
                for handler in list(notification_handlers):
                    handler(
                        "extension.ui.surface.resize",
                        {"id": "early", "sequence": 1, "width": 72, "height": 18},
                    )
                    handler(
                        "extension.ui.surface.input",
                        {
                            "id": "early",
                            "sequence": 2,
                            "kind": "key",
                            "key": "x",
                            "text": "x",
                        },
                    )
                    handler(
                        "extension.ui.surface.input",
                        {"id": "early", "sequence": 3, "kind": "focus"},
                    )
                    handler(
                        "extension.ui.surface.input",
                        {
                            "id": "early",
                            "sequence": 4,
                            "kind": "mouse",
                            "mouse": {
                                "x": 1,
                                "y": 1,
                                "button": "none",
                                "action": "motion",
                            },
                        },
                    )
            return {"accepted": True}

        def on_notification(
            self,
            handler: Callable[[str, Any], None],
        ) -> Callable[[], None]:
            notification_handlers.add(handler)
            return lambda: notification_handlers.discard(handler)

    ext = Extension()

    @ext.command("early", description="Open a surface with early events")
    async def early(_input: Any, ctx: CommandContext) -> CommandResult:
        nonlocal opened_surface
        opened_surface = await ctx.ui.open_surface({"id": "early"})
        unsubscribers.append(opened_surface.on_resize(resize_events.append))
        unsubscribers.append(opened_surface.on_input(input_events.append))
        return {"action": "respond", "response": "opened"}

    harness = await create_test_harness(ext, FakeRPC())
    harness.initialize({"capabilities": {"ui": {"surfaces": True}}})
    await harness.execute_command(
        {
            "name": "early",
            "invocation": {"raw": "/early", "commandName": "early", "args": [], "flags": {}},
        }
    )

    assert opened_surface is not None
    assert opened_surface.size == {"width": 72, "height": 18}
    assert resize_events == [{"sequence": 1, "width": 72, "height": 18}]
    assert input_events == [{"id": "early", "sequence": 3, "kind": "focus"}]
    for unsubscribe in unsubscribers:
        unsubscribe()
    for handler in list(notification_handlers):
        handler(
            "extension.ui.surface.input",
            {"id": "early", "sequence": 5, "kind": "key", "key": "x", "text": "x"},
        )
        handler(
            "extension.ui.surface.input",
            {"id": "early", "sequence": 6, "kind": "blur"},
        )
        handler(
            "extension.ui.surface.resize",
            {"id": "early", "sequence": 7, "width": 73, "height": 19},
        )
    replayed_input: list[UISurfaceInputEvent] = []
    replayed_resize: list[UISurfaceResizeEvent] = []
    opened_surface.on_input(replayed_input.append)
    opened_surface.on_resize(replayed_resize.append)
    assert replayed_input == []
    assert replayed_resize == []
    await opened_surface.close()


@pytest.mark.asyncio
async def test_persistent_ui_ids_validate_before_routing_or_sequence_allocation() -> None:
    requests: list[tuple[str, Any]] = []

    class FakeRPC:
        async def request(self, method: str, params: Any | None = None) -> Any:
            requests.append((method, params))
            return {"accepted": True}

    ext = Extension()

    @ext.command("validated", description="Validate UI object identifiers")
    async def validated(_input: Any, ctx: CommandContext) -> CommandResult:
        with pytest.raises(ValueError, match="id is required"):
            await ctx.ui.set_widget("   ", ["invalid"])
        with pytest.raises(ValueError, match="leading or trailing whitespace"):
            await ctx.ui.set_widget(" status ", ["invalid"])
        with pytest.raises(ValueError, match="id is too long"):
            await ctx.ui.set_widget("é" * 65, ["invalid"])
        await ctx.ui.set_widget("status", ["valid"])
        with pytest.raises(ValueError, match="leading or trailing whitespace"):
            await ctx.ui.open_surface({"id": " game "})
        surface = await ctx.ui.open_surface({"id": "game"})
        await surface.close()
        return {"action": "respond", "response": "validated"}

    harness = await create_test_harness(ext, FakeRPC())
    harness.initialize({"capabilities": {"ui": {"widgets": True, "surfaces": True}}})
    await harness.execute_command(
        {
            "name": "validated",
            "invocation": {
                "raw": "/validated",
                "commandName": "validated",
                "args": [],
                "flags": {},
            },
        }
    )

    assert [method for method, _ in requests] == [
        "kodelet.ui.widget.set",
        "kodelet.ui.surface.open",
        "kodelet.ui.surface.close",
    ]
    assert [
        params.get("sequence", params.get("frame", {}).get("sequence")) for _, params in requests
    ] == [1, 1, 2]


@pytest.mark.asyncio
async def test_persistent_ui_frame_collections_require_lists() -> None:
    requests: list[tuple[str, Any]] = []

    class FakeRPC:
        async def request(self, method: str, params: Any | None = None) -> Any:
            requests.append((method, params))
            return {"accepted": True}

    ext = Extension()

    @ext.command("frames", description="Validate frame collections")
    async def frames(_input: Any, ctx: CommandContext) -> CommandResult:
        with pytest.raises(TypeError, match="frame lines must be a list"):
            await ctx.ui.set_widget("status", cast(Any, "ready"))
        await ctx.ui.set_widget("status", ["ready"])
        with pytest.raises(TypeError, match="frame lines must be a list"):
            await ctx.ui.open_surface({"id": "game", "initialLines": cast(Any, "loading")})
        surface = await ctx.ui.open_surface({"id": "game"})
        with pytest.raises(TypeError, match="frame lines must be a list"):
            surface.update(cast(Any, "updated"))
        await surface.close()
        return {"action": "respond", "response": "validated"}

    harness = await create_test_harness(ext, FakeRPC())
    harness.initialize({"capabilities": {"ui": {"widgets": True, "surfaces": True}}})
    await harness.execute_command(
        {
            "name": "frames",
            "invocation": {
                "raw": "/frames",
                "commandName": "frames",
                "args": [],
                "flags": {},
            },
        }
    )

    assert [method for method, _ in requests] == [
        "kodelet.ui.widget.set",
        "kodelet.ui.surface.open",
        "kodelet.ui.surface.close",
    ]
    assert [
        params.get("sequence", params.get("frame", {}).get("sequence")) for _, params in requests
    ] == [1, 1, 2]


@pytest.mark.asyncio
async def test_rejected_surface_open_releases_ownership_and_preserves_sequence() -> None:
    requests: list[tuple[str, Any]] = []
    open_count = 0

    class FakeRPC:
        async def request(self, method: str, params: Any | None = None) -> Any:
            nonlocal open_count
            requests.append((method, params))
            if method == "kodelet.ui.surface.open":
                open_count += 1
                if open_count == 1:
                    return {"accepted": False, "reason": "denied"}
            return {"accepted": True}

    ext = Extension()

    @ext.command("retry", description="Retry a rejected surface")
    async def retry(_input: Any, ctx: CommandContext) -> CommandResult:
        with pytest.raises(RuntimeError, match="denied"):
            await ctx.ui.open_surface({"id": "game"})
        surface = await ctx.ui.open_surface({"id": "game"})
        await surface.close()
        return {"action": "respond", "response": "opened"}

    harness = await create_test_harness(ext, FakeRPC())
    harness.initialize({"capabilities": {"ui": {"surfaces": True}}})
    await harness.execute_command(
        {
            "name": "retry",
            "invocation": {"raw": "/retry", "commandName": "retry", "args": [], "flags": {}},
        }
    )

    assert [method for method, _ in requests] == [
        "kodelet.ui.surface.open",
        "kodelet.ui.surface.open",
        "kodelet.ui.surface.close",
    ]
    assert [
        params.get("sequence", params.get("frame", {}).get("sequence")) for _, params in requests
    ] == [1, 2, 3]


@pytest.mark.asyncio
async def test_persistent_ui_apis_are_capability_gated() -> None:
    requests: list[str] = []

    class FakeRPC:
        async def request(self, method: str, params: Any | None = None) -> Any:
            del params
            requests.append(method)
            return {}

    ext = Extension()

    @ext.command("ui", description="Try persistent UI")
    async def ui(_input: Any, ctx: CommandContext) -> CommandResult:
        await ctx.ui.set_widget("status", ["ignored"])
        await ctx.ui.append_transcript("ignored")
        with pytest.raises(RuntimeError, match="not available"):
            await ctx.ui.open_surface({"id": "missing"})
        return {"action": "respond", "response": "done"}

    harness = await create_test_harness(ext, FakeRPC())
    harness.initialize({"capabilities": {}})
    await harness.execute_command(
        {
            "name": "ui",
            "invocation": {"raw": "/ui", "commandName": "ui", "args": [], "flags": {}},
        }
    )
    assert requests == []


@pytest.mark.asyncio
async def test_test_harness_host_clients_are_isolated() -> None:
    class FakeRPC:
        def __init__(self) -> None:
            self.messages: list[str] = []

        async def request(self, method: str, params: Any | None = None) -> Any:
            del method
            assert isinstance(params, Mapping)
            self.messages.append(params["message"])
            return {"status": "submitted"}

    ext = Extension()

    @ext.command("notify", description="Notify through the active harness")
    async def notify(input: Any, ctx: CommandContext) -> CommandResult:
        await ctx.ui.notify(input["message"])
        return {"action": "respond", "response": "done"}

    first_rpc = FakeRPC()
    second_rpc = FakeRPC()
    first = await create_test_harness(ext, first_rpc)
    second = await create_test_harness(ext, second_rpc)
    first.initialize()
    second.initialize()

    await first.execute_command(
        {
            "name": "notify",
            "input": {"message": "first"},
            "invocation": {"raw": "/notify", "commandName": "notify", "args": [], "flags": {}},
        }
    )
    await second.execute_command(
        {
            "name": "notify",
            "input": {"message": "second"},
            "invocation": {"raw": "/notify", "commandName": "notify", "args": [], "flags": {}},
        }
    )

    assert first_rpc.messages == ["first"]
    assert second_rpc.messages == ["second"]


@pytest.mark.asyncio
async def test_harness_none_and_falsey_clients_override_the_global_client() -> None:
    class FakeRPC:
        def __init__(self, *, truthy: bool = True) -> None:
            self.truthy = truthy
            self.messages: list[str] = []

        def __bool__(self) -> bool:
            return self.truthy

        async def request(self, method: str, params: Any | None = None) -> Any:
            del method
            assert isinstance(params, Mapping)
            self.messages.append(params["message"])
            return {"status": "submitted"}

    ext = Extension()

    @ext.command("notify", description="Route to the scoped client")
    async def notify(input: Any, ctx: CommandContext) -> CommandResult:
        await ctx.ui.notify(input["message"])
        return {"action": "respond", "response": "done"}

    global_rpc = FakeRPC()
    falsey_rpc = FakeRPC(truthy=False)
    set_active_host_rpc_client(global_rpc)
    try:
        await UIContext().notify("global")
        clientless = await create_test_harness(ext)
        falsey = await create_test_harness(ext, falsey_rpc)
        clientless.initialize()
        falsey.initialize()
        await clientless.execute_command(
            {
                "name": "notify",
                "input": {"message": "ignored"},
                "invocation": {
                    "raw": "/notify",
                    "commandName": "notify",
                    "args": [],
                    "flags": {},
                },
            }
        )
        await falsey.execute_command(
            {
                "name": "notify",
                "input": {"message": "local"},
                "invocation": {
                    "raw": "/notify",
                    "commandName": "notify",
                    "args": [],
                    "flags": {},
                },
            }
        )
    finally:
        set_active_host_rpc_client(None)

    assert global_rpc.messages == ["global"]
    assert falsey_rpc.messages == ["local"]


@pytest.mark.asyncio
async def test_persistent_ui_state_uses_client_identity_for_equal_unhashable_clients() -> None:
    @dataclass(eq=True)
    class EqualRPC:
        name: str
        requests: list[tuple[str, Any]] = field(default_factory=list, compare=False)

        async def request(self, method: str, params: Any | None = None) -> Any:
            self.requests.append((method, params))
            return {"accepted": True}

    ext = Extension()

    @ext.command("widget", description="Set one widget")
    async def widget(_input: Any, ctx: CommandContext) -> CommandResult:
        await ctx.ui.set_widget("status", ["ready"])
        return {"action": "respond", "response": "done"}

    first_rpc = EqualRPC("same")
    second_rpc = EqualRPC("same")
    assert first_rpc == second_rpc
    first = await create_test_harness(ext, first_rpc)
    second = await create_test_harness(ext, second_rpc)
    capabilities = {"capabilities": {"ui": {"widgets": True}}}
    first.initialize(capabilities)
    second.initialize(capabilities)
    invocation = {
        "name": "widget",
        "invocation": {
            "raw": "/widget",
            "commandName": "widget",
            "args": [],
            "flags": {},
        },
    }
    await first.execute_command(invocation)
    await second.execute_command(invocation)

    assert first_rpc.requests[0][1]["frame"]["sequence"] == 1
    assert second_rpc.requests[0][1]["frame"]["sequence"] == 1


@pytest.mark.asyncio
async def test_persistent_surface_state_does_not_keep_fallback_client_alive() -> None:
    @dataclass(frozen=True)
    class FrozenRPC:
        name: str

        __hash__ = None

        async def request(self, method: str, params: Any | None = None) -> Any:
            del method, params
            return {"accepted": True}

    client = FrozenRPC("client")
    client_ref = weakref.ref(client)
    ui = UIContext(
        {"capabilities": {"ui": {"surfaces": True}}},
        client,
    )
    surface = await ui.open_surface({"id": "game"})
    del surface
    del ui
    del client
    gc.collect()

    assert client_ref() is None


async def _settle_event_loop() -> None:
    for _ in range(3):
        await asyncio.sleep(0)

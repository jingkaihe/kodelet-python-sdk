from __future__ import annotations

import asyncio
import gc
import json
import os
import queue
import sys
import weakref
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import CoroutineType
from typing import TYPE_CHECKING, Any, assert_type, cast

import pytest

from kodelet_sdk import (
    BaseModel,
    CommandContext,
    CommandResult,
    EventContext,
    EventName,
    EventResult,
    Extension,
    Field,
    Jinja2,
    JSONSchema,
    Pydantic,
    ToolCallEvent,
    ToolContext,
    ToolExecutionResult,
    ToolInputSchema,
    ToolUpdateEvent,
    UIConfirmRequest,
    UIContext,
    UIFrameLine,
    UIInputRequest,
    UIMargin,
    UINotifyRequest,
    UISelectRequest,
    UIStyle,
    UISurface,
    UISurfaceInputEvent,
    UISurfaceOpenOptions,
    UISurfaceResizeEvent,
    UITranscriptAppendRequest,
    UIWidgetPlacement,
    create_test_harness,
    define_extension,
    pydantic,
    render_template,
    set_active_host_rpc_client,
)
from kodelet_sdk.runtime import StdioHostRPCClient, _StdioRequestState, run_stdio_server


class WeatherInput(BaseModel):
    location: str


def test_reexports_pydantic_and_jinja2() -> None:
    class Model(Pydantic.BaseModel):
        name: str = Pydantic.Field(min_length=1)

    assert pydantic is Pydantic
    assert Model(name="kodelet").name == "kodelet"
    assert Jinja2.Template("Hello {{ name }}").render(name="Kodelet") == "Hello Kodelet"


def test_public_typing_surface() -> None:
    ext = Extension()
    update_event_name: EventName = "tool.update"
    assert update_event_name == "tool.update"
    style: UIStyle = {"foreground": "#00ff00", "bold": True}
    frame_line: UIFrameLine = {"spans": [{"text": "ready", "style": style}]}
    margin: UIMargin = {"top": 1, "bottom": 1}
    surface_options: UISurfaceOpenOptions = {
        "id": "status",
        "initialLines": [frame_line],
        "width": "75%",
        "anchor": "center",
        "margin": margin,
    }
    transcript: UITranscriptAppendRequest = {"title": "Saved", "message": "drawing.png"}
    placement: UIWidgetPlacement = "belowComposer"
    assert surface_options["id"] == "status"
    assert transcript["message"] == "drawing.png"
    assert placement == "belowComposer"

    class EchoInput(BaseModel):
        text: str

    @ext.tool("echo", description="Echo", input_schema=EchoInput)
    async def echo(input: EchoInput, ctx: ToolContext) -> ToolExecutionResult:
        request: UIInputRequest = {"title": "Text", "required": True}
        answer = await ctx.ui.input(request)
        assert_type(answer, str | None)
        return {"content": input.text}

    echo_handler: Callable[[EchoInput, ToolContext], Awaitable[ToolExecutionResult]] = echo
    assert echo_handler is echo
    if TYPE_CHECKING:
        echo_result = echo(EchoInput(text="hello"), ToolContext(None))
        assert_type(echo_result, CoroutineType[Any, Any, ToolExecutionResult])

    @ext.command("ask", description="Ask", input_schema=EchoInput)
    async def ask(input: EchoInput, ctx: CommandContext) -> CommandResult:
        confirm: UIConfirmRequest = {"title": "Continue?"}
        select: UISelectRequest = {"title": "Pick", "options": ["A", "B"]}
        notify: UINotifyRequest = {"message": "Done"}
        confirmed = await ctx.ui.confirm(confirm)
        selection = await ctx.ui.select(select)
        await ctx.ui.notify(notify)
        assert_type(confirmed, bool)
        assert_type(selection, str | None)
        return {"action": "respond", "response": f"{ctx.input['commandName']}: {input.text}"}

    ask_handler: Callable[[EchoInput, CommandContext], Awaitable[CommandResult]] = ask
    assert ask_handler is ask
    if TYPE_CHECKING:
        ask_result = ask(
            EchoInput(text="hello"),
            CommandContext(
                None,
                None,
                {"raw": "/ask", "commandName": "ask", "args": [], "flags": {}},
            ),
        )
        assert_type(ask_result, CoroutineType[Any, Any, CommandResult])

    @ext.on("tool.call")
    def approve(event: ToolCallEvent, _ctx: EventContext) -> EventResult:
        assert_type(event.tool.name, str)
        return {"message": event.tool.name}

    approve_handler: Callable[[ToolCallEvent, EventContext], EventResult] = approve
    assert approve_handler is approve
    if TYPE_CHECKING:
        approve_result = approve(
            ToolCallEvent(
                {
                    "id": "evt",
                    "event": "tool.call",
                    "tool": {"name": "bash", "callId": "call", "input": {}},
                }
            ),
            EventContext(None),
        )
        assert_type(approve_result, EventResult)

    @ext.on("tool.update")
    def sanitize_update(event: ToolUpdateEvent, _ctx: EventContext) -> EventResult:
        assert_type(event.tool.name, str)
        assert_type(event.tool.output, Any)
        return {"output": event.tool.output}

    update_handler: Callable[[ToolUpdateEvent, EventContext], EventResult] = sanitize_update
    assert update_handler is sanitize_update
    if TYPE_CHECKING:
        update_result = sanitize_update(
            ToolUpdateEvent(
                {
                    "id": "evt",
                    "event": "tool.update",
                    "tool": {
                        "name": "bash",
                        "callId": "call",
                        "input": {},
                        "output": {"content": "partial"},
                    },
                }
            ),
            EventContext(None),
        )
        assert_type(update_result, EventResult)


@pytest.mark.asyncio
async def test_registers_tools_commands_events_and_executes_handlers() -> None:
    async def entrypoint(ext: Extension) -> None:
        ext.set_metadata(name="weather", version="0.1.0")

        @ext.tool(
            "get_weather",
            description="Get weather",
            input_schema=WeatherInput,
            timeout_in_sec=600,
        )
        async def get_weather(input: WeatherInput, _ctx: Any) -> dict[str, Any]:
            return {
                "content": f"Weather for {input.location}",
                "data": {"location": input.location},
            }

        class DoctorInput(BaseModel):
            verbose: bool = False

        @ext.command(
            "doctor",
            aliases=["/doctor"],
            description="Inspect extension health",
            input_schema=DoctorInput,
            timeout_in_sec=30,
        )
        async def doctor(input: DoctorInput, ctx: Any) -> dict[str, str]:
            return {
                "action": "respond",
                "response": f"{ctx.input['commandName']}: {'healthy' if input.verbose else 'ok'}",
            }

        @ext.on("tool.call", priority=10, timeout_in_sec=5)
        async def rewrite_weather(event: Any, _ctx: Any) -> dict[str, Any] | None:
            if event.tool.name == "get_weather":
                return {"input": {"location": "Paris"}}
            return None

        @ext.on("agent.end")
        def agent_end(_event: Any, _ctx: Any) -> dict[str, list[str]]:
            return {"followUpMessages": ["inspect tests"]}

    harness = await create_test_harness(define_extension(entrypoint))
    init = harness.initialize({"extension": {"id": "weather", "cwd": os.getcwd()}})

    assert init["name"] == "weather"
    assert init["version"] == "0.1.0"
    assert init["tools"][0]["name"] == "get_weather"
    assert init["tools"][0]["timeoutInSec"] == 600
    assert init["tools"][0]["inputSchema"]["type"] == "object"
    assert init["commands"][0]["name"] == "doctor"
    assert init["commands"][0]["timeoutInSec"] == 30
    assert init["subscriptions"] == [
        {"event": "tool.call", "priority": 10, "timeoutInSec": 5},
        {"event": "agent.end", "priority": 0},
    ]

    tool_result = await harness.execute_tool(
        {"name": "get_weather", "input": {"location": "London"}}
    )
    assert tool_result == {"content": "Weather for London", "data": {"location": "London"}}

    command_result = await harness.execute_command(
        {
            "name": "/doctor",
            "input": {"verbose": True},
            "invocation": {
                "raw": "/doctor verbose=true",
                "commandName": "doctor",
                "args": ["verbose=true"],
                "flags": {"verbose": "true"},
            },
        }
    )
    assert command_result == {"action": "respond", "response": "doctor: healthy"}

    event_result = await harness.handle_event(
        {
            "id": "evt_1",
            "event": "tool.call",
            "payload": {"tool": {"name": "get_weather", "input": {"location": "London"}}},
        }
    )
    assert event_result == {"input": {"location": "Paris"}}

    agent_end_result = await harness.handle_event(
        {
            "id": "evt_2",
            "event": "agent.end",
            "payload": {"messages": [{"role": "assistant", "content": "done"}]},
        }
    )
    assert agent_end_result == {"followUpMessages": ["inspect tests"]}


@pytest.mark.asyncio
async def test_raw_json_schema_is_preserved_and_input_is_passed_through() -> None:
    ext = Extension()
    raw_schema: JSONSchema = {
        "type": "object",
        "description": "Raw JSON Schema",
        "properties": {
            "mode": {"type": "string", "enum": ["fast", "safe"]},
            "target": {"type": ["string", "null"]},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["mode"],
        "additionalProperties": False,
        "oneOf": [
            {"properties": {"mode": {"const": "fast"}}},
            {"properties": {"mode": {"const": "safe"}}},
        ],
        "x-kodelet-test": {"preserved": True},
    }
    tool_schema: ToolInputSchema = raw_schema
    received: list[Any] = []

    @ext.tool("raw_schema", description="Raw schema", input_schema=tool_schema)
    async def raw_schema_tool(input: Any, _ctx: ToolContext) -> str:
        received.append(input)
        return json.dumps(input, sort_keys=True)

    harness = await create_test_harness(ext)
    init = harness.initialize()
    assert init["tools"][0]["inputSchema"] == raw_schema

    unconstrained = {"mode": "not-an-enum-value", "extra": True}
    result = await harness.execute_tool({"name": "raw_schema", "input": unconstrained})
    assert result == {"content": json.dumps(unconstrained, sort_keys=True)}
    null_result = await harness.execute_tool({"name": "raw_schema", "input": None})
    assert null_result == {"content": "null"}
    assert received == [unconstrained, None]


@pytest.mark.asyncio
async def test_command_validation_can_pass_to_next_route() -> None:
    ext = Extension()

    class ReviewInput(BaseModel):
        target: str

    @ext.command("review", description="Review code", input_schema=ReviewInput)
    async def review(input: ReviewInput, _ctx: Any) -> dict[str, str]:
        return {"action": "runAgent", "prompt": f"Review {input.target}"}

    harness = await create_test_harness(ext)
    result = await harness.execute_command(
        {
            "name": "review",
            "input": {},
            "invocation": {"raw": "/review", "commandName": "review", "args": [], "flags": {}},
        }
    )
    assert result == {"action": "pass"}


@pytest.mark.asyncio
async def test_timeout_merging_preserves_zero() -> None:
    ext = Extension()

    @ext.tool("forever_tool", description="Tool with no timeout", input_schema={}, timeout_in_sec=0)
    def forever_tool(_input: Any, _ctx: Any) -> str:
        return "ok"

    @ext.command("forever_command", description="Command with no timeout", timeout_in_sec=0)
    def forever_command(_input: Any, _ctx: Any) -> dict[str, str]:
        return {"action": "respond", "response": "ok"}

    @ext.on("tool.result", priority=1, timeout_in_sec=2)
    async def first(_event: Any, _ctx: Any) -> None:
        return None

    @ext.on("tool.result", priority=3, timeout_in_sec=0)
    async def second(_event: Any, _ctx: Any) -> None:
        return None

    @ext.on("tool.update", priority=2, timeout_in_sec=1)
    async def update(_event: Any, _ctx: Any) -> None:
        return None

    @ext.on("agent.end", timeout_in_sec=4)
    async def third(_event: Any, _ctx: Any) -> None:
        return None

    @ext.on("agent.end", timeout_in_sec=6)
    async def fourth(_event: Any, _ctx: Any) -> None:
        return None

    harness = await create_test_harness(ext)
    init = harness.initialize({"extension": {"id": "timeouts", "cwd": os.getcwd()}})

    assert init["tools"][0]["timeoutInSec"] == 0
    assert init["commands"][0]["timeoutInSec"] == 0
    assert sorted(init["subscriptions"], key=lambda item: item["event"]) == [
        {"event": "agent.end", "priority": 0, "timeoutInSec": 6},
        {"event": "tool.result", "priority": 3, "timeoutInSec": 0},
        {"event": "tool.update", "priority": 2, "timeoutInSec": 1},
    ]


@pytest.mark.asyncio
async def test_tool_update_handler_can_replace_accumulated_snapshot() -> None:
    ext = Extension()

    @ext.on("tool.update")
    async def sanitize(event: ToolUpdateEvent, _ctx: EventContext) -> EventResult:
        assert event.tool.name == "bash"
        assert event.tool.output == {"content": "secret output"}
        return {"output": {"content": "[redacted]"}}

    harness = await create_test_harness(ext)
    assert harness.initialize()["subscriptions"] == [
        {"event": "tool.update", "priority": 0}
    ]
    result = await harness.handle_event(
        {
            "id": "evt",
            "event": "tool.update",
            "payload": {
                "tool": {
                    "name": "bash",
                    "callId": "call-1",
                    "input": {"command": "echo secret"},
                    "output": {"content": "secret output"},
                }
            },
        }
    )
    assert result == {"output": {"content": "[redacted]"}}


@pytest.mark.asyncio
async def test_event_aggregation_patches_payload_and_stops_on_block() -> None:
    ext = Extension()

    @ext.on("tool.result", priority=10)
    async def rewrite_output(event: Any, _ctx: Any) -> dict[str, Any]:
        assert event.tool.output == "old"
        return {"output": "new", "tools": {"disable": ["bash"]}}

    @ext.on("tool.result", priority=5)
    async def observe_rewrite(event: Any, _ctx: Any) -> dict[str, Any]:
        assert event.tool.output == "new"
        return {"followUpMessages": ["done"], "tools": {"enable": ["echo"]}}

    @ext.on("tool.result", priority=1)
    async def block(_event: Any, _ctx: Any) -> dict[str, Any]:
        return {"block": {"reason": "nope"}, "message": "blocked"}

    @ext.on("tool.result", priority=0)
    async def skipped(_event: Any, _ctx: Any) -> dict[str, Any]:
        return {"message": "should not run"}

    harness = await create_test_harness(ext)
    result = await harness.handle_event(
        {
            "id": "evt",
            "event": "tool.result",
            "payload": {"tool": {"name": "bash", "input": {}, "output": "old"}},
        }
    )
    assert result == {
        "output": "new",
        "tools": {"disable": ["bash"], "enable": ["echo"]},
        "followUpMessages": ["done"],
        "message": "blocked",
        "block": {"reason": "nope"},
    }


@pytest.mark.asyncio
async def test_context_helpers_cover_workspace_storage_process_env_and_ui(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data_dir = tmp_path / "data"
    (workspace / "README.md").write_text("hello", encoding="utf-8")
    monkeypatch.setenv("KODELET_SDK_TEST", "ok")

    class FakeRPC:
        def __init__(self) -> None:
            self.requests: list[tuple[str, Any]] = []

        async def request(self, method: str, params: Any | None = None) -> Any:
            self.requests.append((method, params))
            if method == "kodelet.ui.confirm":
                return {"status": "submitted", "confirmed": True}
            if method == "kodelet.ui.select":
                return {"status": "submitted", "value": "Pizza"}
            if method == "kodelet.ui.notify":
                return {"status": "submitted"}
            return {"status": "submitted", "value": "2"}

    fake_rpc = FakeRPC()
    ext = Extension()

    class OpenInput(BaseModel):
        path: str | None = None

    @ext.command("open", description="Open a path", input_schema=OpenInput)
    async def open_command(input: OpenInput, ctx: Any) -> dict[str, str]:
        target = ctx.path.resolve_workspace_path(input.path or ".")
        exists = await ctx.fs.exists(target)
        await ctx.storage.write_json(
            "state.json",
            {"target": ctx.path.relative_to_workspace(target)},
        )
        exec_result = await ctx.process.exec(sys.executable, ["-c", "print('ok', end='')"])
        answer = await ctx.ui.input({"title": "Pick one"})
        confirmed = await ctx.ui.confirm({"title": "Allow?"})
        selection = await ctx.ui.select({"title": "Food", "options": ["Pasta", "Pizza"]})
        await ctx.ui.notify("Done")
        return {
            "action": "respond",
            "response": ":".join(
                [
                    str(exists).lower(),
                    ctx.path.relative_to_workspace(target),
                    exec_result.stdout,
                    ctx.env.get("KODELET_SDK_TEST") or "missing",
                    answer or "none",
                    str(confirmed).lower(),
                    selection or "none",
                ]
            ),
        }

    @ext.tool("stream", description="Stream progress", input_schema={})
    async def stream(_input: Any, ctx: ToolContext) -> str:
        await ctx.update("Searching code", {"step": 1})
        return "done"

    harness = await create_test_harness(ext, fake_rpc)
    harness.initialize(
        {"extension": {"id": "ctx", "cwd": str(workspace), "dataDir": str(data_dir)}}
    )
    result = await harness.execute_command(
        {
            "name": "open",
            "input": {"path": "README.md"},
            "context": {"cwd": str(workspace)},
            "invocation": {
                "raw": "/open README.md",
                "commandName": "open",
                "args": ["README.md"],
                "flags": {},
            },
        }
    )

    assert result == {"action": "respond", "response": "true:README.md:ok:ok:2:true:Pizza"}
    assert json.loads((data_dir / "state.json").read_text(encoding="utf-8")) == {
        "target": "README.md"
    }
    assert [method for method, _params in fake_rpc.requests] == [
        "kodelet.ui.input",
        "kodelet.ui.confirm",
        "kodelet.ui.select",
        "kodelet.ui.notify",
    ]
    assert [params for _method, params in fake_rpc.requests] == [
        {"title": "Pick one"},
        {"title": "Allow?"},
        {"title": "Food", "options": ["Pasta", "Pizza"]},
        {"message": "Done"},
    ]

    assert await harness.execute_tool({"name": "stream", "input": {}}) == {
        "content": "done"
    }
    assert fake_rpc.requests[-1] == (
        "kodelet.tool.update",
        {"content": "Searching code", "data": {"step": 1}},
    )


@pytest.mark.asyncio
async def test_tool_updates_are_ignored_without_host_capability() -> None:
    requests: list[str] = []

    class FakeRPC:
        async def request(self, method: str, params: Any | None = None) -> Any:
            del params
            requests.append(method)
            return {"accepted": True}

    ext = Extension()

    @ext.tool("stream", description="Stream progress", input_schema={})
    async def stream(_input: Any, ctx: ToolContext) -> str:
        await ctx.update("Working", {"step": 1})
        return "done"

    harness = await create_test_harness(ext, FakeRPC())
    harness.initialize({"capabilities": {}})
    assert await harness.execute_tool({"name": "stream", "input": {}}) == {
        "content": "done"
    }
    assert requests == []


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
            "invocation": {"raw": "/ui", "commandName": "ui", "args": [], "flags": {}},
        }
    )

    assert requests[:5] == [
        (
            "kodelet.ui.widget.set",
            {
                "id": "status",
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
                "placement": "belowComposer",
                "frame": {"sequence": 2, "lines": ["updated"]},
            },
        ),
        ("kodelet.ui.widget.remove", {"id": "status", "sequence": 3}),
        (
            "kodelet.ui.transcript.append",
            {"title": "Saved", "message": "./drawing.png"},
        ),
        (
            "kodelet.ui.surface.open",
            {
                "id": "game",
                "options": {"width": "75%", "maxHeight": "95%", "anchor": "center"},
                "frame": {"sequence": 1, "lines": ["loading"]},
            },
        ),
    ]

    assert opened_surface is not None
    for handler in list(notification_handlers):
        handler(
            "extension.ui.surface.unknown",
            {"id": "game", "sequence": 100},
        )
        handler(
            "extension.ui.surface.resize",
            {"id": "game", "sequence": 99, "width": "invalid", "height": 1},
        )
        handler(
            "extension.ui.surface.resize",
            {"id": "game", "sequence": 1, "width": 80, "height": 20},
        )
        handler(
            "extension.ui.surface.input",
            {"id": "game", "sequence": 2, "kind": "key", "key": "q", "text": "q"},
        )
        handler(
            "extension.ui.surface.resize",
            {"id": "game", "sequence": 1, "width": 1, "height": 1},
        )

    assert opened_surface.size == {"width": 80, "height": 20}
    assert resize_events == [{"sequence": 1, "width": 80, "height": 20}]
    assert input_events == [
        {"id": "game", "sequence": 2, "kind": "key", "key": "q", "text": "q"}
    ]
    await opened_surface.close()
    assert requests[-1] == (
        "kodelet.ui.surface.close",
        {"id": "game", "sequence": 2},
    )


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
        params.get("sequence", params.get("frame", {}).get("sequence"))
        for _, params in requests
    ] == [1, 2, 3]
    assert opened_surface is not None
    await opened_surface.close()
    assert requests[-1] == (
        "kodelet.ui.surface.close",
        {"id": "singleton", "sequence": 4},
    )


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
            {"id": "bounded", "frame": {"sequence": 2, "lines": ["frame 1"]}},
        ),
        (
            "kodelet.ui.surface.frame",
            {"id": "bounded", "frame": {"sequence": 3, "lines": ["frame 3"]}},
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
        params.get("sequence", params.get("frame", {}).get("sequence"))
        for _, params in requests
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
            await ctx.ui.open_surface(
                {"id": "game", "initialLines": cast(Any, "loading")}
            )
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
        params.get("sequence", params.get("frame", {}).get("sequence"))
        for _, params in requests
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
        params.get("sequence", params.get("frame", {}).get("sequence"))
        for _, params in requests
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
async def test_workspace_and_storage_paths_cannot_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data_dir = tmp_path / "data"
    ext = Extension()

    @ext.tool("escape", description="Try to escape", input_schema={})
    async def escape(_input: Any, ctx: Any) -> str:
        with pytest.raises(ValueError, match="Path escapes workspace"):
            ctx.path.resolve_workspace_path("../outside")
        with pytest.raises(ValueError, match="Path escapes extension storage"):
            await ctx.storage.write_text("../outside", "no")
        return "ok"

    harness = await create_test_harness(ext)
    harness.initialize(
        {"extension": {"id": "escape", "cwd": str(workspace), "dataDir": str(data_dir)}}
    )
    assert await harness.execute_tool({"name": "escape", "input": {}}) == {"content": "ok"}


def test_renders_jinja2_templates() -> None:
    assert (
        render_template(
            "Review {{ target }} with {{ focus }}",
            {"target": "main", "focus": "correctness"},
        )
        == "Review main with correctness"
    )
    with pytest.raises(Exception, match="missing"):
        render_template("{{ missing }}", {})


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

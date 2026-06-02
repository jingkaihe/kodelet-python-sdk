from __future__ import annotations

import asyncio
import json
import os
import queue
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, assert_type

import pytest

from kodelet_sdk import (
    BaseModel,
    CommandContext,
    CommandResult,
    EventContext,
    EventResult,
    Extension,
    Field,
    Jinja2,
    Pydantic,
    ToolCallEvent,
    ToolContext,
    ToolExecutionResult,
    UIConfirmRequest,
    UIInputRequest,
    UINotifyRequest,
    UISelectRequest,
    create_test_harness,
    define_extension,
    pydantic,
    render_template,
)
from kodelet_sdk.runtime import run_stdio_server


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

    @ext.on("tool.call")
    def approve(event: ToolCallEvent, _ctx: EventContext) -> EventResult:
        assert_type(event.tool.name, str)
        return {"message": event.tool.name}

    approve_handler: Callable[[ToolCallEvent, EventContext], EventResult] = approve
    assert approve_handler is approve


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
    ]


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
            "capabilities": {"ui": {"input": True}},
        },
    )
    assert init["name"] == "rpc"
    assert init["tools"][0]["name"] == "echo"

    result = await client.call(
        "extension.tool.execute",
        {"name": "echo", "input": {"text": "hello"}, "context": {"cwd": os.getcwd()}},
    )
    assert result == {"content": "HELLO:from-host"}
    assert [request["method"] for request in client.host_requests] == ["kodelet.ui.input"]

    server_reader.close()
    await asyncio.wait_for(task, timeout=1)


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

    async def call(self, method: str, params: Any) -> Any:
        self._next_id += 1
        request_id = self._next_id
        self._server_reader.feed(
            _frame({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        )
        while True:
            response = await self._server_writer.read_frame()
            if response.get("method"):
                self.host_requests.append(response)
                self._server_reader.feed(
                    _frame(
                        {
                            "jsonrpc": "2.0",
                            "id": response["id"],
                            "result": {"status": "submitted", "value": "from-host"},
                        }
                    )
                )
                continue
            if response.get("id") != request_id:
                continue
            if "error" in response:
                raise RuntimeError(response["error"]["message"])
            return response.get("result")


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

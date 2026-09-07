from __future__ import annotations

import asyncio
import os
import signal
import sys
import textwrap
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from kodelet_sdk import BaseModel, BridgeTransport, Client, Extension, ToolContext
from kodelet_sdk.agent import SpawnedProcess, SpawnOptions
from kodelet_sdk.agent.rpc import ACPRPCClient
from kodelet_sdk.agent.transport import ACP_MESSAGE_LIMIT, spawn_acp


def server_script(tmp_path: Path, action: str) -> str:
    script = tmp_path / "acp-server"
    script.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            r"""
            import json, os, signal, sys, time
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            def send(value):
                sys.stdout.write(json.dumps(value, ensure_ascii=False) + "\n")
                sys.stdout.flush()
            def reply(request, result):
                send({"jsonrpc": "2.0", "id": request["id"], "result": result})
            for line in sys.stdin:
                request = json.loads(line)
                if request["method"] == "initialize":
                    reply(request, {"_meta": {"steering": {"supported": True}}})
                elif request["method"] == "session/new":
                    reply(request, {"sessionId": "session"})
                else:
            """
        )
        + textwrap.indent(textwrap.dedent(action), "        "),
        encoding="utf-8",
    )
    script.chmod(0o755)
    return str(script)


async def wait_until(predicate: Callable[[], bool]) -> None:
    ready = asyncio.Event()
    loop = asyncio.get_running_loop()
    timer: asyncio.TimerHandle | None = None

    def check() -> None:
        nonlocal timer
        if predicate():
            ready.set()
        else:
            timer = loop.call_later(0.005, check)

    check()
    try:
        await asyncio.wait_for(ready.wait(), timeout=3)
    finally:
        if timer is not None:
            timer.cancel()


def assert_reaped(process: Any) -> None:
    assert process._process.returncode is not None
    with pytest.raises(ChildProcessError):
        os.waitpid(process._process.pid, os.WNOHANG)


@pytest.mark.asyncio
@pytest.mark.parametrize("resume", ["", "resumed-session"], ids=["new", "resume"])
async def test_default_acp_session_lifecycle_without_inline_extensions(
    tmp_path: Path, resume: str
) -> None:
    script = server_script(
        tmp_path,
        """
        if request["method"] == "session/load":
            reply(request, {})
        elif request["method"] == "session/prompt":
            send({"jsonrpc": "2.0", "method": "session/update", "params": {
                "sessionId": request["params"]["sessionId"],
                "update": {"sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "done"}}}})
            reply(request, {"stopReason": "end_turn"})
        """,
    )
    client = Client(command=script)
    try:
        session = await asyncio.wait_for(
            client.create_session(resume=resume, extensions=[], ui=cast(Any, None)), timeout=3
        )
        process = session._rpc._process
        assert session.id == (resume or "session")
        assert session in client._sessions
        assert session._rpc in client._rpcs

        response = await asyncio.wait_for(session.run_and_wait("hello"), timeout=3)
        assert response.content == "done"
        assert response.conversation_id == session.id
        assert response.stop_reason == "end_turn"

        await asyncio.wait_for(session.close(), timeout=3)
        await asyncio.wait_for(session.close(), timeout=3)
        assert not client._sessions
        assert not client._rpcs
        assert_reaped(process)
        with pytest.raises(RuntimeError, match="closed"):
            await session.run_and_wait("after close")
    finally:
        await asyncio.wait_for(client.close(), timeout=3)


@pytest.mark.asyncio
async def test_default_acp_spawn_accepts_replay_and_live_lines_above_16mib(tmp_path: Path) -> None:
    script = server_script(
        tmp_path,
        """
        if request["method"] == "session/load":
            reply(request, {"replayed": "R" * (17 * 1024 * 1024)})
        elif request["method"] == "session/prompt":
            send({"jsonrpc": "2.0", "method": "session/update", "params": {
                "sessionId": "session", "update": {"sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "π" * (9 * 1024 * 1024)}}}})
            reply(request, {"stopReason": "end_turn"})
        """,
    )
    client = Client(command=script)
    try:
        session = await asyncio.wait_for(client.create_session(resume="session"), timeout=10)
        response = await asyncio.wait_for(session.run_and_wait("large result"), timeout=10)
        assert response["content"] == "π" * (9 * 1024 * 1024)
        assert len(response["content"].encode()) > 16 * 1024 * 1024
        process = session._rpc._process
    finally:
        await asyncio.wait_for(client.close(), timeout=3)
    assert_reaped(process)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", ["stdout", "stderr"])
@pytest.mark.parametrize("newline", [True, False])
async def test_acp_rejects_real_lines_over_64mib_and_reaps(
    tmp_path: Path, stream: str, newline: bool
) -> None:
    assert ACP_MESSAGE_LIMIT == 64 * 1024 * 1024
    suffix = b"\n" if newline else b""
    script = server_script(
        tmp_path,
        f"""
        output = sys.{stream}.buffer
        output.write(b'x' * (64 * 1024 * 1024 + 1) + {suffix!r})
        output.flush()
        time.sleep(60)
        """,
    )
    client = Client(command=script)
    session = await asyncio.wait_for(client.create_session(), timeout=3)
    process = session._rpc._process
    try:
        with pytest.raises(RuntimeError, match=f"ACP {stream} read failed.*67108864"):
            await asyncio.wait_for(session.run_and_wait("oversize"), timeout=10)
    finally:
        await asyncio.wait_for(client.close(), timeout=3)
    assert_reaped(process)


@pytest.mark.asyncio
@pytest.mark.parametrize("close_stderr", [False, True])
async def test_acp_unexpected_stdout_eof_rejects_without_waiting_for_exit(
    tmp_path: Path, close_stderr: bool
) -> None:
    script = server_script(
        tmp_path,
        f"""
        if {close_stderr!r}: os.close(2)
        os.close(1)
        time.sleep(60)
        """,
    )
    client = Client(command=script)
    session = await client.create_session()
    process = session._rpc._process
    try:
        with pytest.raises(RuntimeError, match="stdout ended"):
            await asyncio.wait_for(session.run_and_wait("eof"), timeout=2)
    finally:
        await asyncio.wait_for(client.close(), timeout=3)
    assert_reaped(process)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", ["stdout", "stderr"])
async def test_acp_reader_failure_rejects_prompt_and_steering(tmp_path: Path, stream: str) -> None:
    script = server_script(tmp_path, "time.sleep(60)\n")
    client = Client(command=script)
    session = await client.create_session()
    rpc = session._rpc
    process = rpc._process
    pending = [
        asyncio.create_task(session.run_and_wait("blocked")),
        asyncio.create_task(rpc.steer_session(session.id, "guide")),
    ]
    try:
        await wait_until(lambda: len(rpc._pending) == 2)
        reader = cast(asyncio.StreamReader, getattr(process, stream))
        reader.set_exception(OSError("injected pipe failure"))
        results = await asyncio.wait_for(
            asyncio.gather(*pending, return_exceptions=True), timeout=2
        )
        assert all(
            isinstance(result, RuntimeError) and f"ACP {stream} read failed" in str(result)
            for result in results
        )
        assert rpc._pending == {}
    finally:
        await asyncio.wait_for(client.close(), timeout=3)
    assert_reaped(process)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["write", "drain", "blocked drain"])
async def test_acp_stdin_failure_or_blocked_drain_never_strands_pending_rpc(
    tmp_path: Path, failure: str
) -> None:
    client = Client(command=server_script(tmp_path, "time.sleep(60)\n"))
    session = await client.create_session()
    rpc = session._rpc
    process = cast(Any, rpc._process)
    draining = asyncio.Event()

    class Writer:
        def write(self, _data: bytes) -> None:
            if failure == "write":
                raise OSError("broken stdin")

        async def drain(self) -> None:
            draining.set()
            if failure == "drain":
                raise OSError("broken stdin drain")
            await asyncio.Event().wait()

        def close(self) -> None:
            pass

    process.stdin = Writer()
    pending = asyncio.create_task(rpc.prompt(session.id, []))
    try:
        if failure == "blocked drain":
            await asyncio.wait_for(draining.wait(), timeout=1)
            cast(asyncio.StreamReader, process.stdout).set_exception(
                OSError("stdout failed during stdin drain")
            )
        with pytest.raises(RuntimeError, match=r"ACP (stdin write|stdout read) failed"):
            await asyncio.wait_for(pending, timeout=2)
    finally:
        await asyncio.wait_for(client.close(), timeout=3)
    assert_reaped(process)


@pytest.mark.asyncio
async def test_acp_sigkill_closes_a_real_paused_pipe_before_reaping() -> None:
    process = cast(
        Any,
        await spawn_acp(
            sys.executable,
            [
                "-u",
                "-c",
                "import os,signal; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                "os.write(2,b'ready\\n'); data=b'x'*1048576\nwhile True: os.write(1,data)",
            ],
            {"env": dict(os.environ)},
        ),
    )
    stdout_pipe = process._transport.get_pipe_transport(1)
    stdout_pipe.pause_reading()
    assert not stdout_pipe.is_reading()
    assert await asyncio.wait_for(process.stderr.readline(), timeout=2) == b"ready\n"
    rpc = ACPRPCClient(process)
    pending = asyncio.create_task(rpc.request("blocked"))
    try:
        await wait_until(lambda: bool(rpc._pending))
        await asyncio.wait_for(rpc.close(), timeout=3)
        with pytest.raises(RuntimeError, match="closed"):
            await pending
        assert stdout_pipe.is_closing()
        assert process._process.returncode == -signal.SIGKILL
        assert_reaped(process)
    finally:
        process.kill()
        await asyncio.wait_for(process.wait(), timeout=1)


@pytest.mark.asyncio
async def test_repeated_startup_cancellation_retains_process_until_cleanup(tmp_path: Path) -> None:
    script = server_script(
        tmp_path, "sys.stderr.write('loading\\n'); sys.stderr.flush(); time.sleep(60)\n"
    )
    client = Client(command=script)
    creating = asyncio.create_task(client.create_session(resume="session"))
    await wait_until(lambda: bool(client._rpcs))
    rpc = next(iter(client._rpcs))
    await wait_until(lambda: "loading" in "".join(rpc._stderr_chunks))
    process = rpc._process
    creating.cancel()
    await wait_until(lambda: rpc._close_task is not None)
    creating.cancel()
    with pytest.raises(asyncio.CancelledError):
        await creating
    assert rpc in client._rpcs
    await asyncio.wait_for(client.close(), timeout=3)
    assert not client._rpcs
    assert_reaped(process)


def inline_server_script(tmp_path: Path, *, supported: bool = True, count: int = 1) -> str:
    """Run a provider-free ACP peer with real bidirectional extension frames.

    Both sides acknowledge relay acceptance independently of raw extension RPC
    completion. Initial extension frames arrive before session creation/loading
    completes, and nested host calls require the execute frame's prior ACK.
    """

    script = tmp_path / "inline-acp-server"
    script.write_text(
        f"#!{sys.executable}\nsupported = {supported!r}\ncount = {count}\n"
        + textwrap.dedent(
            r"""
            import json, os, sys, time

            def send(value):
                sys.stdout.write(json.dumps(value) + "\n")
                sys.stdout.flush()

            def reply(request, result):
                send({"jsonrpc": "2.0", "id": request["id"], "result": result})

            session_id = None
            pending = {}
            accepted = set()
            frame_id = 0
            attached = None
            prompt = None
            initialized = []

            def frame(extension_id, message=None, close=False):
                global frame_id
                frame_id += 1
                request_id = "frame-" + str(frame_id)
                params = {"sessionId": session_id, "runId": "run-1",
                          "extensionId": extension_id}
                if close:
                    params["close"] = True
                else:
                    params["message"] = message
                pending[request_id] = params
                send({"jsonrpc": "2.0", "id": request_id,
                      "method": "kodelet/extensionFrame", "params": params})

            def initialize_extension(index):
                extension_id = "inline-" + str(index)
                frame(extension_id, {"jsonrpc": "2.0", "id": 1,
                    "method": "extension.initialize", "params": {
                        "extension": {"id": extension_id, "cwd": "/runner/workspace"},
                        "capabilities": {"tools": {"updates": True},
                                         "conversations": {"fork": True},
                                         "ui": {"input": False}}}})

            def update(kind, **fields):
                send({"jsonrpc": "2.0", "method": "session/update", "params": {
                    "sessionId": session_id, "update": {"sessionUpdate": kind, **fields}}})

            def tool_content(content):
                return [{"type": "content", "content": {"type": "text", "text": content}}]

            for line in sys.stdin:
                request = json.loads(line)
                method = request.get("method")
                if method is None:
                    envelope = pending.pop(request["id"])
                    assert request.get("result") == {}, request
                    if envelope.get("close"):
                        reply(prompt, {"stopReason": "cancelled" if
                              prompt["params"]["prompt"][0]["text"] == "channel-close"
                              else "end_turn"})
                    else:
                        raw = envelope["message"]
                        if raw.get("method"):
                            accepted.add((envelope["extensionId"], raw.get("id")))
                    continue

                if method == "initialize":
                    assert request["params"]["clientCapabilities"]["_meta"][
                        "sessionExtensions"] == {"version": 1}
                    reply(request, {"protocolVersion": 1, "_meta": {
                        "sessionExtensions": {"version": 1}} if supported else {}})
                elif method in ("session/new", "session/load"):
                    assert supported, "Session requested without relay capability"
                    expected = {"version": 1, "extensionIds": [
                        "inline-" + str(i) for i in range(1, count + 1)]}
                    assert request["params"]["_meta"]["sessionExtensions"] == expected
                    session_id = request["params"].get("sessionId", "inline-session")
                    attached = request
                    initialize_extension(1)
                elif method == "session/prompt":
                    assert request["params"]["sessionId"] == session_id
                    prompt = request
                    update("tool_call", toolCallId="call-1", toolName="echo", rawInput={})
                    frame("inline-1", {"jsonrpc": "2.0", "id": 2,
                        "method": "extension.tool.execute", "params": {
                            "name": "echo", "input": {
                                "text": request["params"]["prompt"][0]["text"]},
                            "context": {"sessionId": session_id,
                                        "conversationId": session_id,
                                        "cwd": "/runner/workspace"}}})
                elif method == "kodelet/extensionFrame":
                    envelope = request["params"]
                    assert envelope["sessionId"] == session_id, envelope
                    assert envelope["runId"] == "run-1", envelope
                    assert envelope["extensionId"] in [
                        "inline-" + str(i) for i in range(1, count + 1)], envelope
                    reply(request, {})
                    if envelope.get("close"):
                        continue
                    raw = envelope["message"]
                    extension_id = envelope["extensionId"]
                    if not raw.get("method"):
                        assert (extension_id, raw["id"]) in accepted, "Missing frame ACK"
                        assert "error" not in raw, raw
                        if raw["id"] == 1:
                            initialized.append(raw["result"]["name"])
                            if len(initialized) < count:
                                initialize_extension(len(initialized) + 1)
                            else:
                                assert initialized == ["local-" + str(i)
                                                       for i in range(1, count + 1)]
                                reply(attached, {"sessionId": session_id})
                        else:
                            assert raw["id"] == 2, raw
                            content = raw["result"]["content"]
                            update("tool_call_update", toolCallId="call-1", status="completed",
                                   content=tool_content(content))
                            update("agent_message_chunk", content={"type": "text", "text": content})
                            frame(extension_id, close=True)
                        continue

                    assert raw["parentId"] == 2, raw
                    assert (extension_id, 2) in accepted, "Callback awaited before frame ACK"
                    if raw["method"] == "kodelet.conversation.fork":
                        result = {"conversationId": "fork:" + raw["params"]["name"]}
                    elif raw["method"] == "kodelet.tool.update":
                        mode = prompt["params"]["prompt"][0]["text"]
                        if mode == "acp-failure":
                            os.close(1)
                            time.sleep(60)
                            continue
                        if mode == "session-close":
                            sys.stderr.write("host-rpc-pending\n")
                            sys.stderr.flush()
                            continue
                        if mode == "channel-close":
                            frame(extension_id, close=True)
                            continue
                        update("tool_call_update", toolCallId="call-1", status="in_progress",
                               content=tool_content(raw["params"]["content"]))
                        result = {}
                    else:
                        raise AssertionError("Unexpected host RPC: " + raw["method"])
                    frame(extension_id, {"jsonrpc": "2.0", "id": raw["id"], "result": result})
                elif method != "session/cancel":
                    raise AssertionError("Unexpected ACP request: " + method)
            """
        ),
        encoding="utf-8",
    )
    script.chmod(0o755)
    return str(script)


@pytest.mark.asyncio
@pytest.mark.parametrize("resume", ["", "resumed-inline"], ids=["new", "resume"])
@pytest.mark.parametrize("transport", ["unix", "tcp"])
async def test_inline_acp_subprocess_callback_host_rpc_and_updates(
    tmp_path: Path, resume: str, transport: BridgeTransport
) -> None:
    ext = Extension(name="local-1")
    callback_calls: list[tuple[str, str | None, str]] = []
    updates: list[str] = []
    ui_calls: list[str] = []

    class EchoInput(BaseModel):
        text: str

    @ext.tool("echo", description="Live local callback", input_schema=EchoInput)
    async def echo(input: EchoInput, ctx: ToolContext) -> str:
        callback_calls.append((input.text, ctx.conversation_id, ctx.cwd))
        await ctx.update("starting")
        fork_id = await ctx.fork_conversation(name=input.text)
        await ctx.update(f"created {fork_id}")
        answer = await ctx.ui.input({"title": "Local input"})
        return f"{fork_id}:{answer}"

    def answer(request: Any) -> str:
        ui_calls.append(request["title"])
        return "in-memory"

    client = Client(command=inline_server_script(tmp_path, count=2))
    try:
        session = await asyncio.wait_for(
            client.create_session(
                extensions=[ext, Extension(name="local-2")], resume=resume,
                extension_transport=transport, ui={"input": answer},
            ),
            timeout=3,
        )
        process = session._rpc._process
        assert session.id == (resume or "inline-session")
        session.on("tool.update", lambda event: updates.append(event.data.result))
        response = await asyncio.wait_for(session.run_and_wait("hello"), timeout=3)
        assert response.content == "fork:hello:in-memory"
        assert response.conversation_id == session.id
        assert response.stop_reason == "end_turn"
        assert callback_calls == [("hello", session.id, "/runner/workspace")]
        assert ui_calls == ["Local input"]
        assert updates == ["starting", "created fork:hello"]
        assert [event.data.result for event in response.events if event.type == "tool.update"] == [
            "created fork:hello"
        ]
        assert [event.data.result for event in response.events if event.type == "tool.result"] == [
            "fork:hello:in-memory"
        ]
    finally:
        await asyncio.wait_for(client.close(), timeout=3)
    assert_reaped(process)


@pytest.mark.asyncio
@pytest.mark.parametrize("resume", ["", "resumed-inline"], ids=["new", "resume"])
async def test_inline_acp_subprocess_missing_capability_fails_before_attachment_and_reaps(
    tmp_path: Path, resume: str
) -> None:
    processes: list[SpawnedProcess] = []

    async def spawn(command: str, args: Sequence[str], options: SpawnOptions) -> SpawnedProcess:
        process = await spawn_acp(command, args, options)
        processes.append(process)
        return process

    def entrypoint(_ext: Extension) -> None:
        pytest.fail("Extensions must not initialize without the relay capability")

    client = Client(command=inline_server_script(tmp_path, supported=False), spawn=spawn)
    try:
        with pytest.raises(RuntimeError, match=r"(?i)(session.?extensions|inline extension)"):
            await asyncio.wait_for(client.create_session(extensions=[entrypoint], resume=resume), 3)
        assert len(processes) == 1
        assert not client._sessions
        assert not client._rpcs
        assert_reaped(processes[0])
    finally:
        await asyncio.wait_for(client.close(), timeout=3)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["channel-close", "session-close", "acp-failure"])
async def test_inline_acp_subprocess_cancels_pending_host_rpc_and_callback(
    tmp_path: Path, mode: str
) -> None:
    ext = Extension(name="local-1")
    started = asyncio.Event()
    cancelled = asyncio.Event()

    @ext.tool("echo", description="Wait for nested host RPC", input_schema={"type": "object"})
    async def echo(_input: Any, ctx: ToolContext) -> str:
        started.set()
        try:
            await ctx.update("waiting for runner")
            pytest.fail("Pending host RPC must not complete successfully")
        except asyncio.CancelledError:
            cancelled.set()
            raise

    client = Client(command=inline_server_script(tmp_path))
    session = await asyncio.wait_for(client.create_session(extensions=[ext]), timeout=3)
    process = session._rpc._process
    running = asyncio.create_task(session.run_and_wait(mode))
    try:
        await asyncio.wait_for(started.wait(), timeout=3)
        if mode == "session-close":
            await wait_until(lambda: "host-rpc-pending" in "".join(session._rpc._stderr_chunks))
            await asyncio.wait_for(session.close(), timeout=3)
        if mode == "channel-close":
            response = await asyncio.wait_for(running, timeout=3)
            assert response.stop_reason == "cancelled"
        else:
            with pytest.raises(RuntimeError, match=r"closed|stdout ended"):
                await asyncio.wait_for(running, timeout=3)
        await asyncio.wait_for(cancelled.wait(), timeout=3)
        assert not session._rpc._pending
        if mode == "acp-failure":
            await wait_until(lambda: cast(Any, process)._process.returncode is not None)
            assert_reaped(process)
    finally:
        await asyncio.wait_for(client.close(), timeout=3)
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
    assert_reaped(process)

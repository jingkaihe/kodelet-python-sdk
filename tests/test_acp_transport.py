from __future__ import annotations

import asyncio
import os
import signal
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from kodelet_sdk import Client
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

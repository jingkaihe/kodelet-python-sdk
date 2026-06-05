from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .types import AgentRunError, SpawnedProcess

ACP_PROTOCOL_VERSION = 1


class RPCError(RuntimeError):
    def __init__(self, error: Mapping[str, Any]) -> None:
        super().__init__(str(error.get("message") or "JSON-RPC error"))
        self.code = int(error.get("code") or 0)
        self.data = error.get("data")


class ACPRPCClient:
    """Line-oriented JSON-RPC client for the ``kodelet acp`` subprocess."""

    def __init__(self, process: SpawnedProcess) -> None:
        if process.stdin is None:
            raise RuntimeError("kodelet acp process did not expose stdin")
        self._process = process
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._stderr_chunks: list[str] = []
        self._notification_handlers: set[Callable[[str, Any], None]] = set()
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._closed = False
        self._stdout_task = asyncio.create_task(self._read_stdout()) if process.stdout else None
        self._stderr_task = asyncio.create_task(self._read_stderr()) if process.stderr else None
        self._wait_task = asyncio.create_task(self._wait_for_process())

    async def initialize(self) -> None:
        await self.request(
            "initialize",
            {
                "protocolVersion": ACP_PROTOCOL_VERSION,
                "clientCapabilities": {
                    "terminal": True,
                    "fs": {"readTextFile": False, "writeTextFile": False},
                },
                "clientInfo": {"name": "kodelet-sdk", "title": "Kodelet SDK"},
            },
        )

    async def create_session(self, cwd: str) -> str:
        result = await self.request("session/new", {"cwd": cwd})
        if not isinstance(result, Mapping) or not isinstance(result.get("sessionId"), str):
            raise RuntimeError("Invalid session/new response from kodelet acp")
        return str(result["sessionId"])

    async def load_session(self, session_id: str, cwd: str) -> str:
        await self.request("session/load", {"sessionId": session_id, "cwd": cwd})
        return session_id

    async def prompt(self, session_id: str, prompt: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        result = await self.request(
            "session/prompt",
            {"sessionId": session_id, "prompt": list(prompt)},
        )
        if not isinstance(result, Mapping):
            return {}
        stop_reason = result.get("stopReason")
        return {"stopReason": stop_reason} if isinstance(stop_reason, str) else {}

    def cancel_session(self, session_id: str) -> None:
        self.notify("session/cancel", {"sessionId": session_id})

    def on_notification(self, handler: Callable[[str, Any], None]) -> Callable[[], None]:
        self._notification_handlers.add(handler)

        def unsubscribe() -> None:
            self._notification_handlers.discard(handler)

        return unsubscribe

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._process.terminate()
        try:
            await asyncio.wait_for(self._process.wait(), timeout=1)
        except TimeoutError:
            self._process.kill()
            await self._process.wait()
        for task in (self._stdout_task, self._stderr_task, self._wait_task):
            if task is not None and not task.done():
                task.cancel()
        for task in list(self._background_tasks):
            task.cancel()

    async def request(self, method: str, params: Any | None = None) -> Any:
        if self._closed:
            raise RuntimeError("kodelet acp process is closed")
        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._write(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            )
        except Exception:
            self._pending.pop(request_id, None)
            raise
        return await future

    def notify(self, method: str, params: Any | None = None) -> None:
        if not self._closed:
            task = asyncio.create_task(
                self._write({"jsonrpc": "2.0", "method": method, "params": params})
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def _write(self, message: Mapping[str, Any]) -> None:
        stdin = self._process.stdin
        if stdin is None:
            raise RuntimeError("kodelet acp process stdin is closed")
        payload = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
        stdin.write(payload)
        drain = getattr(stdin, "drain", None)
        if drain is not None:
            result = drain()
            if inspect.isawaitable(result):
                await result

    async def _read_stdout(self) -> None:
        stdout = self._process.stdout
        if stdout is None:
            return
        while True:
            line = await stdout.readline()
            if not line:
                return
            await self._handle_line(line.decode("utf-8", errors="replace").rstrip("\r\n"))

    async def _read_stderr(self) -> None:
        stderr = self._process.stderr
        if stderr is None:
            return
        while True:
            line = await stderr.readline()
            if not line:
                return
            self._stderr_chunks.append(line.decode("utf-8", errors="replace"))

    async def _wait_for_process(self) -> None:
        code = await self._process.wait()
        if self._closed and not self._pending:
            return
        self._closed = True
        stderr = "".join(self._stderr_chunks)
        status = code if code is not None else "unknown"
        message = stderr.strip() or f"kodelet acp exited with status {status}"
        self._reject_pending(AgentRunError(message, code=code, signal=None, stderr=stderr))

    async def _handle_line(self, line: str) -> None:
        trimmed = line.strip()
        if not trimmed:
            return
        try:
            message = json.loads(trimmed)
        except json.JSONDecodeError:
            self._notify_handlers("$/stdout", {"line": line})
            return
        if not isinstance(message, Mapping):
            return

        method = message.get("method")
        message_id = message.get("id")
        if isinstance(method, str) and message_id is not None:
            await self._respond_to_server_request(message)
            return
        if isinstance(method, str):
            self._notify_handlers(method, message.get("params"))
            return
        if not isinstance(message_id, int):
            return
        pending = self._pending.pop(message_id, None)
        if pending is None:
            return
        error = message.get("error")
        if isinstance(error, Mapping):
            pending.set_exception(RPCError(error))
        else:
            pending.set_result(message.get("result"))

    async def _respond_to_server_request(self, message: Mapping[str, Any]) -> None:
        await self._write(
            {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "error": {
                    "code": -32601,
                    "message": f"Unsupported client RPC method: {message.get('method')}",
                },
            }
        )

    def _notify_handlers(self, method: str, params: Any) -> None:
        for handler in list(self._notification_handlers):
            handler(method, params)

    def _reject_pending(self, error: Exception) -> None:
        for pending in self._pending.values():
            pending.set_exception(error)
        self._pending.clear()


__all__ = ["ACP_PROTOCOL_VERSION", "ACPRPCClient", "RPCError"]

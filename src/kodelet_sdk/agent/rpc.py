from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from typing import Any, cast

from ..api import Entrypoint, Extension
from .relay import EXTENSION_FRAME_METHOD, SESSION_EXTENSIONS_VERSION, SessionExtensionRelay
from .transport import ACP_MESSAGE_LIMIT
from .types import (
    AgentRunError,
    AgentUIHandlers,
    SessionSteeringOutcome,
    SessionSteerResult,
    SpawnedProcess,
)

ACP_PROTOCOL_VERSION = 1


class RPCError(RuntimeError):
    def __init__(self, error: Mapping[str, Any]) -> None:
        super().__init__(str(error.get("message") or "JSON-RPC error"))
        self.code = int(error.get("code") or 0)
        self.data = error.get("data")


class ACPRPCClient:
    """Line-oriented JSON-RPC client for the ``kodelet acp`` subprocess."""

    def __init__(
        self,
        process: SpawnedProcess,
        *,
        extensions: Sequence[Entrypoint | Extension] = (),
        ui: AgentUIHandlers | None = None,
    ) -> None:
        self._process = process
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._stderr_chunks: list[str] = []
        self._notification_handlers: set[Callable[[str, Any], None]] = set()
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._closed = False
        self._terminal_error: Exception | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._steering_supported = False
        self._extensions_supported = False
        self._hierarchy_supported = False
        self._extension_relay = (
            SessionExtensionRelay(extensions, self.request, ui) if extensions else None
        )
        self._session_started = False
        self._stdout_task = asyncio.create_task(self._read_stdout()) if process.stdout else None
        self._stderr_task = asyncio.create_task(self._read_stderr()) if process.stderr else None
        self._wait_task = asyncio.create_task(self._wait_for_process())
        for name, stream in (("stdin", process.stdin), ("stdout", process.stdout)):
            if stream is None:
                self._fail_transport(RuntimeError(f"kodelet acp process did not expose {name}"))

    async def initialize(self) -> None:
        result = await self.request(
            "initialize",
            {
                "protocolVersion": ACP_PROTOCOL_VERSION,
                "clientCapabilities": {
                    "terminal": True,
                    "fs": {"readTextFile": False, "writeTextFile": False},
                    "_meta": {"sessionExtensions": {"version": SESSION_EXTENSIONS_VERSION}},
                },
                "clientInfo": {"name": "kodelet-sdk", "title": "Kodelet SDK"},
            },
        )
        metadata = result.get("_meta") if isinstance(result, Mapping) else None
        steering = metadata.get("steering") if isinstance(metadata, Mapping) else None
        self._steering_supported = (
            isinstance(steering, Mapping) and steering.get("supported") is True
        )
        extensions = metadata.get("sessionExtensions") if isinstance(metadata, Mapping) else None
        self._extensions_supported = (
            isinstance(extensions, Mapping)
            and type(extensions.get("version")) is int
            and extensions["version"] == SESSION_EXTENSIONS_VERSION
        )
        hierarchy = metadata.get("conversationHierarchy") if isinstance(metadata, Mapping) else None
        self._hierarchy_supported = (
            isinstance(hierarchy, Mapping)
            and type(hierarchy.get("version")) is int
            and hierarchy["version"] == 1
        )
        if self._extension_relay is not None and not self._extensions_supported:
            raise RuntimeError(
                "Inline extensions require kodelet acp sessionExtensions version 1 support; "
                "upgrade the selected daemon/runner and ACP client"
            )

    async def create_session(self, cwd: str, *, parent_conversation_id: str | None = None) -> str:
        if parent_conversation_id is not None and not self._hierarchy_supported:
            raise RuntimeError(
                "Child conversations require kodelet acp conversationHierarchy version 1 support; "
                "update Kodelet and the daemon"
            )
        self._session_started = True
        result = await self.request(
            "session/new", {"cwd": cwd, **self._session_params(parent_conversation_id)}
        )
        if not isinstance(result, Mapping) or not isinstance(result.get("sessionId"), str):
            raise RuntimeError("Invalid session/new response from kodelet acp")
        if self._extension_relay is not None:
            self._extension_relay.bind_session(result["sessionId"])
        return str(result["sessionId"])

    async def load_session(self, session_id: str, cwd: str) -> str:
        if self._extension_relay is not None:
            self._extension_relay.bind_session(session_id)
        self._session_started = True
        await self.request(
            "session/load", {"sessionId": session_id, "cwd": cwd, **self._session_params()}
        )
        return session_id

    def _session_params(self, parent_conversation_id: str | None = None) -> dict[str, Any]:
        metadata = dict(self._extension_relay.metadata) if self._extension_relay else {}
        if parent_conversation_id is not None:
            metadata["conversationHierarchy"] = {
                "version": 1,
                "parentConversationId": parent_conversation_id,
            }
        return {"_meta": metadata} if metadata else {}

    async def prompt(self, session_id: str, prompt: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        result = await self.request(
            "session/prompt",
            {"sessionId": session_id, "prompt": list(prompt)},
        )
        if not isinstance(result, Mapping):
            return {}
        stop_reason = result.get("stopReason")
        return {"stopReason": stop_reason} if isinstance(stop_reason, str) else {}

    async def steer_session(self, session_id: str, message: str) -> SessionSteerResult:
        """Queue steering for an active ACP session."""

        if not self._steering_supported:
            raise RuntimeError("kodelet acp does not advertise session steering support")
        result = await self.request(
            "_session/steering",
            {
                "sessionId": session_id,
                "prompt": [{"type": "text", "text": message}],
                "_meta": {"steering": {"idleBehavior": "promptRequired"}},
            },
        )
        outcome = result.get("outcome") if isinstance(result, Mapping) else None
        if outcome not in {"injected", "startedNewTurn", "promptRequired", "failed"}:
            raise RuntimeError("Invalid _session/steering response from kodelet acp")
        reason = result.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise RuntimeError("Invalid _session/steering response from kodelet acp")
        if outcome == "promptRequired" and reason != "noRunningTurn":
            raise RuntimeError("Invalid _session/steering response from kodelet acp")
        response: SessionSteerResult = {
            "outcome": cast(SessionSteeringOutcome, outcome),
        }
        if isinstance(reason, str):
            response["reason"] = reason
        return response

    def cancel_session(self, session_id: str) -> None:
        self.notify("session/cancel", {"sessionId": session_id})

    def on_notification(self, handler: Callable[[str, Any], None]) -> Callable[[], None]:
        self._notification_handlers.add(handler)

        def unsubscribe() -> None:
            self._notification_handlers.discard(handler)

        return unsubscribe

    async def close(self) -> None:
        self._closed = True
        self._reject_pending(RuntimeError("kodelet acp process closed"))
        # Repeated caller cancellation must not cancel process ownership or
        # interrupt SIGKILL/reaping. Failed cleanup remains explicitly retryable.
        if self._close_task is None or (
            self._close_task.done() and self._close_task.exception() is not None
        ):
            self._start_close()
        assert self._close_task is not None
        await asyncio.shield(self._close_task)

    def _start_close(self) -> None:
        self._close_task = asyncio.create_task(self._close_process())
        # Transport failure can start cleanup without a waiting caller. Keep
        # its error on the task for close(), without an unhandled-task warning.
        self._close_task.add_done_callback(
            lambda task: None if task.cancelled() else task.exception()
        )

    async def _close_process(self) -> None:
        try:
            with suppress(ProcessLookupError):
                self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=1)
            except TimeoutError:
                with suppress(ProcessLookupError):
                    self._process.kill()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=1)
                except TimeoutError as exc:
                    raise RuntimeError(
                        "kodelet acp process did not close after SIGKILL; cleanup is incomplete"
                    ) from exc
        finally:
            if self._extension_relay is not None:
                await self._extension_relay.close()
            tasks = [
                task
                for task in (
                    self._stdout_task,
                    self._stderr_task,
                    self._wait_task,
                    *self._background_tasks,
                )
                if task is not None
            ]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def _fail_transport(self, error: Exception) -> None:
        if self._closed:
            return
        self._closed, self._terminal_error = True, error
        self._reject_pending(error)
        with suppress(ProcessLookupError):
            self._process.kill()
        if self._close_task is None:
            self._start_close()

    async def request(self, method: str, params: Any | None = None) -> Any:
        if self._closed:
            raise self._terminal_error or RuntimeError("kodelet acp process is closed")
        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        writing = self._send(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        )
        try:
            return await future
        finally:
            self._pending.pop(request_id, None)
            writing.cancel()
            await asyncio.gather(writing, return_exceptions=True)

    def notify(self, method: str, params: Any | None = None) -> None:
        if not self._closed:
            self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _send(self, message: Mapping[str, Any]) -> asyncio.Task[None]:
        task = asyncio.create_task(self._write(message))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def _write(self, message: Mapping[str, Any]) -> None:
        try:
            if self._closed:
                return
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
        except Exception as exc:
            self._fail_transport(RuntimeError(f"ACP stdin write failed: {exc}"))

    async def _read_stdout(self) -> None:
        stdout = self._process.stdout
        if stdout is None:
            return
        try:
            while not self._closed:
                line = await stdout.readline()
                if not line:
                    self._fail_transport(
                        RuntimeError("ACP stdout ended before the client closed the session")
                    )
                    return
                self._check_line_limit(line)
                await self._handle_line(line.decode("utf-8", errors="replace").rstrip("\r\n"))
        except Exception as exc:
            self._fail_transport(
                RuntimeError(f"ACP stdout read failed (limit {ACP_MESSAGE_LIMIT} bytes): {exc}")
            )

    async def _read_stderr(self) -> None:
        stderr = self._process.stderr
        if stderr is None:
            return
        try:
            while True:
                line = await stderr.readline()
                if not line:
                    return
                self._check_line_limit(line)
                self._stderr_chunks.append(line.decode("utf-8", errors="replace"))
        except Exception as exc:
            self._fail_transport(
                RuntimeError(f"ACP stderr read failed (limit {ACP_MESSAGE_LIMIT} bytes): {exc}")
            )

    @staticmethod
    def _check_line_limit(line: bytes) -> None:
        if len(line) - int(line.endswith(b"\n")) > ACP_MESSAGE_LIMIT:
            raise ValueError(f"ACP message exceeds {ACP_MESSAGE_LIMIT} byte limit")

    async def _wait_for_process(self) -> None:
        try:
            code = await self._process.wait()
        except Exception as exc:
            self._fail_transport(RuntimeError(f"ACP process wait failed: {exc}"))
            return
        if self._closed and not self._pending:
            return
        stderr = "".join(self._stderr_chunks)
        status = code if code is not None else "unknown"
        message = stderr.strip() or f"kodelet acp exited with status {status}"
        self._fail_transport(AgentRunError(message, code=code, signal=None, stderr=stderr))

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
        if pending.done():
            return
        error = message.get("error")
        if isinstance(error, Mapping):
            pending.set_exception(RPCError(error))
        else:
            pending.set_result(message.get("result"))

    async def _respond_to_server_request(self, message: Mapping[str, Any]) -> None:
        if message.get("method") == EXTENSION_FRAME_METHOD:
            response: dict[str, Any] = {"jsonrpc": "2.0", "id": message.get("id")}
            try:
                if not self._extensions_supported or not self._session_started:
                    raise ValueError("Inline extension session is not attached")
                if self._extension_relay is None:
                    raise ValueError("Unknown inline extensionId; no inline extensions attached")
                self._extension_relay.accept(message.get("params"))
                response["result"] = {}
            except ValueError as exc:
                response["error"] = {"code": -32602, "message": str(exc)}
            # Acceptance only: callbacks run off the reader, which must remain
            # available to receive their nested host RPC responses and ACKs.
            self._send(response)
            return
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
            if not pending.done():
                pending.set_exception(error)
        self._pending.clear()


__all__ = ["ACP_PROTOCOL_VERSION", "ACPRPCClient", "RPCError"]

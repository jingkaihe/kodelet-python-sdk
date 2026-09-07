from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import copy
from typing import Any, cast

from .._utils import maybe_await
from ..api import Entrypoint, Extension, create_extension_host
from ..runtime import ExtensionMessageDispatcher, StdioHostRPCClient
from .types import AgentUIHandlers

EXTENSION_FRAME_METHOD = "kodelet/extensionFrame"
SESSION_EXTENSIONS_VERSION = 1


class SessionExtensionRelay:
    """Own run-scoped extension channels on one ACP session connection."""

    def __init__(
        self,
        entrypoints: Sequence[Entrypoint | Extension],
        request: Callable[[str, Any], Awaitable[Any]],
        ui: AgentUIHandlers | None = None,
    ) -> None:
        self._entrypoints = {
            f"inline-{index}": entrypoint for index, entrypoint in enumerate(entrypoints, start=1)
        }
        self._request = request
        self._ui = ui or {}
        self._session_id: str | None = None
        self._channels: dict[tuple[str, str], _ExtensionChannel] = {}
        self._retired: set[tuple[str, str]] = set()
        self._closing: set[asyncio.Task[None]] = set()
        self._closed = False

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "sessionExtensions": {
                "version": SESSION_EXTENSIONS_VERSION,
                "extensionIds": list(self._entrypoints),
            }
        }

    def bind_session(self, session_id: str) -> None:
        # A runner can initialize while session/new is still pending. Pin that
        # first session ID and verify it against the eventual creation response.
        if self._session_id is not None and self._session_id != session_id:
            raise ValueError("Unknown inline extension sessionId")
        self._session_id = session_id

    def accept(self, params: Any) -> None:
        """Validate and enqueue a frame without awaiting callback or host RPC work."""

        if self._closed:
            raise ValueError("Inline extension relay is closed")
        if not isinstance(params, Mapping):
            raise ValueError("Invalid inline extension frame")
        session_id, run_id, extension_id = (
            params.get(key) for key in ("sessionId", "runId", "extensionId")
        )
        if not all(
            isinstance(value, str) and value for value in (session_id, run_id, extension_id)
        ):
            raise ValueError("Extension frames require sessionId, runId and extensionId")
        if extension_id not in self._entrypoints:
            raise ValueError(f"Unknown inline extensionId: {extension_id}")
        if self._session_id is not None and session_id != self._session_id:
            raise ValueError("Unknown inline extension sessionId")
        key = (cast(str, run_id), cast(str, extension_id))
        message = params.get("message")
        closing = params.get("close", False)
        if not isinstance(closing, bool) or (closing and message is not None):
            raise ValueError("Extension frames require either message or close: true")
        if closing:
            if key not in self._channels and key not in self._retired:
                raise ValueError("Unknown inline extension runId")
            self._retire(key)
            return
        if not isinstance(message, Mapping):
            raise ValueError("Extension frame message must be a JSON-RPC object")
        if key in self._retired:
            raise ValueError("Inline extension channel is closed; it cannot be restarted")
        channel = self._channels.get(key)
        if channel is None:
            if message.get("method") != "extension.initialize" or not isinstance(
                message.get("id"), int | str
            ):
                raise ValueError("Unknown inline extension runId; initialize must be first")
            self.bind_session(cast(str, session_id))
            channel = _ExtensionChannel(self, key, self._entrypoints[extension_id])
            self._channels[key] = channel
        elif message.get("method") == "extension.initialize":
            raise ValueError("Inline extension channel is already initialized")
        channel.enqueue(message)

    def _retire(self, key: tuple[str, str], *, notify: bool = False) -> None:
        channel = self._channels.pop(key, None)
        self._retired.add(key)
        if channel is None:
            return
        channel.closed = True
        task = asyncio.create_task(self._dispose(channel, notify=notify))
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)

    async def _dispose(self, channel: _ExtensionChannel, *, notify: bool) -> None:
        await channel.close()
        if notify and not self._closed:
            try:
                await self._request(EXTENSION_FRAME_METHOD, {**channel.envelope, "close": True})
            except Exception:
                # The channel is terminal even if the ACP connection also failed.
                pass

    async def close(self) -> None:
        self._closed = True
        for key in list(self._channels):
            self._retire(key)
        if self._closing:
            await asyncio.gather(*self._closing)
        self._retired.clear()
        self._entrypoints.clear()
        self._ui = {}


class _ExtensionChannel:
    def __init__(
        self, relay: SessionExtensionRelay, key: tuple[str, str], entrypoint: Entrypoint | Extension
    ) -> None:
        self._relay = relay
        self._key = key
        self.envelope = {
            "sessionId": relay._session_id,
            "runId": key[0],
            "extensionId": key[1],
        }
        self.closed = False
        self._messages: asyncio.Queue[Mapping[str, Any]] = asyncio.Queue()
        self._host_client = StdioHostRPCClient(send_message=self._send)
        self._dispatcher: ExtensionMessageDispatcher | None = None
        self._ui_tasks: set[asyncio.Task[None]] = set()
        self._worker = asyncio.create_task(self._run(entrypoint))

    def enqueue(self, message: Mapping[str, Any]) -> None:
        self._messages.put_nowait(message)

    async def _run(self, entrypoint: Entrypoint | Extension) -> None:
        try:
            # Share registrations and the original callback closures, not the
            # mutable initialization context of a different session or run.
            host = copy(await create_extension_host(entrypoint))
            host._init_params = None
            self._dispatcher = ExtensionMessageDispatcher(
                host, self._host_client, on_error=self._failed
            )
            while not self.closed:
                message = await self._messages.get()
                if message.get("method") == "extension.initialize":
                    message = self._with_ui_capabilities(message)
                await self._dispatcher.handle_message(message)
        except Exception as exc:
            self._failed(exc)

    def _failed(self, _error: Exception) -> None:
        self._relay._retire(self._key, notify=True)

    def _with_ui_capabilities(self, message: Mapping[str, Any]) -> Mapping[str, Any]:
        if not self._relay._ui:
            return message
        params = dict(message.get("params") or {})
        capabilities = dict(params.get("capabilities") or {})
        ui = dict(capabilities.get("ui") or {})
        ui.update({name: True for name, handler in self._relay._ui.items() if callable(handler)})
        return {
            **message,
            "params": {**params, "capabilities": {**capabilities, "ui": ui}},
        }

    async def _send(self, message: Mapping[str, Any]) -> None:
        if self.closed:
            raise RuntimeError("Inline extension channel is closed")
        method, request_id = message.get("method"), message.get("id")
        name = (
            method.removeprefix("kodelet.ui.")
            if isinstance(method, str) and method.startswith("kodelet.ui.")
            else ""
        )
        handler = self._relay._ui.get(name)
        if callable(handler) and isinstance(request_id, int):
            # Do not hold the runtime's write lock while waiting for user input.
            # Link to the runtime-owned future so caller cancellation, timeout,
            # request completion and ID reuse also cancel the local interaction.
            _, pending = self._host_client._pending[request_id]
            task = asyncio.create_task(self._handle_ui(message, name, handler))
            self._ui_tasks.add(task)
            pending.add_done_callback(
                lambda _future: task.cancel() if not task.cancelling() else None
            )
            task.add_done_callback(self._ui_tasks.discard)
            return
        try:
            await self._relay._request(
                EXTENSION_FRAME_METHOD, {**self.envelope, "message": dict(message)}
            )
        except Exception as exc:
            self._failed(exc)
            raise

    async def _handle_ui(
        self, message: Mapping[str, Any], name: str, handler: Callable[..., Any]
    ) -> None:
        response: dict[str, Any] = {"jsonrpc": "2.0", "id": message["id"]}
        try:
            value = await maybe_await(handler(message.get("params")))
            result: dict[str, Any] = {"status": "submitted"}
            if name == "confirm":
                result["confirmed"] = bool(value)
            elif name != "notify":
                result = {"status": "dismissed"} if value is None else {**result, "value": value}
            response["result"] = result
        except asyncio.CancelledError:
            response["result"] = {"status": "dismissed"}
        except Exception as exc:
            response["error"] = {"code": -32000, "message": str(exc)}
        if not self.closed:
            self._host_client.handle_response(response)

    async def close(self) -> None:
        self.closed = True
        tasks = [self._worker, *self._ui_tasks]
        for task in tasks:
            if not task.cancelling():
                task.cancel()
        # Cancel originating callbacks before joining UI cleanup: a UI handler's
        # finally block may itself wait for the tool's cancellation to finish.
        if self._dispatcher is not None:
            await self._dispatcher.close()
        else:
            await self._host_client.close()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._messages = asyncio.Queue()

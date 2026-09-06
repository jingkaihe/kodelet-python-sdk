"""Scoped delegated executions over the existing authenticated runner host RPC."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from .execution import ExecutionOptions

if TYPE_CHECKING:
    from .context import BackgroundTaskLease, HostRPCClient


def _result(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not value.get("conversationId") or not value.get("runId"):
        raise RuntimeError("Invalid central child execution response")
    if not isinstance(value.get("done"), bool):
        raise RuntimeError("Invalid central child execution status")
    return value


class ChildExecution:
    """A durable child identity with progress polling and exact cancellation."""

    def __init__(self, result: dict[str, Any], call: Callable[..., Awaitable[Any]],
                 lease_id: str | None) -> None:
        self.conversation_id = str(result["conversationId"])
        self.run_id = str(result["runId"])
        self._result, self._call, self._lease_id = result, call, lease_id
        self._after = 0

    def _params(self) -> dict[str, Any]:
        return {"childId": self.conversation_id,
                **({"leaseId": self._lease_id} if self._lease_id else {})}

    async def read(self) -> dict[str, Any]:
        """Read bounded progress after the last observed sequence."""
        self._result = _result(await self._call("kodelet.child.read",
                                               {**self._params(), "after": self._after}))
        return self._result

    async def cancel(self) -> None:
        """Cancel only this child, not its parent or siblings."""
        await self._call("kodelet.child.cancel", self._params())

    async def wait(self, *, on_event: Callable[[dict[str, Any]], Any] | None = None
                   ) -> dict[str, Any]:
        """Wait for completion; cancel the child if this wait task is canceled."""
        try:
            while True:
                for event in self._result.get("events", []):
                    if event["sequence"] > self._after:
                        if on_event is not None:
                            value = on_event(event)
                            if inspect.isawaitable(value):
                                await value
                        self._after = event["sequence"]
                if self._result["done"]:
                    if self._result.get("error") or self._result.get("cancelled"):
                        raise RuntimeError(self._result.get("error") or "Child canceled")
                    return self._result
                await asyncio.sleep(0.05)
                await self.read()
        except asyncio.CancelledError:
            await asyncio.shield(self.cancel())
            raise


class ChildClient:
    """Create children only within active tool authority or a retained lease."""

    def __init__(self, client: HostRPCClient | None) -> None:
        self._client = client
        self._retained: set[str] = set()

    async def _persistent(self, method: str, params: Any) -> Any:
        call = getattr(self._client, "request_persistent", None)
        if callable(call):
            return await call(method, params)
        persistent = getattr(self._client, "persistent", None)
        if persistent is None:
            raise RuntimeError("Retained child execution requires persistent host RPC")
        return await persistent.request(method, params)

    async def start(self, *, profile: str, message: str, request_id: str | None = None,
                    options: ExecutionOptions | Mapping[str, Any] | None = None,
                    system_prompt: str | None = None, cwd: str | None = None,
                    lease: BackgroundTaskLease | None = None) -> ChildExecution:
        """Submit a preset with typed narrowing options, never credentials/config."""
        if self._client is None:
            raise RuntimeError("Central child execution unavailable; no local fallback")
        lease_id = lease.id if lease else None
        if lease is not None and not lease_id:
            raise RuntimeError("Child execution requires a real runner background lease")
        params: dict[str, Any] = {"requestId": request_id or str(uuid4()),
                                  "profile": profile, "message": message}
        if options is not None:
            params["options"] = ExecutionOptions.model_validate(options).to_wire()
        for name, value in {"systemPrompt": system_prompt, "cwd": cwd,
                            "leaseId": lease_id}.items():
            if value is not None:
                params[name] = value
        call = (self._persistent if lease_id in self._retained else self._client.request)
        result = _result(await call("kodelet.child.start", params))
        if lease_id:
            self._retained.add(lease_id)
        return ChildExecution(result, self._persistent if lease_id else self._client.request,
                              lease_id)

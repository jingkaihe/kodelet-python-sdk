"""Scoped delegated executions over the existing authenticated runner host RPC."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict, cast
from uuid import uuid4

from .execution import ExecutionOptions

if TYPE_CHECKING:
    from .context import BackgroundTaskLease, HostRPCClient


class _Unset:
    pass


_UNSET = _Unset()


class ChildSteerResult(TypedDict):
    """Guidance was queued, or the caller must submit an explicit next turn."""

    outcome: Literal["injected", "promptRequired"]
    reason: NotRequired[Literal["noRunningTurn"]]


def _identity(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 128 or "\x00" in value:
        raise ValueError(f"{name} must be a nonempty identity of at most 128 characters")
    return value


def _message(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 512 * 1024:
        raise ValueError("message must be nonempty and at most 512 KiB")
    return value


def _result(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("Invalid central child execution response")
    try:
        _identity(value.get("conversationId"), "conversationId")
        _identity(value.get("runId"), "runId")
    except ValueError as exc:
        raise RuntimeError("Invalid central child execution response") from exc
    if not isinstance(value.get("done"), bool):
        raise RuntimeError("Invalid central child execution status")
    return value


class ChildExecution:
    """A durable child identity with progress polling and exact cancellation."""

    def __init__(
        self, result: dict[str, Any], call: Callable[..., Awaitable[Any]], lease_id: str | None
    ) -> None:
        _result(result)
        self.conversation_id: str = result["conversationId"]
        self.run_id: str = result["runId"]
        self._result, self._call, self._lease_id = result, call, lease_id
        self._after = 0

    def _params(self) -> dict[str, Any]:
        return {
            "childId": self.conversation_id,
            "childRunId": self.run_id,
            **({"leaseId": self._lease_id} if self._lease_id else {}),
        }

    async def read(self) -> dict[str, Any]:
        """Read bounded progress after the last observed sequence."""
        result = _result(
            await self._call("kodelet.child.read", {**self._params(), "after": self._after})
        )
        if result["conversationId"] != self.conversation_id or result["runId"] != self.run_id:
            raise RuntimeError("Child response does not match this execution")
        self._result = result
        return self._result

    async def cancel(self) -> None:
        """Cancel only this child, not its parent or siblings."""
        await self._call("kodelet.child.cancel", self._params())

    async def steer(self, message: str, *, request_id: str | _Unset = _UNSET) -> ChildSteerResult:
        """Queue guidance for this exact run; never start another turn automatically.

        Supply a stable request_id to deduplicate an uncertain steering write.
        No retry is performed automatically.
        """
        params = {
            **self._params(),
            "message": _message(message),
            "requestId": str(uuid4())
            if request_id is _UNSET
            else _identity(request_id, "request_id"),
        }
        result = await self._call("kodelet.child.steer", params)
        if (
            not isinstance(result, dict)
            or result.get("outcome") not in ("injected", "promptRequired")
            or ("reason" in result and result["reason"] != "noRunningTurn")
            or result.keys() - {"outcome", "reason"}
        ):
            raise RuntimeError("Invalid central child steering response")
        return cast(ChildSteerResult, result)

    async def wait(
        self, *, on_event: Callable[[dict[str, Any]], Any] | None = None
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
        # A retained RPC may outlive a still-active originating handler. The
        # UI-oriented helper remains parent-scoped until that handler returns.
        persistent = getattr(self._client, "persistent", None)
        if persistent is not None:
            return await persistent.request(method, params)
        call = getattr(self._client, "request_persistent", None)
        if callable(call):
            return await call(method, params)
        raise RuntimeError("Retained child execution requires persistent host RPC")

    async def start(
        self,
        *,
        profile: str,
        message: str,
        request_id: str | _Unset = _UNSET,
        options: ExecutionOptions | Mapping[str, Any] | None = None,
        system_prompt: str | None = None,
        cwd: str | None = None,
        context_mode: Literal["fresh", "fork"] | _Unset = _UNSET,
        resume: str | _Unset = _UNSET,
        lease: BackgroundTaskLease | None = None,
    ) -> ChildExecution:
        """Submit a preset with typed narrowing options, never credentials/config.

        context_mode="fork" snapshots the active parent; omitted context is fresh.
        resume continues an owned child conversation with a new run/request ID and
        cannot be combined with fork. Cancellation before acknowledgement leaves
        admission uncertain; keep the stable request ID and lease for cleanup.
        """
        if self._client is None:
            raise RuntimeError("Central child execution unavailable; no local fallback")
        lease_id = lease.id if lease else None
        if lease is not None:
            try:
                _identity(lease_id, "lease_id")
            except ValueError as exc:
                raise RuntimeError(
                    "Child execution requires a real runner background lease"
                ) from exc
        params: dict[str, Any] = {
            "requestId": str(uuid4())
            if request_id is _UNSET
            else _identity(request_id, "request_id"),
            "profile": _identity(profile, "profile"),
            "message": _message(message),
        }
        if context_mode is not _UNSET:
            if context_mode not in ("fresh", "fork"):
                raise ValueError("context_mode must be fresh or fork")
            params["contextMode"] = context_mode
        if resume is not _UNSET:
            params["resume"] = _identity(resume, "resume")
            if context_mode == "fork":
                raise ValueError("Cannot combine resume with fork context")
        if options is not None:
            params["options"] = ExecutionOptions.model_validate(options).to_wire()
        if system_prompt is not None and (
            not isinstance(system_prompt, str) or len(system_prompt) > 256 * 1024
        ):
            raise ValueError("system_prompt must be text of at most 256 KiB")
        if cwd is not None and (
            not isinstance(cwd, str) or not cwd.strip() or len(cwd) > 8192 or "\x00" in cwd
        ):
            raise ValueError("cwd must be a nonempty path of at most 8192 characters")
        for name, value in {"systemPrompt": system_prompt, "cwd": cwd, "leaseId": lease_id}.items():
            if value is not None:
                params[name] = value
        # Fork always needs live originating-tool authority, even on a lease
        # already bound by an earlier child. Fresh/resume may use that lease.
        call = (
            self._persistent
            if lease_id in self._retained and context_mode != "fork"
            else self._client.request
        )
        result = _result(await call("kodelet.child.start", params))
        if resume is not _UNSET and result["conversationId"] != resume:
            raise RuntimeError("Child response does not match the resumed conversation")
        if lease_id:
            self._retained.add(lease_id)
        return ChildExecution(
            result, self._persistent if lease_id else self._client.request, lease_id
        )

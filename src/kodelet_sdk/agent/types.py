from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal, Protocol, TypeAlias, TypedDict, cast

from .._utils import AttrDict
from ..api import Entrypoint, Extension
from ..context import UIConfirmRequest, UIInputRequest, UINotifyRequest, UISelectRequest

ProfileInput: TypeAlias = Mapping[str, Any]
BridgeTransport: TypeAlias = Literal["unix", "tcp"]


class Profile:
    """Named or inline Kodelet profile used when launching agent sessions."""

    def __init__(self, config: str | ProfileInput) -> None:
        if isinstance(config, str):
            self.name: str | None = config
            self.config: dict[str, Any] = {"name": config}
            return

        normalized = dict(config)
        if normalized.get("provider") is None and isinstance(normalized.get("profiler"), str):
            normalized["provider"] = normalized["profiler"]
        normalized.pop("profiler", None)
        self.name = normalized.get("name") if isinstance(normalized.get("name"), str) else None
        self.config = normalized

    @classmethod
    def named(cls, name: str) -> Profile:
        """Create a profile reference that launches Kodelet with ``--profile``."""

        return cls(name)

    def is_named_only(self) -> bool:
        return all(key == "name" for key in self.config)

    def isNamedOnly(self) -> bool:
        """CamelCase alias for :meth:`is_named_only`."""

        return self.is_named_only()

    def to_launch_config(self) -> dict[str, Any]:
        """Return command-line args and optional inline config for this profile."""

        if self.name and self.is_named_only():
            return {"args": ["--profile", self.name]}
        return {"args": [], "config": dict(self.config)}

    def toLaunchConfig(self) -> dict[str, Any]:
        """CamelCase alias for :meth:`to_launch_config`."""

        return self.to_launch_config()


class AgentUIHandlers(TypedDict, total=False):
    """SDK-provided handlers for extension UI requests in agent sessions."""

    input: Callable[[UIInputRequest], Awaitable[str | None] | str | None]
    confirm: Callable[[UIConfirmRequest], Awaitable[bool] | bool]
    select: Callable[[UISelectRequest], Awaitable[str | None] | str | None]
    notify: Callable[[UINotifyRequest], Awaitable[None] | None]


class CreateSessionOptions(TypedDict, total=False):
    """Options accepted by :meth:`Client.create_session`."""

    profile: str | Profile | ProfileInput
    extensions: Sequence[Entrypoint | Extension]
    streaming: bool
    cwd: str
    resume: str
    max_turns: int
    extension_transport: BridgeTransport
    ui: AgentUIHandlers


class RunOptions(TypedDict, total=False):
    """Options accepted by :meth:`Session.run_and_wait`."""

    message: str
    images: Sequence[str]
    max_turns: int


class AssistantMessageDeltaData(TypedDict):
    deltaContent: str


class AssistantMessageData(TypedDict):
    content: str


class AssistantThinkingDeltaData(TypedDict):
    deltaContent: str


class ToolCallData(TypedDict, total=False):
    toolName: str
    input: Any
    rawInput: str
    toolCallId: str


class ToolResultData(TypedDict, total=False):
    toolName: str
    result: str
    toolCallId: str
    status: str


class SpawnOptions(TypedDict, total=False):
    cwd: str
    env: Mapping[str, str]
    stdio: Sequence[str]


class BinaryLineReader(Protocol):
    async def readline(self) -> bytes: ...


class BinaryWriter(Protocol):
    def write(self, data: bytes, /) -> object: ...

    def close(self) -> object: ...


class DrainableBinaryWriter(BinaryWriter, Protocol):
    async def drain(self) -> None: ...


class SpawnedProcess(Protocol):
    """Minimal subprocess protocol used by the ACP client."""

    stdin: BinaryWriter | None
    stdout: BinaryLineReader | None
    stderr: BinaryLineReader | None

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    async def wait(self) -> int: ...


SpawnFunction: TypeAlias = Callable[
    [str, Sequence[str], SpawnOptions], Awaitable[SpawnedProcess] | SpawnedProcess
]


class ClientOptions(TypedDict, total=False):
    """Options accepted by :class:`Client`."""

    command: str
    cwd: str | os.PathLike[str]
    env: Mapping[str, str | None]
    spawn: SpawnFunction


class AgentStreamEvent(AttrDict):
    """Event emitted by :class:`Session` while an agent run is in progress."""

    @property
    def conversation_id(self) -> str | None:
        return cast(str | None, self.get("conversationId"))


class AgentResponse(AttrDict):
    """Final response returned by :meth:`Session.run_and_wait`."""

    @property
    def conversation_id(self) -> str | None:
        return cast(str | None, self.get("conversationId"))

    @property
    def exit_code(self) -> int:
        return int(self.get("exitCode") or 0)

    @property
    def stop_reason(self) -> str | None:
        return cast(str | None, self.get("stopReason"))


class AgentRunError(RuntimeError):
    """Raised when the launched ``kodelet acp`` process exits unexpectedly."""

    def __init__(
        self,
        message: str,
        *,
        code: int | None,
        signal: str | None,
        stderr: str,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.signal = signal
        self.stderr = stderr


__all__ = [
    "AgentResponse",
    "AgentRunError",
    "AgentStreamEvent",
    "AgentUIHandlers",
    "AssistantMessageData",
    "AssistantMessageDeltaData",
    "AssistantThinkingDeltaData",
    "BinaryLineReader",
    "BinaryWriter",
    "BridgeTransport",
    "ClientOptions",
    "CreateSessionOptions",
    "DrainableBinaryWriter",
    "Profile",
    "ProfileInput",
    "RunOptions",
    "SpawnFunction",
    "SpawnOptions",
    "SpawnedProcess",
    "ToolCallData",
    "ToolResultData",
]

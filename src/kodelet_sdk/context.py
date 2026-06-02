from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from builtins import list as list_type
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NotRequired, Protocol, Required, TypeAlias, TypedDict, cast

CommandFlagValue: TypeAlias = str | bool | list[str]


class CommandInvocation(TypedDict):
    """User prompt metadata that invoked an extension command."""

    raw: str
    commandName: str
    args: list[str]
    flags: Mapping[str, CommandFlagValue]


UIInputStatus: TypeAlias = Literal["submitted", "dismissed", "timeout", "unavailable"]


class UIInputRequest(TypedDict):
    """User-input prompt request sent to the Kodelet host."""

    title: str
    id: NotRequired[str]
    helpText: NotRequired[str]
    message: NotRequired[str]
    placeholder: NotRequired[str]
    defaultValue: NotRequired[str]
    submitButtonText: NotRequired[str]
    cancelButtonText: NotRequired[str]
    required: NotRequired[bool]
    secret: NotRequired[bool]


class UIConfirmRequest(TypedDict):
    """Yes/no confirmation request sent to the Kodelet host."""

    title: str
    id: NotRequired[str]
    message: NotRequired[str]
    confirmButtonText: NotRequired[str]
    cancelButtonText: NotRequired[str]


class UISelectRequest(TypedDict):
    """Single-choice selection request sent to the Kodelet host."""

    title: str
    options: Sequence[str]
    id: NotRequired[str]
    message: NotRequired[str]
    submitButtonText: NotRequired[str]
    cancelButtonText: NotRequired[str]


class UINotifyRequest(TypedDict):
    """Fire-and-forget notification request sent to the Kodelet host."""

    message: str
    title: NotRequired[str]


class UIInputResponse(TypedDict, total=False):
    """Host response for UI input, confirmation, selection, and notification calls."""

    status: Required[UIInputStatus]
    value: str
    confirmed: bool
    reason: str


class HostRPCClient(Protocol):
    """Reverse-RPC client used by extension contexts to call the Kodelet host."""

    async def request(self, method: str, params: Any | None = None) -> Any: ...


_active_host_rpc_client: HostRPCClient | None = None


def set_active_host_rpc_client(client: HostRPCClient | None) -> None:
    """Set the process-global reverse-RPC client for context UI helpers.

    Args:
        client: Host RPC client to use, or ``None`` to disable host UI calls.
    """

    global _active_host_rpc_client
    _active_host_rpc_client = client


@dataclass(frozen=True)
class ExecResult:
    """Result returned by :meth:`ProcessContext.exec`.

    Attributes:
        stdout: Captured standard output as text.
        stderr: Captured standard error as text.
        exit_code: Process exit code. Non-zero values are returned instead of
            raising, matching the TypeScript SDK behavior.
    """

    stdout: str
    stderr: str
    exit_code: int

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-compatible object using Kodelet protocol field names."""

        return {"stdout": self.stdout, "stderr": self.stderr, "exitCode": self.exit_code}


class StorageContext:
    """Async file storage scoped to the extension data directory."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = str(data_dir)
        self._data_dir = data_dir

    async def read_text(self, path: str) -> str | None:
        """Read a UTF-8 text file from extension storage.

        Args:
            path: Relative path under ``data_dir``.

        Returns:
            File contents, or ``None`` when the file does not exist.

        Raises:
            ValueError: If ``path`` escapes the extension storage directory.
        """

        resolved = _resolve_inside(self._data_dir, path, "extension storage")
        try:
            return await asyncio.to_thread(resolved.read_text, encoding="utf-8")
        except FileNotFoundError:
            return None

    async def write_text(self, path: str, content: str) -> None:
        """Write UTF-8 text to extension storage, creating parents as needed.

        Args:
            path: Relative path under ``data_dir``.
            content: Text content to write.
        """

        resolved = _resolve_inside(self._data_dir, path, "extension storage")
        await asyncio.to_thread(resolved.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(resolved.write_text, content, encoding="utf-8")

    async def read_json(self, path: str) -> Any | None:
        """Read and decode JSON from extension storage.

        Args:
            path: Relative path under ``data_dir``.

        Returns:
            Decoded JSON value, or ``None`` if the file does not exist.
        """

        content = await self.read_text(path)
        if content is None:
            return None
        return json.loads(content)

    async def write_json(self, path: str, value: Any) -> None:
        """Encode and write a JSON value to extension storage.

        Args:
            path: Relative path under ``data_dir``.
            value: JSON-serializable value to write.
        """

        await self.write_text(path, f"{json.dumps(value, indent=2)}\n")


class PathContext:
    """Path helpers rooted at the active workspace directory."""

    def __init__(self, cwd: Path) -> None:
        self._cwd = cwd

    def resolve_workspace_path(self, path: str) -> str:
        """Resolve a path inside the workspace.

        Args:
            path: Relative path under the workspace. Empty strings resolve to
                the workspace root.

        Returns:
            Absolute path string.

        Raises:
            ValueError: If ``path`` escapes the workspace directory.
        """

        return str(_resolve_inside(self._cwd, path or ".", "workspace"))

    def relative_to_workspace(self, path: str) -> str:
        """Return ``path`` relative to the workspace root.

        Args:
            path: Absolute or workspace-relative path.

        Returns:
            Relative path string, or ``"."`` for the workspace root.
        """

        target = Path(path)
        if not target.is_absolute():
            target = self._cwd / target
        relative = os.path.relpath(target.resolve(strict=False), self._cwd)
        return "." if relative == "." else relative


class FileSystemContext:
    """Async file-system helpers for workspace files."""

    def __init__(self, cwd: Path) -> None:
        self._cwd = cwd

    async def exists(self, path: str) -> bool:
        """Return whether a file-system path exists.

        Relative paths are resolved inside the workspace; absolute paths are
        used as-is.
        """

        return await asyncio.to_thread(_resolve_fs_path(self._cwd, path).exists)

    async def read_text(self, path: str) -> str:
        """Read a UTF-8 text file.

        Args:
            path: Absolute path or relative workspace path.
        """

        resolved = _resolve_fs_path(self._cwd, path)
        return await asyncio.to_thread(resolved.read_text, encoding="utf-8")

    async def write_text(self, path: str, content: str) -> None:
        """Write UTF-8 text, creating parent directories as needed.

        Args:
            path: Absolute path or relative workspace path.
            content: Text content to write.
        """

        resolved = _resolve_fs_path(self._cwd, path)
        await asyncio.to_thread(resolved.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(resolved.write_text, content, encoding="utf-8")

    async def list(self, path: str) -> list_type[FileInfo]:
        """List directory entries.

        Args:
            path: Absolute path or relative workspace path.

        Returns:
            A list of ``{"name", "path", "type"}`` dictionaries where type is
            ``"file"``, ``"dir"``, or ``"other"``.
        """

        resolved = _resolve_fs_path(self._cwd, path)
        entries = await asyncio.to_thread(lambda: list(resolved.iterdir()))
        return [
            {
                "name": entry.name,
                "path": str(entry),
                "type": "file" if entry.is_file() else "dir" if entry.is_dir() else "other",
            }
            for entry in entries
        ]


class FileInfo(TypedDict):
    """File-system entry returned by :meth:`FileSystemContext.list`."""

    name: str
    path: str
    type: Literal["file", "dir", "other"]


class ProcessContext:
    """Async process helpers rooted at the active workspace directory."""

    def __init__(self, cwd: Path) -> None:
        self._cwd = cwd

    async def exec(
        self,
        command: str,
        args: Sequence[str] | None = None,
        opts: ProcessExecOptions | None = None,
    ) -> ExecResult:
        """Run a process and capture output.

        Args:
            command: Executable name or path.
            args: Optional argument sequence.
            opts: Optional execution options. Supported keys are ``cwd``
                (workspace-relative working directory) and ``timeoutMs``.

        Returns:
            Captured stdout, stderr, and exit code. Non-zero exits do not raise.
        """

        opts = opts or {}
        cwd = _option_cwd(self._cwd, opts)
        timeout = opts.get("timeoutMs")
        timeout_sec = float(timeout) / 1000 if timeout is not None else None
        process = await asyncio.create_subprocess_exec(
            command,
            *(args or []),
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout_sec)
        except TimeoutError:
            process.kill()
            stdout_bytes, stderr_bytes = await process.communicate()
            return ExecResult(
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
                exit_code=1,
            )
        return ExecResult(
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            exit_code=process.returncode if process.returncode is not None else 1,
        )

    async def spawn(
        self,
        command: str,
        args: Sequence[str] | None = None,
        opts: ProcessSpawnOptions | None = None,
    ) -> None:
        """Spawn a process.

        Args:
            command: Executable name or path.
            args: Optional argument sequence.
            opts: Optional execution options. Supported keys are ``cwd`` and
                ``detach``. Detached processes are started in a new session and
                this method returns after spawn.

        Raises:
            RuntimeError: For non-detached processes that exit non-zero.
        """

        opts = opts or {}
        cwd = _option_cwd(self._cwd, opts)
        if opts.get("detach"):
            await asyncio.to_thread(
                subprocess.Popen,
                [command, *(args or [])],
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return
        process = await asyncio.create_subprocess_exec(command, *(args or []), cwd=str(cwd))
        exit_code = await process.wait()
        if exit_code != 0:
            raise RuntimeError(f"{command} exited with status {exit_code}")


class ProcessExecOptions(TypedDict, total=False):
    """Options for :meth:`ProcessContext.exec`."""

    cwd: str
    timeoutMs: int | float


class ProcessSpawnOptions(TypedDict, total=False):
    """Options for :meth:`ProcessContext.spawn`."""

    cwd: str
    detach: bool


class EnvContext:
    """Read-only environment variable access."""

    def get(self, name: str) -> str | None:
        """Return an environment variable value, or ``None`` if unset."""

        return os.environ.get(name)


class LogContext:
    """Structured JSON logger that writes to stderr."""

    def __init__(self, extension_id: str | None) -> None:
        self._extension_id = extension_id

    def debug(self, message: str, fields: Mapping[str, Any] | None = None) -> None:
        """Write a debug log message."""

        self._write("debug", message, fields)

    def info(self, message: str, fields: Mapping[str, Any] | None = None) -> None:
        """Write an info log message."""

        self._write("info", message, fields)

    def warn(self, message: str, fields: Mapping[str, Any] | None = None) -> None:
        """Write a warning log message."""

        self._write("warn", message, fields)

    def error(self, message: str, fields: Mapping[str, Any] | None = None) -> None:
        """Write an error log message."""

        self._write("error", message, fields)

    def _write(self, level: str, message: str, fields: Mapping[str, Any] | None) -> None:
        payload = {"level": level, "extension": self._extension_id, "message": message}
        if fields:
            payload.update(fields)
        print(json.dumps(payload), file=sys.stderr, flush=True)


class UIContext:
    """Host UI helpers backed by Kodelet reverse-RPC methods."""

    async def input(self, request: UIInputRequest) -> str | None:
        """Ask the host for text input.

        Args:
            request: UI input request. ``title`` is required; optional fields
                include ``message``, ``placeholder``, ``required``, and
                ``secret``.

        Returns:
            Submitted text, or ``None`` if no host client is available or the
            request was cancelled.
        """

        if _active_host_rpc_client is None:
            return None
        result = await _active_host_rpc_client.request("kodelet.ui.input", dict(request))
        if isinstance(result, Mapping) and result.get("status") == "submitted":
            value = result.get("value")
            if isinstance(value, str):
                return value
        return None

    async def confirm(self, request: UIConfirmRequest) -> bool:
        """Ask the host for confirmation.

        Args:
            request: UI confirmation request. ``title`` is required.

        Returns:
            ``True`` only when the host returns a submitted positive response.
        """

        if _active_host_rpc_client is None:
            return False
        result = await _active_host_rpc_client.request("kodelet.ui.confirm", dict(request))
        return (
            isinstance(result, Mapping)
            and result.get("status") == "submitted"
            and result.get("confirmed") is True
        )

    async def select(self, request: UISelectRequest) -> str | None:
        """Ask the host to select one option.

        Args:
            request: UI select request containing required ``title`` and
                ``options``.

        Returns:
            Selected option value, or ``None`` if unavailable/cancelled.
        """

        if _active_host_rpc_client is None:
            return None
        result = await _active_host_rpc_client.request("kodelet.ui.select", dict(request))
        if isinstance(result, Mapping) and result.get("status") == "submitted":
            value = result.get("value")
            if isinstance(value, str):
                return value
        return None

    async def notify(self, request: str | UINotifyRequest) -> None:
        """Send a notification to the host UI.

        Args:
            request: Either a message string or notification request mapping.
        """

        if _active_host_rpc_client is None:
            return
        payload = {"message": request} if isinstance(request, str) else dict(request)
        await _active_host_rpc_client.request("kodelet.ui.notify", payload)


class SharedContext:
    """Common context passed to tools, commands, and event handlers.

    Attributes mirror the Kodelet call context and include helper namespaces:
    ``storage``, ``path``, ``fs``, ``process``, ``env``, ``log``, and ``ui``.
    """

    def __init__(
        self,
        init: Mapping[str, Any] | None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        context = context or {}
        extension = _extension_info(init)
        cwd = Path(str(context.get("cwd") or extension.get("cwd") or os.getcwd())).resolve(
            strict=False
        )
        data_dir_value = extension.get("dataDir") or _default_data_dir(
            str(extension.get("id") or "extension")
        )
        data_dir = Path(str(data_dir_value)).resolve(strict=False)
        self.session_id = _optional_str(context.get("sessionId"))
        self.conversation_id = _optional_str(context.get("conversationId"))
        self.cwd = str(cwd)
        self.provider = _optional_str(context.get("provider"))
        self.model = _optional_str(context.get("model"))
        self.profile = _optional_str(context.get("profile"))
        self.recipe_name = _optional_str(context.get("recipeName"))
        self.invoked_by = _optional_str(context.get("invokedBy"))
        self.storage = StorageContext(data_dir)
        self.path = PathContext(cwd)
        self.fs = FileSystemContext(cwd)
        self.process = ProcessContext(cwd)
        self.env = EnvContext()
        self.log = LogContext(_optional_str(extension.get("id")))
        self.ui = UIContext()


class ToolContext(SharedContext):
    """Context passed to tool handlers."""

    pass


class EventContext(SharedContext):
    """Context passed to event handlers."""

    pass


class CommandContext(SharedContext):
    """Context passed to command handlers.

    Attributes:
        input: Raw command invocation metadata containing ``raw``,
            ``commandName``, ``args``, and ``flags``.
    """

    def __init__(
        self,
        init: Mapping[str, Any] | None,
        context: Mapping[str, Any] | None,
        invocation: Mapping[str, Any],
    ) -> None:
        super().__init__(init, context)
        self.input = cast(CommandInvocation, invocation)


def create_tool_context(
    init: Mapping[str, Any] | None,
    context: Mapping[str, Any] | None = None,
) -> ToolContext:
    return ToolContext(init, context)


def create_command_context(
    init: Mapping[str, Any] | None,
    context: Mapping[str, Any] | None,
    invocation: Mapping[str, Any],
) -> CommandContext:
    return CommandContext(init, context, invocation)


def create_event_context(
    init: Mapping[str, Any] | None,
    context: Mapping[str, Any] | None = None,
) -> EventContext:
    return EventContext(init, context)


def _extension_info(init: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not init:
        return {}
    extension = init.get("extension")
    return extension if isinstance(extension, Mapping) else {}


def _default_data_dir(extension_id: str) -> str:
    return str(Path.home() / ".kodelet" / "extensions" / "data" / extension_id)


def _resolve_inside(parent: Path, target: str, label: str) -> Path:
    resolved_parent = parent.resolve(strict=False)
    resolved = (resolved_parent / (target or ".")).resolve(strict=False)
    try:
        common = os.path.commonpath([resolved, resolved_parent])
    except ValueError as exc:
        raise ValueError(f"Path escapes {label}: {target}") from exc
    if common != str(resolved_parent):
        raise ValueError(f"Path escapes {label}: {target}")
    return resolved


def _resolve_fs_path(cwd: Path, target: str) -> Path:
    path = Path(target)
    if path.is_absolute():
        return path.resolve(strict=False)
    return _resolve_inside(cwd, target, "workspace")


def _option_cwd(cwd: Path, opts: Mapping[str, Any]) -> Path:
    option = opts.get("cwd")
    if option is None:
        return cwd
    return _resolve_inside(cwd, str(option), "workspace")


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None

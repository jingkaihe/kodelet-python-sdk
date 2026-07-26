from __future__ import annotations

import asyncio
import contextvars
import inspect
import json
import os
import subprocess
import sys
import weakref
from builtins import list as list_type
from collections.abc import Awaitable, Callable, Mapping, Sequence
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


class UITranscriptAppendRequest(TypedDict):
    """Persistent informational transcript entry sent to the Kodelet host."""

    message: str
    title: NotRequired[str]


UIWidgetPlacement: TypeAlias = Literal["aboveComposer", "belowComposer"]


class UIWidgetOptions(TypedDict, total=False):
    """Optional placement settings for a persistent passive widget."""

    placement: UIWidgetPlacement


class UIStyle(TypedDict, total=False):
    """Optional style applied to one span in a persistent UI frame."""

    foreground: str
    background: str
    bold: bool
    dim: bool
    italic: bool
    underline: bool
    strikethrough: bool
    reverse: bool


class UIStyledSpan(TypedDict):
    """Text and optional styling for part of a persistent UI line."""

    text: str
    style: NotRequired[UIStyle]


class UIStyledLine(TypedDict):
    """A persistent UI line composed of styled spans."""

    spans: list[UIStyledSpan]


UIFrameLine: TypeAlias = str | UIStyledLine
UISizeValue: TypeAlias = int | str
UISurfaceAnchor: TypeAlias = Literal[
    "topLeft",
    "top",
    "topRight",
    "left",
    "center",
    "right",
    "bottomLeft",
    "bottom",
    "bottomRight",
]


class UIMargin(TypedDict, total=False):
    """Optional terminal-cell margins around an interactive surface."""

    top: int
    right: int
    bottom: int
    left: int


class UISurfaceOpenOptions(TypedDict):
    """Options for opening a persistent interactive host surface."""

    id: str
    initialLines: NotRequired[list[UIFrameLine]]
    width: NotRequired[UISizeValue]
    height: NotRequired[UISizeValue]
    maxWidth: NotRequired[UISizeValue]
    maxHeight: NotRequired[UISizeValue]
    anchor: NotRequired[UISurfaceAnchor]
    offsetX: NotRequired[int]
    offsetY: NotRequired[int]
    margin: NotRequired[UIMargin]
    nonCapturing: NotRequired[bool]


class UISurfaceSize(TypedDict):
    """Current interactive surface size in terminal cells."""

    width: int
    height: int


class UISurfaceMouseEvent(TypedDict):
    """Mouse details attached to an interactive surface input event."""

    x: int
    y: int
    button: str
    action: str
    shift: NotRequired[bool]
    alt: NotRequired[bool]
    ctrl: NotRequired[bool]


class UISurfaceInputEvent(TypedDict):
    """Key, mouse, focus, or blur event delivered to a surface."""

    sequence: int
    kind: Literal["key", "mouse", "focus", "blur"]
    id: NotRequired[str]
    key: NotRequired[str]
    text: NotRequired[str]
    alt: NotRequired[bool]
    shift: NotRequired[bool]
    ctrl: NotRequired[bool]
    mouse: NotRequired[UISurfaceMouseEvent]


class UISurfaceResizeEvent(UISurfaceSize):
    """Ordered resize event delivered to a surface."""

    sequence: int


class UISurface(Protocol):
    """Persistent interactive UI surface owned by an extension connection."""

    @property
    def id(self) -> str: ...

    @property
    def size(self) -> UISurfaceSize | None: ...

    def update(self, lines: list[UIFrameLine]) -> None: ...

    async def close(self) -> None: ...

    def on_input(
        self,
        handler: Callable[[UISurfaceInputEvent], None],
    ) -> Callable[[], None]: ...

    def on_resize(
        self,
        handler: Callable[[UISurfaceResizeEvent], None],
    ) -> Callable[[], None]: ...


class ToolUpdateRequest(TypedDict):
    """Accumulated tool-result snapshot sent to the Kodelet host."""

    content: str
    data: NotRequired[Mapping[str, Any]]


class UIInputResponse(TypedDict, total=False):
    """Host response for UI input, confirmation, selection, and notification calls."""

    status: Required[UIInputStatus]
    value: str
    confirmed: bool
    reason: str


class HostRPCClient(Protocol):
    """Reverse-RPC client used by extension contexts to call the Kodelet host.

    Clients may additionally expose ``persistent``, ``notify``, and
    ``on_notification`` attributes. Persistent UI helpers discover those
    optional capabilities dynamically so simple request-only test doubles stay
    valid implementations of this protocol.
    """

    async def request(self, method: str, params: Any | None = None) -> Any: ...


@dataclass
class _PersistentUIState:
    widget_sequences: dict[str, int]
    surface_sequences: dict[str, int]
    surfaces: dict[str, _UISurfaceHandle]
    notification_routing_installed: bool = False


_PERSISTENT_UI_STATE_ATTR = "_kodelet_sdk_persistent_ui_state"
_persistent_ui_states_by_id: dict[
    int,
    tuple[weakref.ReferenceType[Any], _PersistentUIState],
] = {}
_HOST_RPC_CLIENT_UNSET = object()


_active_host_rpc_client: HostRPCClient | None = None
_host_rpc_client_context: contextvars.ContextVar[HostRPCClient | None | object] = (
    contextvars.ContextVar(
        "kodelet_sdk_host_rpc_client",
        default=_HOST_RPC_CLIENT_UNSET,
    )
)


def set_active_host_rpc_client(client: HostRPCClient | None) -> None:
    """Set the process-global reverse-RPC client for context UI helpers.

    Args:
        client: Host RPC client to use, or ``None`` to disable host UI calls.
    """

    global _active_host_rpc_client
    _active_host_rpc_client = client
    _ensure_persistent_notification_routing(_persistent_host_rpc_client(client))


async def run_with_host_rpc_client(
    client: HostRPCClient | None,
    func: Callable[[], Awaitable[Any] | Any],
) -> Any:
    """Run ``func`` with a task-local reverse-RPC client.

    Runtimes and test harnesses use task-local routing while a handler is being
    dispatched. The process-global setter remains available for custom hosts.
    """

    token = _host_rpc_client_context.set(client)
    try:
        result = func()
        if inspect.isawaitable(result):
            return await result
        return result
    finally:
        _host_rpc_client_context.reset(token)


def _current_host_rpc_client() -> HostRPCClient | None:
    client = _host_rpc_client_context.get()
    if client is _HOST_RPC_CLIENT_UNSET:
        return _active_host_rpc_client
    return cast(HostRPCClient | None, client)


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

    def __init__(
        self,
        init: Mapping[str, Any] | None = None,
        client: HostRPCClient | None | object = _HOST_RPC_CLIENT_UNSET,
    ) -> None:
        resolved_client = (
            _current_host_rpc_client()
            if client is _HOST_RPC_CLIENT_UNSET
            else cast(HostRPCClient | None, client)
        )
        self._init = init
        self._client = resolved_client
        self._persistent_client = _persistent_host_rpc_client(resolved_client)

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

        client = self._client
        if client is None:
            return None
        result = await client.request("kodelet.ui.input", dict(request))
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

        client = self._client
        if client is None:
            return False
        result = await client.request("kodelet.ui.confirm", dict(request))
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

        client = self._client
        if client is None:
            return None
        result = await client.request("kodelet.ui.select", dict(request))
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

        client = self._client
        if client is None:
            return
        payload = {"message": request} if isinstance(request, str) else dict(request)
        await client.request("kodelet.ui.notify", payload)

    async def append_transcript(self, request: str | UITranscriptAppendRequest) -> None:
        """Append a persistent informational entry to the host transcript.

        The call is a no-op when the host does not advertise transcript support.

        Args:
            request: Message string or protocol-shaped transcript entry.
        """

        client = self._persistent_client
        if not _extension_ui_supported(self._init, "transcript") or client is None:
            return
        payload = {"message": request} if isinstance(request, str) else dict(request)
        await client.request("kodelet.ui.transcript.append", payload)

    async def set_widget(
        self,
        id: str,
        lines: list[UIFrameLine] | None,
        options: UIWidgetOptions | None = None,
    ) -> None:
        """Create, replace, move, or remove a persistent passive widget.

        Reusing ``id`` replaces the widget. Passing ``None`` for ``lines``
        removes it. The call is a no-op when widgets are unsupported.

        Args:
            id: Extension-scoped widget identifier.
            lines: Plain or styled frame lines, or ``None`` to remove.
            options: Optional mapping containing ``placement``.
        """

        client = self._persistent_client
        if not _extension_ui_supported(self._init, "widgets") or client is None:
            return
        object_id = _validate_ui_object_id(id)
        normalized_lines = None if lines is None else _normalize_frame_lines(lines)
        sequence = _next_client_sequence(client, object_id, surface=False)
        if normalized_lines is None:
            await client.request(
                "kodelet.ui.widget.remove",
                {"id": object_id, "sequence": sequence},
            )
            return
        placement = (options or {}).get("placement", "aboveComposer")
        await client.request(
            "kodelet.ui.widget.set",
            {
                "id": object_id,
                "placement": placement,
                "frame": {"sequence": sequence, "lines": normalized_lines},
            },
        )

    async def open_surface(self, options: UISurfaceOpenOptions) -> UISurface:
        """Open a persistent interactive surface in a capable host.

        The returned handle remains usable after the opening handler returns.

        Args:
            options: Surface ID, initial frame, and optional layout settings.

        Raises:
            RuntimeError: If surfaces are unavailable, the ID is already owned,
                or the host rejects the open request.
            ValueError: If the surface ID is invalid.
        """

        client = self._persistent_client
        if not _extension_ui_supported(self._init, "surfaces") or client is None:
            raise RuntimeError("Interactive extension surfaces are not available in this host")

        surface_options = dict(options)
        requested_id = cast(str, surface_options.pop("id"))
        initial_lines = _normalize_frame_lines(
            surface_options.pop("initialLines", []),
        )
        object_id = _validate_ui_object_id(requested_id)
        state = _persistent_ui_state(client)
        if object_id in state.surfaces:
            raise RuntimeError(
                f'Interactive surface "{object_id}" is already open, opening, or closing; '
                "close it before reusing the ID"
            )

        surface = _UISurfaceHandle(object_id, client)
        state.surfaces[object_id] = surface
        _ensure_persistent_notification_routing(client)
        try:
            response = await client.request(
                "kodelet.ui.surface.open",
                {
                    "id": object_id,
                    "options": surface_options,
                    "frame": {
                        "sequence": surface._next_sequence(),
                        "lines": initial_lines,
                    },
                },
            )
            if isinstance(response, Mapping) and response.get("accepted") is False:
                reason = response.get("reason")
                raise RuntimeError(
                    reason
                    if isinstance(reason, str)
                    else "The host rejected the interactive surface"
                )
        except BaseException:
            if state.surfaces.get(object_id) is surface:
                state.surfaces.pop(object_id, None)
            raise
        surface._activate()
        return surface


class _UISurfaceHandle:
    def __init__(self, id: str, client: HostRPCClient) -> None:
        self.id = id
        try:
            self._client_ref: weakref.ReferenceType[Any] | None = weakref.ref(client)
            self._strong_client: HostRPCClient | None = None
        except TypeError:
            self._client_ref = None
            self._strong_client = client
        self._closed = False
        self._active = False
        self._pending_lines: list[UIFrameLine] | None = None
        self._frame_scheduled = False
        self._frame_in_flight = False
        self._frame_task: asyncio.Task[None] | None = None
        self._latest_event_sequence: int | float = 0
        self._input_handlers: set[Callable[[UISurfaceInputEvent], None]] = set()
        self._pending_focus_event: UISurfaceInputEvent | None = None
        self._resize_handlers: set[Callable[[UISurfaceResizeEvent], None]] = set()
        self._pending_resize_event: UISurfaceResizeEvent | None = None
        self._current_size: UISurfaceSize | None = None

    @property
    def size(self) -> UISurfaceSize | None:
        """Return the latest allocated size, if the host has reported one."""

        return (
            cast(UISurfaceSize, dict(self._current_size))
            if self._current_size is not None
            else None
        )

    def update(self, lines: list[UIFrameLine]) -> None:
        """Queue a replacement frame without blocking the caller."""

        if self._closed:
            return
        self._pending_lines = _normalize_frame_lines(lines)
        self._schedule_frame_flush()

    async def close(self) -> None:
        """Close the surface and release its ID after the host acknowledges it."""

        if self._closed:
            return
        self._clear_local_state()
        client = self._resolve_client()
        state = _find_persistent_ui_state(client) if client is not None else None
        try:
            if self._active and client is not None:
                await client.request(
                    "kodelet.ui.surface.close",
                    {"id": self.id, "sequence": self._next_sequence()},
                )
        finally:
            if state is not None and state.surfaces.get(self.id) is self:
                state.surfaces.pop(self.id, None)

    def on_input(
        self,
        handler: Callable[[UISurfaceInputEvent], None],
    ) -> Callable[[], None]:
        """Subscribe to ordered key, mouse, focus, and blur events."""

        self._input_handlers.add(handler)
        pending = self._pending_focus_event
        self._pending_focus_event = None
        if pending is not None:
            handler(pending)

        def unsubscribe() -> None:
            self._input_handlers.discard(handler)

        return unsubscribe

    def on_resize(
        self,
        handler: Callable[[UISurfaceResizeEvent], None],
    ) -> Callable[[], None]:
        """Subscribe to ordered surface resize events."""

        self._resize_handlers.add(handler)
        pending = self._pending_resize_event
        self._pending_resize_event = None
        if pending is not None:
            handler(pending)

        def unsubscribe() -> None:
            self._resize_handlers.discard(handler)

        return unsubscribe

    def _activate(self) -> None:
        self._active = True

    def _next_sequence(self) -> int:
        client = self._resolve_client()
        if client is None:
            raise RuntimeError("Extension host connection is closed")
        return _next_client_sequence(client, self.id, surface=True)

    def _resolve_client(self) -> HostRPCClient | None:
        if self._client_ref is not None:
            return cast(HostRPCClient | None, self._client_ref())
        return self._strong_client

    def _clear_local_state(self) -> None:
        self._closed = True
        self._pending_lines = None
        self._input_handlers.clear()
        self._pending_focus_event = None
        self._resize_handlers.clear()
        self._pending_resize_event = None

    def _disconnect(self) -> None:
        self._clear_local_state()
        frame_task = self._frame_task
        if frame_task is not None and not frame_task.done():
            try:
                current_task = asyncio.current_task()
            except RuntimeError:
                current_task = None
            if frame_task is not current_task:
                frame_task.cancel()
        self._strong_client = None

    def _schedule_frame_flush(self) -> None:
        if (
            self._closed
            or self._frame_scheduled
            or self._frame_in_flight
            or self._pending_lines is None
        ):
            return
        self._frame_scheduled = True
        asyncio.get_running_loop().call_soon(self._start_frame_flush)

    def _start_frame_flush(self) -> None:
        self._frame_scheduled = False
        if self._closed or self._frame_in_flight or self._pending_lines is None:
            return
        lines = self._pending_lines
        self._pending_lines = None
        self._frame_in_flight = True
        self._frame_task = asyncio.create_task(self._flush_frame(lines))

    async def _flush_frame(self, lines: list[UIFrameLine]) -> None:
        try:
            if self._closed:
                return
            client = self._resolve_client()
            if client is None:
                return
            params = {
                "id": self.id,
                "frame": {"sequence": self._next_sequence(), "lines": lines},
            }
            notify = getattr(client, "notify", None)
            if callable(notify):
                result = notify("kodelet.ui.surface.frame", params)
                if inspect.isawaitable(result):
                    await result
            else:
                await client.request("kodelet.ui.surface.frame", params)
        except Exception:
            # Process cleanup removes the surface when the host connection is gone.
            pass
        finally:
            self._frame_in_flight = False
            self._schedule_frame_flush()

    def _handle_notification(self, method: str, params: Any) -> None:
        if self._closed or not isinstance(params, Mapping) or params.get("id") != self.id:
            return

        input_event = method == "extension.ui.surface.input" and params.get("kind") in {
            "key",
            "mouse",
            "focus",
            "blur",
        }
        width = params.get("width")
        height = params.get("height")
        resize_event = (
            method == "extension.ui.surface.resize"
            and _is_number(width)
            and _is_number(height)
        )
        if not input_event and not resize_event:
            return

        sequence = params.get("sequence")
        if not _is_number(sequence) or sequence <= self._latest_event_sequence:
            return
        self._latest_event_sequence = sequence

        if input_event:
            event = cast(UISurfaceInputEvent, dict(params))
            if not self._input_handlers:
                if not self._active and event["kind"] in {"focus", "blur"}:
                    self._pending_focus_event = event
                return
            for handler in list(self._input_handlers):
                handler(event)
            return

        event = cast(
            UISurfaceResizeEvent,
            {"sequence": sequence, "width": width, "height": height},
        )
        self._current_size = cast(UISurfaceSize, {"width": width, "height": height})
        if not self._resize_handlers:
            self._pending_resize_event = None if self._active else event
            return
        for handler in list(self._resize_handlers):
            handler(event)


def _persistent_host_rpc_client(client: HostRPCClient | None) -> HostRPCClient | None:
    if client is None:
        return None
    persistent = getattr(client, "persistent", None)
    return cast(HostRPCClient, persistent) if persistent is not None else client


def _find_persistent_ui_state(client: HostRPCClient | None) -> _PersistentUIState | None:
    if client is None:
        return None
    try:
        state = getattr(client, _PERSISTENT_UI_STATE_ATTR)
    except AttributeError:
        state = None
    if isinstance(state, _PersistentUIState):
        return state

    client_id = id(client)
    entry = _persistent_ui_states_by_id.get(client_id)
    if entry is None:
        return None
    client_ref, state = entry
    if client_ref() is client:
        return state
    _persistent_ui_states_by_id.pop(client_id, None)
    return None


def _persistent_ui_state(client: HostRPCClient) -> _PersistentUIState:
    state = _find_persistent_ui_state(client)
    if state is not None:
        return state

    state = _PersistentUIState(widget_sequences={}, surface_sequences={}, surfaces={})
    try:
        setattr(client, _PERSISTENT_UI_STATE_ATTR, state)
        return state
    except (AttributeError, TypeError):
        pass

    client_id = id(client)

    def remove_client_state(client_ref: weakref.ReferenceType[Any]) -> None:
        entry = _persistent_ui_states_by_id.get(client_id)
        if entry is not None and entry[0] is client_ref:
            _persistent_ui_states_by_id.pop(client_id, None)

    try:
        client_ref = weakref.ref(client, remove_client_state)
    except TypeError as exc:
        raise TypeError(
            "Host RPC clients must support private attributes or weak references"
        ) from exc
    _persistent_ui_states_by_id[client_id] = (client_ref, state)
    return state


def _release_persistent_ui_state(client: HostRPCClient) -> None:
    state = _find_persistent_ui_state(client)
    if state is None:
        return
    for surface in list(state.surfaces.values()):
        surface._disconnect()
    state.surfaces.clear()
    state.widget_sequences.clear()
    state.surface_sequences.clear()
    state.notification_routing_installed = False

    try:
        if getattr(client, _PERSISTENT_UI_STATE_ATTR) is state:
            delattr(client, _PERSISTENT_UI_STATE_ATTR)
    except (AttributeError, TypeError):
        pass
    entry = _persistent_ui_states_by_id.get(id(client))
    if entry is not None and entry[0]() is client and entry[1] is state:
        _persistent_ui_states_by_id.pop(id(client), None)


def _ensure_persistent_notification_routing(client: HostRPCClient | None) -> None:
    if client is None:
        return
    on_notification = getattr(client, "on_notification", None)
    if not callable(on_notification):
        return
    state = _persistent_ui_state(client)
    if state.notification_routing_installed:
        return
    state.notification_routing_installed = True

    try:
        client_ref: weakref.ReferenceType[Any] | None = weakref.ref(client)
    except TypeError:
        client_ref = None

    def route(method: str, params: Any) -> None:
        if not isinstance(params, Mapping):
            return
        object_id = params.get("id")
        if not isinstance(object_id, str):
            return
        routed_client = cast(HostRPCClient | None, client_ref()) if client_ref else client
        routed_state = _find_persistent_ui_state(routed_client)
        if routed_state is None:
            return
        surface = routed_state.surfaces.get(object_id)
        if surface is not None:
            surface._handle_notification(method, params)

    try:
        on_notification(route)
    except Exception:
        state.notification_routing_installed = False
        raise


def _next_client_sequence(
    client: HostRPCClient,
    object_id: str,
    *,
    surface: bool,
) -> int:
    state = _persistent_ui_state(client)
    sequences = state.surface_sequences if surface else state.widget_sequences
    sequence = sequences.get(object_id, 0) + 1
    sequences[object_id] = sequence
    return sequence


def _normalize_frame_lines(lines: Any) -> list[UIFrameLine]:
    if not isinstance(lines, list):
        raise TypeError("UI frame lines must be a list")
    return cast(list[UIFrameLine], list(lines))


def _validate_ui_object_id(object_id: str) -> str:
    if object_id.strip() == "":
        raise ValueError("Extension UI id is required")
    if object_id != object_id.strip():
        raise ValueError("Extension UI id must not have leading or trailing whitespace")
    if len(object_id.encode("utf-8")) > 128:
        raise ValueError("Extension UI id is too long")
    return object_id


def _extension_ui_supported(
    init: Mapping[str, Any] | None,
    feature: Literal["widgets", "surfaces", "transcript"],
) -> bool:
    if init is None:
        return False
    capabilities = init.get("capabilities")
    if not isinstance(capabilities, Mapping):
        return False
    ui = capabilities.get("ui")
    return isinstance(ui, Mapping) and ui.get(feature) is True


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


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
        self._host_rpc_client = _current_host_rpc_client()
        self.ui = UIContext(init, self._host_rpc_client)


class ToolContext(SharedContext):
    """Context passed to tool handlers."""

    def __init__(
        self,
        init: Mapping[str, Any] | None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(init, context)
        self._tool_updates_enabled = _tool_updates_supported(init)

    async def update(
        self,
        content: str,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        """Publish the latest accumulated result snapshot for this tool call.

        Updates are transient and replace the previous snapshot for the active
        tool call. They are sent only when the Kodelet host advertises tool
        update support; on older hosts this method is a no-op.

        Args:
            content: Concise textual fallback for clients without structured
                update rendering.
            data: Optional JSON-serializable structured snapshot data.
        """

        if not self._tool_updates_enabled:
            return
        client = self._host_rpc_client
        if client is None:
            return
        payload: ToolUpdateRequest = {"content": content}
        if data is not None:
            payload["data"] = data
        await client.request("kodelet.tool.update", payload)


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


def _tool_updates_supported(init: Mapping[str, Any] | None) -> bool:
    if not init:
        return False
    capabilities = init.get("capabilities")
    if not isinstance(capabilities, Mapping):
        return False
    if capabilities.get("toolUpdates") is True:
        return True
    tools = capabilities.get("tools")
    return isinstance(tools, Mapping) and tools.get("updates") is True


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

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any, Literal, NotRequired, Protocol, TypedDict

from .agent import AgentStreamEvent

_UPDATE_INTERVAL_SECONDS = 0.2
_MAX_VISIBLE_RUNNING = 8
_MAX_VISIBLE_FAILED = 3
_MAX_VISIBLE_SUCCEEDED = 3
_MAX_ACTIVITY_ID_LENGTH = 256
_MAX_KIND_LENGTH = 64
_MAX_LABEL_LENGTH = 160
_MAX_PREVIEW_LENGTH = 180
_MAX_TASK_LENGTH = 1000
_MAX_CWD_LENGTH = 4096

TaskRunStatus = Literal["running", "completed", "failed"]
TaskRunPhase = Literal["starting", "working", "responding", "completed", "failed"]
TaskActivityStatus = Literal["running", "succeeded", "failed"]


class TaskRunCounts(TypedDict):
    """Observed task activity counts by state."""

    succeeded: int
    failed: int
    running: int


class TaskActivity(TypedDict):
    """One visible activity in an accumulated task-run snapshot."""

    id: str
    sequence: int
    kind: str
    label: str
    detail: str
    status: TaskActivityStatus
    preview: NotRequired[str]


class TaskRunSnapshot(TypedDict):
    """Bounded accumulated view of a long-running task."""

    version: Literal[1]
    revision: int
    kind: str
    status: TaskRunStatus
    phase: TaskRunPhase
    title: str
    detail: str
    task: str
    cwd: str
    elapsedMs: int
    counts: TaskRunCounts
    activities: list[TaskActivity]
    omittedSucceeded: int
    omittedFailed: NotRequired[int]
    omittedRunning: NotRequired[int]


class TaskProgressLogger(Protocol):
    """Logger surface used when a progress publication fails."""

    def warn(
        self,
        message: str,
        fields: Mapping[str, Any] | None = None,
    ) -> None: ...


class TaskProgressContext(Protocol):
    """Tool context surface required by :class:`TaskProgress`."""

    log: TaskProgressLogger

    def update(
        self,
        content: str,
        data: Mapping[str, Any] | None = None,
    ) -> Awaitable[None]: ...


class TaskProgressSession(Protocol):
    """Agent session event surface accepted by :meth:`TaskProgress.attach`."""

    def on(
        self,
        event_name: str,
        listener: Callable[[AgentStreamEvent], Any],
    ) -> Any: ...

    def off(
        self,
        event_name: str,
        listener: Callable[[AgentStreamEvent], Any],
    ) -> Any: ...


TaskActivityLabeler = Callable[
    [str, Mapping[str, Any], str],
    tuple[str, str],
]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _single_line(value: str, limit: int = _MAX_LABEL_LENGTH) -> str:
    collapsed = " ".join(value.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(1, limit - 1)].rstrip() + "…"


def _quoted(value: str) -> str:
    return json.dumps(_single_line(value, 80), ensure_ascii=False)


def _display_path(value: str, cwd: str) -> str:
    value = value.strip()
    if not value:
        return "."
    try:
        root = Path(cwd).expanduser().resolve(strict=False)
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve(strict=False)
        return str(path.relative_to(root)) or "."
    except (OSError, ValueError):
        return value


def _last_line(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    lines = [
        line.strip()
        for line in value.splitlines()
        if line.strip()
        and not line.strip().startswith("```")
        and not line.strip().startswith("~~~")
    ]
    if not lines:
        return None
    return _single_line(lines[-1], _MAX_PREVIEW_LENGTH)


def format_task_tool_activity(
    tool_name: str,
    tool_input: Mapping[str, Any],
    cwd: str,
) -> tuple[str, str]:
    """Return row and headline labels for a child tool call."""

    normalized = tool_name.strip().lower()
    path = _display_path(
        _text(tool_input.get("file_path")) or _text(tool_input.get("path")), cwd
    )

    if normalized in {"grep", "grep_tool"}:
        pattern = _text(tool_input.get("pattern"))
        label = f"Search {_quoted(pattern)} in {path}" if pattern else f"Search in {path}"
        detail = f"searching {path}"
    elif normalized in {"glob", "glob_tool"}:
        pattern = _text(tool_input.get("pattern"))
        label = f"Find files {_quoted(pattern)} in {path}" if pattern else f"Find files in {path}"
        detail = f"finding files in {path}"
    elif normalized == "file_read":
        label = f"Read {path}"
        detail = f"reading {path}"
    elif normalized == "file_write":
        label = f"Write {path}"
        detail = f"writing {path}"
    elif normalized == "file_edit":
        label = f"Edit {path}"
        detail = f"editing {path}"
    elif normalized == "apply_patch":
        label = "Apply patch"
        detail = "applying a patch"
    elif normalized == "bash":
        description = _text(tool_input.get("description"))
        command = _text(tool_input.get("command"))
        value = description or command or "command"
        label = f"Bash: {_single_line(value, 120)}"
        detail = _single_line(description or "running a command", 120).lower()
    elif normalized == "web_fetch":
        url = _text(tool_input.get("url"))
        label = f"Fetch {url}" if url else "Fetch web page"
        detail = f"fetching {url}" if url else "fetching a web page"
    elif normalized in {"web_search", "openai_web_search"}:
        query = _text(tool_input.get("query"))
        label = f"Search web for {_quoted(query)}" if query else "Search web"
        detail = "searching the web"
    elif normalized == "view_image":
        label = f"View image {path}"
        detail = f"viewing {path}"
    elif normalized == "skill":
        skill_name = _text(tool_input.get("skill_name")) or _text(
            tool_input.get("skillName")
        )
        label = f"Load skill {skill_name}" if skill_name else "Load skill"
        detail = f"loading {skill_name}" if skill_name else "loading a skill"
    elif normalized == "code_search":
        query = _text(tool_input.get("query"))
        label = f"Search code: {_single_line(query, 120)}" if query else "Search code"
        detail = "searching code"
    else:
        display_name = normalized.replace("_", " ").strip() or "activity"
        label = display_name[:1].upper() + display_name[1:]
        detail = f"running {display_name}"

    return _single_line(label), _single_line(detail)


class TaskProgress:
    """Publish bounded accumulated progress for a long-running task.

    Callers may report activities directly with :meth:`start_activity`,
    :meth:`update_activity`, and :meth:`finish_activity`. :meth:`attach` is a
    convenience adapter for child Kodelet sessions.
    """

    def __init__(
        self,
        ctx: TaskProgressContext,
        *,
        kind: str,
        task: str,
        cwd: str,
        running_title: str,
        completed_title: str,
        failed_title: str,
        responding_detail: str,
        labeler: TaskActivityLabeler = format_task_tool_activity,
    ) -> None:
        self._ctx = ctx
        self._kind = kind
        self._task = task
        self._cwd = cwd
        self._running_title = running_title
        self._completed_title = completed_title
        self._failed_title = failed_title
        self._responding_detail = responding_detail
        self._labeler = labeler
        self._started_at = time.monotonic()
        self._revision = 0
        self._sequence = 0
        self._status: TaskRunStatus = "running"
        self._phase: TaskRunPhase = "starting"
        self._running: dict[str, TaskActivity] = {}
        self._recent_succeeded: list[TaskActivity] = []
        self._recent_failed: list[TaskActivity] = []
        self._succeeded_count = 0
        self._failed_count = 0
        self._dirty = False
        self._publish_task: asyncio.Task[None] | None = None
        self._attached_session: TaskProgressSession | None = None

    async def start(self) -> None:
        """Publish the initial task snapshot."""

        self._changed(immediate=True)
        await self.flush()

    def attach(self, session: TaskProgressSession) -> None:
        """Track tool and response events from a child Kodelet session."""

        self._detach()
        self._attached_session = session
        session.on("tool.call", self._on_tool_call)
        session.on("tool.update", self._on_tool_update)
        session.on("tool.result", self._on_tool_result)
        session.on("assistant.message_delta", self._on_message_delta)

    def start_activity(
        self,
        activity_id: str,
        *,
        label: str,
        detail: str = "",
        kind: str = "",
    ) -> None:
        """Record a newly running task activity."""

        if not activity_id:
            return
        if existing := self._running.get(activity_id):
            existing["kind"] = kind
            existing["label"] = _single_line(label)
            existing["detail"] = _single_line(detail)
            self._phase = "working"
            self._changed(immediate=True)
            return
        self._sequence += 1
        activity: TaskActivity = {
            "id": activity_id,
            "sequence": self._sequence,
            "kind": kind,
            "label": _single_line(label),
            "detail": _single_line(detail),
            "status": "running",
        }
        self._running[activity_id] = activity
        self._phase = "working"
        self._changed(immediate=True)

    def update_activity(self, activity_id: str, result: str | None) -> None:
        """Update the bounded preview for a running activity."""

        activity = self._running.get(activity_id)
        if activity is None:
            return
        preview = _last_line(result)
        if preview and activity.get("preview") != preview:
            activity["preview"] = preview
            self._changed()

    def finish_activity(
        self,
        activity_id: str,
        *,
        success: bool,
        result: str | None = None,
    ) -> None:
        """Mark an observed activity as succeeded or failed."""

        activity = self._running.pop(activity_id, None)
        if activity is None:
            return
        activity["status"] = "succeeded" if success else "failed"
        preview = _last_line(result)
        if not success and preview:
            activity["preview"] = preview
        elif success:
            activity.pop("preview", None)
        if success:
            self._succeeded_count += 1
            self._recent_succeeded.append(activity)
            del self._recent_succeeded[:-_MAX_VISIBLE_SUCCEEDED]
        else:
            self._failed_count += 1
            self._recent_failed.append(activity)
            del self._recent_failed[:-_MAX_VISIBLE_FAILED]
        self._changed(immediate=True)

    def mark_responding(self) -> None:
        """Indicate that the task is producing its final response."""

        if self._phase == "responding":
            return
        self._phase = "responding"
        self._changed(immediate=True)

    async def finish(
        self,
        *,
        success: bool,
        error: str | None = None,
    ) -> TaskRunSnapshot:
        """Return the terminal snapshot to include in the tool result."""

        self._detach()
        await self.flush()
        self._status = "completed" if success else "failed"
        self._phase = "completed" if success else "failed"
        terminal_status: TaskActivityStatus = "succeeded" if success else "failed"
        terminal_activities = list(self._running.values())
        self._running.clear()
        for activity in terminal_activities:
            activity["status"] = terminal_status
            if error and not success:
                if preview := _last_line(error):
                    activity["preview"] = preview
        if success:
            self._succeeded_count += len(terminal_activities)
            self._recent_succeeded.extend(terminal_activities)
            del self._recent_succeeded[:-_MAX_VISIBLE_SUCCEEDED]
        else:
            self._failed_count += len(terminal_activities)
            self._recent_failed.extend(terminal_activities)
            del self._recent_failed[:-_MAX_VISIBLE_FAILED]
        self._revision += 1
        return self.snapshot()

    async def flush(self) -> None:
        """Wait until all pending progress publications have completed."""

        while True:
            task = self._publish_task
            if task is not None:
                await task
                continue
            if not self._dirty:
                return
            self._schedule_publish(immediate=True)

    def snapshot(self) -> TaskRunSnapshot:
        """Return the latest bounded accumulated task snapshot."""

        running = sorted(self._running.values(), key=lambda item: item["sequence"])
        selected = self._recent_succeeded + self._recent_failed + running[-_MAX_VISIBLE_RUNNING:]
        selected = sorted(selected, key=lambda item: item["sequence"])

        title = self._running_title
        if self._status == "completed":
            title = self._completed_title
        elif self._status == "failed":
            title = self._failed_title

        snapshot: TaskRunSnapshot = {
            "version": 1,
            "revision": self._revision,
            "kind": _single_line(self._kind, _MAX_KIND_LENGTH),
            "status": self._status,
            "phase": self._phase,
            "title": _single_line(title),
            "detail": _single_line(self._detail(running)),
            "task": _single_line(self._task, _MAX_TASK_LENGTH),
            "cwd": _single_line(self._cwd, _MAX_CWD_LENGTH),
            "elapsedMs": max(0, round((time.monotonic() - self._started_at) * 1000)),
            "counts": {
                "succeeded": self._succeeded_count,
                "failed": self._failed_count,
                "running": len(running),
            },
            "activities": [self._snapshot_activity(activity) for activity in selected],
            "omittedSucceeded": max(
                0,
                self._succeeded_count - len(self._recent_succeeded),
            ),
        }
        omitted_failed = max(0, self._failed_count - len(self._recent_failed))
        omitted_running = max(0, len(running) - _MAX_VISIBLE_RUNNING)
        if omitted_failed:
            snapshot["omittedFailed"] = omitted_failed
        if omitted_running:
            snapshot["omittedRunning"] = omitted_running
        return snapshot

    def _on_tool_call(self, event: AgentStreamEvent) -> None:
        data = _mapping(_mapping(event).get("data"))
        call_id = _text(data.get("toolCallId"))
        if not call_id:
            return
        tool_name = _text(data.get("toolName")) or "tool"
        tool_input = _mapping(data.get("input"))
        if not tool_input:
            raw_input = data.get("rawInput")
            if isinstance(raw_input, str):
                try:
                    tool_input = _mapping(json.loads(raw_input))
                except json.JSONDecodeError:
                    tool_input = {}
        label, detail = self._labeler(tool_name, tool_input, self._cwd)
        self.start_activity(
            call_id,
            label=label,
            detail=detail,
            kind=tool_name,
        )

    def _on_tool_update(self, event: AgentStreamEvent) -> None:
        data = _mapping(_mapping(event).get("data"))
        self.update_activity(_text(data.get("toolCallId")), _text(data.get("result")))

    def _detach(self) -> None:
        session = self._attached_session
        if session is None:
            return
        session.off("tool.call", self._on_tool_call)
        session.off("tool.update", self._on_tool_update)
        session.off("tool.result", self._on_tool_result)
        session.off("assistant.message_delta", self._on_message_delta)
        self._attached_session = None

    @staticmethod
    def _snapshot_activity(activity: TaskActivity) -> TaskActivity:
        snapshot = activity.copy()
        snapshot["id"] = _single_line(activity["id"], _MAX_ACTIVITY_ID_LENGTH)
        snapshot["kind"] = _single_line(activity["kind"], _MAX_KIND_LENGTH)
        snapshot["label"] = _single_line(activity["label"])
        snapshot["detail"] = _single_line(activity["detail"])
        if preview := activity.get("preview"):
            snapshot["preview"] = _single_line(preview, _MAX_PREVIEW_LENGTH)
        return snapshot

    def _on_tool_result(self, event: AgentStreamEvent) -> None:
        data = _mapping(_mapping(event).get("data"))
        self.finish_activity(
            _text(data.get("toolCallId")),
            success=_text(data.get("status")) != "failed",
            result=_text(data.get("result")),
        )

    def _on_message_delta(self, event: AgentStreamEvent) -> None:
        data = _mapping(_mapping(event).get("data"))
        if _text(data.get("deltaContent")):
            self.mark_responding()

    def _detail(self, running: list[TaskActivity]) -> str:
        if self._status == "failed":
            return "failed"
        if self._status == "completed":
            return ""
        if self._phase == "starting":
            return self._task or "starting task"
        if self._phase == "responding":
            return self._responding_detail
        if len(running) == 1:
            return _text(running[0].get("detail")) or "1 action running"
        if len(running) > 1:
            return f"{len(running)} actions running"
        return self._task or "planning next step"

    def _changed(self, *, immediate: bool = False) -> None:
        if self._status != "running":
            return
        self._revision += 1
        self._dirty = True
        self._schedule_publish(immediate=immediate)

    def _schedule_publish(self, *, immediate: bool) -> None:
        if self._publish_task is not None:
            return
        delay = 0 if immediate else _UPDATE_INTERVAL_SECONDS
        self._publish_task = asyncio.create_task(self._publish(delay))

    async def _publish(self, delay: float) -> None:
        try:
            if delay > 0:
                await asyncio.sleep(delay)
            while self._dirty:
                self._dirty = False
                snapshot = self.snapshot()
                content = snapshot["title"]
                if snapshot["detail"]:
                    content += f" - {snapshot['detail']}"
                try:
                    await self._ctx.update(content, {"taskRun": snapshot})
                except Exception as exc:
                    self._ctx.log.warn(
                        "failed to publish tool update",
                        {"error": str(exc)},
                    )
                if self._dirty:
                    await asyncio.sleep(_UPDATE_INTERVAL_SECONDS)
        finally:
            self._publish_task = None
            if self._dirty:
                self._schedule_publish(immediate=False)


__all__ = [
    "TaskActivity",
    "TaskActivityLabeler",
    "TaskActivityStatus",
    "TaskProgress",
    "TaskProgressContext",
    "TaskProgressLogger",
    "TaskProgressSession",
    "TaskRunCounts",
    "TaskRunPhase",
    "TaskRunSnapshot",
    "TaskRunStatus",
    "format_task_tool_activity",
]

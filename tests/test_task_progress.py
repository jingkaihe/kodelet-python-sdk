from __future__ import annotations

import unittest
from collections.abc import Callable, Mapping
from typing import Any

from kodelet_sdk import (
    AgentStreamEvent,
    TaskProgress,
    TaskProgressLogger,
    format_task_tool_activity,
)


class _Log:
    def warn(
        self,
        message: str,
        fields: Mapping[str, Any] | None = None,
    ) -> None:
        del message, fields


class _Context:
    def __init__(self) -> None:
        self.log: TaskProgressLogger = _Log()
        self.updates: list[tuple[str, dict[str, Any]]] = []

    async def update(
        self,
        content: str,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        self.updates.append((content, dict(data or {})))


class _Session:
    def __init__(self) -> None:
        self.listeners: dict[str, list[Callable[[AgentStreamEvent], Any]]] = {}

    def on(
        self,
        event_name: str,
        listener: Callable[[AgentStreamEvent], Any],
    ) -> _Session:
        self.listeners.setdefault(event_name, []).append(listener)
        return self

    def off(
        self,
        event_name: str,
        listener: Callable[[AgentStreamEvent], Any],
    ) -> _Session:
        listeners = self.listeners.get(event_name, [])
        if listener in listeners:
            listeners.remove(listener)
        return self

    def emit(self, event_name: str, data: Mapping[str, Any]) -> None:
        event = AgentStreamEvent({"type": event_name, "data": data})
        for listener in self.listeners.get(event_name, []):
            listener(event)


class TaskProgressTest(unittest.IsolatedAsyncioTestCase):
    async def test_tracks_bounded_child_session_activity(self) -> None:
        ctx = _Context()
        session = _Session()
        progress = TaskProgress(
            ctx,
            kind="code_search",
            task="Find the update path",
            cwd="/workspace",
            running_title="Searching code",
            completed_title="Searched code",
            failed_title="Code search failed",
            responding_detail="writing summary",
        )
        progress.attach(session)
        await progress.start()

        for index in range(5):
            session.emit(
                "tool.call",
                {
                    "toolCallId": f"call-{index}",
                    "toolName": "file_read",
                    "input": {"file_path": f"/workspace/pkg/file-{index}.go"},
                },
            )
            session.emit(
                "tool.result",
                {
                    "toolCallId": f"call-{index}",
                    "status": "completed",
                    "result": "done",
                },
            )

        session.emit(
            "tool.call",
            {
                "toolCallId": "running",
                "toolName": "grep_tool",
                "input": {"pattern": "HandleToolUpdate", "path": "/workspace/pkg"},
            },
        )
        await progress.flush()

        snapshot = progress.snapshot()
        self.assertEqual(snapshot["counts"], {"succeeded": 5, "failed": 0, "running": 1})
        self.assertEqual(snapshot["omittedSucceeded"], 2)
        self.assertEqual(snapshot["detail"], "searching pkg")
        self.assertEqual(len(snapshot["activities"]), 4)
        self.assertIn("Searching code", ctx.updates[-1][0])
        self.assertIn("taskRun", ctx.updates[-1][1])

        final = await progress.finish(success=True)
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["counts"]["running"], 0)
        self.assertTrue(all(not listeners for listeners in session.listeners.values()))

    async def test_supports_direct_non_agent_task_activity(self) -> None:
        ctx = _Context()
        progress = TaskProgress(
            ctx,
            kind="download",
            task="Fetch artifacts",
            cwd="/workspace",
            running_title="Downloading",
            completed_title="Downloaded",
            failed_title="Download failed",
            responding_detail="writing manifest",
        )
        await progress.start()
        progress.start_activity(
            "artifact-1",
            kind="download",
            label="Download artifact.tar.zst",
            detail="downloading artifact.tar.zst",
        )
        progress.finish_activity("artifact-1", success=True)
        await progress.flush()

        self.assertEqual(progress.snapshot()["counts"]["succeeded"], 1)

    def test_formats_tool_labels_relative_to_workspace(self) -> None:
        label, detail = format_task_tool_activity(
            "grep_tool",
            {"pattern": "HandleToolUpdate", "path": "/workspace/pkg"},
            "/workspace",
        )
        self.assertEqual(label, 'Search "HandleToolUpdate" in pkg')
        self.assertEqual(detail, "searching pkg")


if __name__ == "__main__":
    unittest.main()

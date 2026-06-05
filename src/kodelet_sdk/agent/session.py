from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable, Coroutine, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, Unpack, cast

from .._utils import AttrDict
from .rpc import ACPRPCClient
from .types import AgentResponse, AgentStreamEvent, RunOptions

if TYPE_CHECKING:
    from .bridge import InMemoryExtensionBridge, TempConfig
    from .client import Client


EventListener: TypeAlias = Callable[[AgentStreamEvent], Any]


class Session:
    """A single Kodelet ACP conversation session."""

    def __init__(
        self,
        client: Client,
        *,
        cwd: str,
        session_id: str,
        rpc: ACPRPCClient,
        max_turns: int | None = None,
        extension_bridge: InMemoryExtensionBridge | None = None,
        temp_config: TempConfig | None = None,
    ) -> None:
        self.cwd = cwd
        self._client = client
        self._rpc = rpc
        self._max_turns = max_turns
        self._conversation_id = session_id
        self._extension_bridge = extension_bridge
        self._temp_config = temp_config
        self._closed = False
        self._running = False
        self._listeners: dict[str, list[EventListener]] = {}
        self._listener_tasks: set[asyncio.Task[Any]] = set()

    @property
    def id(self) -> str:
        """Current Kodelet conversation/session id."""

        return self._conversation_id

    def on(self, event_name: str, listener: EventListener) -> Session:
        """Register a listener for SDK stream events."""

        self._listeners.setdefault(event_name, []).append(listener)
        return self

    def once(self, event_name: str, listener: EventListener) -> Session:
        """Register a listener that is removed after its first event."""

        def wrapper(event: AgentStreamEvent) -> Any:
            self.off(event_name, wrapper)
            return listener(event)

        return self.on(event_name, wrapper)

    def off(self, event_name: str, listener: EventListener) -> Session:
        """Remove a previously registered listener."""

        listeners = self._listeners.get(event_name)
        if listeners and listener in listeners:
            listeners.remove(listener)
        return self

    async def run_and_wait(
        self,
        options: RunOptions | str | None = None,
        **kwargs: Unpack[RunOptions],
    ) -> AgentResponse:
        """Run a prompt and wait for the final ACP response."""

        run_options = _normalize_run_options(options, kwargs)
        message = run_options["message"]
        max_turns = _option_int(run_options, "max_turns")
        if self._closed:
            raise RuntimeError("Cannot run a closed Kodelet session")
        if self._running:
            raise RuntimeError("Cannot run a Kodelet session while another run is in progress")
        if max_turns is not None and max_turns != self._max_turns:
            raise RuntimeError(
                "Per-run max_turns is not supported by the RPC transport; "
                "set max_turns in create_session instead"
            )

        self._running = True
        events: list[AgentStreamEvent] = []
        assistant_chunks: list[str] = []
        thinking_active = {"value": False}
        tool_names: dict[str, str] = {}

        def notification_handler(method: str, params: Any) -> None:
            if method == "session/update":
                self._handle_session_update(
                    params,
                    events,
                    assistant_chunks,
                    thinking_active,
                    tool_names,
                )

        unsubscribe = self._rpc.on_notification(notification_handler)
        self._emit_sdk_event("agent.start", {"message": message}, events)
        self._emit_sdk_event("user.message", {"content": message}, events)

        try:
            prompt_blocks = _build_prompt_blocks(run_options)
            result = await self._rpc.prompt(self._conversation_id, prompt_blocks)
            if thinking_active["value"]:
                thinking_active["value"] = False
                self._emit_sdk_event("assistant.thinking_end", {}, events)
            self._emit_sdk_event("assistant.content_end", {}, events)
            content = "".join(assistant_chunks)
            if content:
                self._emit_sdk_event("assistant.message", {"content": content}, events)
            response = AgentResponse(
                {
                    "content": content,
                    "conversationId": self._conversation_id,
                    "events": events,
                    "exitCode": 0,
                    "stopReason": result.get("stopReason"),
                }
            )
            self._emit_sdk_event("agent.end", {**response, "events": [*events]}, events)
            return response
        except asyncio.CancelledError:
            self._rpc.cancel_session(self._conversation_id)
            self._emit_sdk_event("agent.error", {"message": "cancelled"}, events)
            raise
        except Exception as exc:
            self._emit_sdk_event("agent.error", {"message": str(exc)}, events)
            raise
        finally:
            unsubscribe()
            self._running = False

    async def runAndWait(self, *args: Any, **kwargs: Any) -> AgentResponse:
        """CamelCase alias for :meth:`run_and_wait`."""

        return await self.run_and_wait(*args, **kwargs)

    async def close(self) -> None:
        """Close the underlying ACP process and temporary session resources."""

        if self._closed:
            return
        self._closed = True
        await self._rpc.close()
        if self._extension_bridge is not None:
            await self._extension_bridge.close()
        if self._temp_config is not None:
            await self._temp_config.close()
        for task in list(self._listener_tasks):
            task.cancel()
        self._client._delete_session(self)

    def _handle_session_update(
        self,
        params: Any,
        events: list[AgentStreamEvent],
        assistant_chunks: list[str],
        thinking_active: dict[str, bool],
        tool_names: dict[str, str],
    ) -> None:
        if not isinstance(params, Mapping):
            return
        if params.get("sessionId") != self._conversation_id:
            return
        update = params.get("update")
        if not isinstance(update, Mapping):
            return

        session_update = update.get("sessionUpdate")
        if session_update == "agent_message_chunk":
            content = _text_from_acp_content(update.get("content"))
            if content:
                assistant_chunks.append(content)
                self._emit_sdk_event(
                    "assistant.message_delta", {"deltaContent": content}, events, update
                )
            return
        if session_update == "agent_thought_chunk":
            if not thinking_active["value"]:
                thinking_active["value"] = True
                self._emit_sdk_event("assistant.thinking_start", {}, events, update)
            content = _text_from_acp_content(update.get("content"))
            if content:
                self._emit_sdk_event(
                    "assistant.thinking_delta", {"deltaContent": content}, events, update
                )
            return
        if session_update == "tool_call":
            tool_call_id = _string_field(update, "toolCallId")
            tool_name = _tool_name_from_update(update)
            if tool_call_id and tool_name:
                tool_names[tool_call_id] = tool_name
            raw_input = update.get("rawInput")
            raw_input_text = (
                raw_input
                if isinstance(raw_input, str)
                else json.dumps(raw_input if raw_input is not None else None, separators=(",", ":"))
            )
            self._emit_sdk_event(
                "tool.call",
                {
                    "toolName": tool_name,
                    "input": raw_input,
                    "rawInput": raw_input_text,
                    **({"toolCallId": tool_call_id} if tool_call_id else {}),
                },
                events,
                update,
            )
            return
        if session_update == "tool_call_update":
            status_value = _string_field(update, "status")
            if status_value not in {"completed", "failed"}:
                return
            tool_call_id = _string_field(update, "toolCallId")
            self._emit_sdk_event(
                "tool.result",
                {
                    "toolName": (tool_names.get(tool_call_id or "") if tool_call_id else None)
                    or _tool_name_from_update(update),
                    "result": _tool_content_to_text(update.get("content")),
                    **({"toolCallId": tool_call_id} if tool_call_id else {}),
                    "status": status_value,
                },
                events,
                update,
            )
            return

        self._emit_sdk_event("event", dict(update), events, update)

    def _emit_sdk_event(
        self,
        event_type: str,
        data: Any,
        events: list[AgentStreamEvent],
        raw: Any | None = None,
    ) -> AgentStreamEvent:
        event = AgentStreamEvent(
            {
                "type": event_type,
                "data": _to_attr_dict(data),
                "conversationId": self._conversation_id,
                **({"raw": raw} if raw is not None else {}),
            }
        )
        events.append(event)
        self._emit(event_type, event)
        if event_type != "event":
            self._emit("event", event)
        return event

    def _emit(self, event_name: str, event: AgentStreamEvent) -> None:
        for listener in list(self._listeners.get(event_name, [])):
            result = listener(event)
            if inspect.isawaitable(result):
                task = asyncio.create_task(cast(Coroutine[Any, Any, Any], result))
                self._listener_tasks.add(task)
                task.add_done_callback(self._listener_tasks.discard)


def _normalize_run_options(
    options: RunOptions | str | None,
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(options, str):
        run_options = {"message": options}
    else:
        run_options = dict(options or {})
    run_options.update(kwargs)
    message = run_options.get("message")
    if not isinstance(message, str) or not message:
        raise ValueError("run_and_wait requires a non-empty message")
    return run_options


def _option_int(options: Mapping[str, Any], key: str) -> int | None:
    value = options.get(key)
    return value if isinstance(value, int) else None


def _build_prompt_blocks(options: Mapping[str, Any]) -> list[dict[str, Any]]:
    prompt = [{"type": "text", "text": str(options["message"])}]
    images = options.get("images")
    if isinstance(images, Sequence) and not isinstance(images, str | bytes):
        for image in images:
            if isinstance(image, str):
                prompt.append(_image_to_content_block(image))
    return prompt


def _image_to_content_block(image: str) -> dict[str, Any]:
    prefix = "data:"
    marker = ";base64,"
    if image.startswith(prefix) and marker in image:
        mime_type, data = image[len(prefix) :].split(marker, 1)
        return {"type": "image", "mimeType": mime_type, "data": data}
    return {"type": "image", "uri": image}


def _text_from_acp_content(content: Any) -> str:
    if not isinstance(content, Mapping):
        return ""
    text = content.get("text")
    if isinstance(text, str):
        return text
    resource = content.get("resource")
    if isinstance(resource, Mapping) and isinstance(resource.get("text"), str):
        return str(resource["text"])
    return ""


def _tool_name_from_update(update: Mapping[str, Any]) -> str:
    tool_name = update.get("toolName")
    return tool_name.strip() if isinstance(tool_name, str) and tool_name.strip() else ""


def _tool_content_to_text(content: Any) -> str:
    if not isinstance(content, Sequence) or isinstance(content, str | bytes):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, Mapping):
            continue
        if item.get("type") == "content" and isinstance(item.get("content"), Mapping):
            text = _text_from_acp_content(item.get("content"))
            if text:
                parts.append(text)
            continue
        path = item.get("path")
        if isinstance(path, str):
            parts.append(path)
        new_text = item.get("newText")
        if isinstance(new_text, str):
            parts.append(new_text)
    return "\n".join(parts)


def _string_field(record: Mapping[str, Any], key: str) -> str | None:
    value = record.get(key)
    return value if isinstance(value, str) else None


def _to_attr_dict(value: Any) -> Any:
    if isinstance(value, Mapping):
        return AttrDict({str(key): _to_attr_dict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_attr_dict(item) for item in value]
    return value


__all__ = ["EventListener", "Session"]

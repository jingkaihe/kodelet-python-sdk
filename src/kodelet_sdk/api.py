from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, overload

from pydantic import ValidationError

from ._utils import (
    AttrDict,
    json_clone,
    maybe_await,
    merge_timeout_in_sec,
    normalize_command_name,
    optional_timeout,
    to_plain,
)
from .context import create_command_context, create_event_context, create_tool_context
from .schemas import SchemaAdapter, SchemaLike, infer_schema_from_callable

CommandAction = Literal["pass", "respond", "runAgent"]
CommandKind = Literal["command", "recipe"]
ToolHandler = Callable[[Any, Any], Awaitable[Any] | Any]
CommandHandler = Callable[[Any, Any], Awaitable[Any] | Any]
EventHandler = Callable[[AttrDict, Any], Awaitable[Any] | Any]
Entrypoint = Callable[["Extension"], Awaitable[None] | None]


@dataclass(frozen=True)
class ToolRegistration:
    name: str
    description: str
    input_schema: SchemaAdapter
    timeout_in_sec: float | None
    handler: ToolHandler


@dataclass(frozen=True)
class CommandRegistration:
    name: str
    aliases: tuple[str, ...]
    description: str
    input_schema: SchemaAdapter | None
    kind: CommandKind | None
    timeout_in_sec: float | None
    handler: CommandHandler


@dataclass(frozen=True)
class EventHandlerRegistration:
    event: str
    priority: int
    timeout_in_sec: float | None
    order: int
    handler: EventHandler


class Extension:
    """Register and run a Kodelet extension.

    An ``Extension`` owns tool, command, and event-handler registrations and
    exposes the JSON-RPC methods that the Kodelet host calls over stdio. Most
    extensions create one instance, decorate async functions with :meth:`tool`,
    :meth:`command`, and :meth:`on`, then call :meth:`run` or :meth:`run_sync`.

    Args:
        name: Optional display name returned during ``extension.initialize``.
            If omitted, Kodelet's configured extension id is used.
        version: Optional extension version returned during initialization.
    """

    def __init__(self, *, name: str | None = None, version: str | None = None) -> None:
        self._metadata: dict[str, str] = {}
        if name is not None:
            self._metadata["name"] = name
        if version is not None:
            self._metadata["version"] = version
        self._tools: dict[str, ToolRegistration] = {}
        self._commands_by_name: dict[str, CommandRegistration] = {}
        self._command_registrations: list[CommandRegistration] = []
        self._handlers: list[EventHandlerRegistration] = []
        self._order = 0
        self._init_params: Mapping[str, Any] | None = None

    def set_metadata(
        self,
        metadata: Mapping[str, str | None] | None = None,
        **kwargs: str | None,
    ) -> None:
        """Set metadata advertised to Kodelet during initialization.

        Args:
            metadata: Mapping of metadata keys such as ``name`` and ``version``.
            **kwargs: Additional metadata values. Keyword arguments override
                values from ``metadata``. ``None`` values are ignored.
        """

        for key, value in {**dict(metadata or {}), **kwargs}.items():
            if value is not None:
                self._metadata[key] = value

    def register_tool(
        self,
        *,
        name: str,
        description: str,
        input_schema: SchemaLike = None,
        execute: ToolHandler,
        timeout_in_sec: float | None = None,
    ) -> None:
        """Register a tool callable explicitly.

        Args:
            name: Tool name exposed to the LLM and Kodelet host.
            description: Human-readable tool description.
            input_schema: Pydantic model/type adapter or JSON Schema mapping
                used for initialization metadata. Pydantic schemas are also
                used to validate inputs before ``execute`` is called.
            execute: Callable invoked as ``execute(input, ctx)``. It may be
                sync or async and may return a string or a tool-result mapping.
            timeout_in_sec: Optional per-tool timeout hint. ``0`` asks the host
                to run without a timeout.

        Raises:
            ValueError: If another tool with the same name is already
                registered.
        """

        if name in self._tools:
            raise ValueError(f"Duplicate extension tool registration: {name}")
        self._tools[name] = ToolRegistration(
            name=name,
            description=description,
            input_schema=SchemaAdapter(input_schema),
            timeout_in_sec=timeout_in_sec,
            handler=execute,
        )

    def tool(
        self,
        name: str | None = None,
        *,
        description: str | None = None,
        input_schema: SchemaLike = None,
        timeout_in_sec: float | None = None,
    ) -> Callable[[ToolHandler], ToolHandler]:
        """Decorate a function as a Kodelet tool.

        Args:
            name: Optional public tool name. Defaults to the function name.
            description: Optional description. Defaults to the function
                docstring, then the tool name.
            input_schema: Pydantic model/type adapter or JSON Schema mapping.
                If omitted and the first parameter is annotated with a Pydantic
                ``BaseModel`` subclass, that annotation is used automatically.
            timeout_in_sec: Optional per-tool timeout hint. Use ``0`` for no
                timeout.

        Returns:
            A decorator that returns the original function unchanged.
        """

        def decorator(func: ToolHandler) -> ToolHandler:
            tool_name = name or _callable_name(func)
            tool_description = description or inspect.getdoc(func) or tool_name
            schema = input_schema if input_schema is not None else infer_schema_from_callable(func)
            self.register_tool(
                name=tool_name,
                description=tool_description,
                input_schema=schema,
                execute=func,
                timeout_in_sec=timeout_in_sec,
            )
            return func

        return decorator

    def register_command(
        self,
        *,
        name: str,
        description: str,
        execute: CommandHandler,
        input_schema: SchemaLike = None,
        aliases: Sequence[str] | None = None,
        kind: CommandKind | None = None,
        timeout_in_sec: float | None = None,
    ) -> None:
        """Register a slash command or recipe command explicitly.

        Args:
            name: Primary command name. Leading slashes are accepted and
                normalized away for dispatch.
            description: Human-readable command description.
            execute: Callable invoked as ``execute(input, ctx)``. It may be
                sync or async and should return a command-result mapping, or
                ``None`` to pass to the next command route.
            input_schema: Optional Pydantic model/type adapter or JSON Schema
                mapping. Pydantic validation failures return ``{"action":
                "pass"}`` so other command routes can match.
            aliases: Optional alternate command names. Aliases may include a
                leading slash.
            kind: Optional Kodelet command kind, ``"command"`` or
                ``"recipe"``.
            timeout_in_sec: Optional per-command timeout hint. Use ``0`` for no
                timeout.

        Raises:
            ValueError: If the primary name or any alias conflicts with an
                existing registration.
        """

        primary_name = normalize_command_name(name)
        alias_names = tuple(
            candidate
            for alias in aliases or ()
            if (candidate := normalize_command_name(alias)) and candidate != primary_name
        )
        names = (primary_name, *alias_names)
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate extension command registration: {name}")
        for command_name in names:
            if command_name in self._commands_by_name:
                raise ValueError(f"Duplicate extension command registration: {command_name}")
        registration = CommandRegistration(
            name=primary_name,
            aliases=tuple(aliases or ()),
            description=description,
            input_schema=SchemaAdapter(input_schema) if input_schema is not None else None,
            kind=kind,
            timeout_in_sec=timeout_in_sec,
            handler=execute,
        )
        self._command_registrations.append(registration)
        for command_name in names:
            self._commands_by_name[command_name] = registration

    def command(
        self,
        name: str | None = None,
        *,
        description: str | None = None,
        input_schema: SchemaLike = None,
        aliases: Sequence[str] | None = None,
        kind: CommandKind | None = None,
        timeout_in_sec: float | None = None,
    ) -> Callable[[CommandHandler], CommandHandler]:
        """Decorate a function as a Kodelet command.

        Args:
            name: Optional public command name. Defaults to the function name.
            description: Optional description. Defaults to the function
                docstring, then the command name.
            input_schema: Optional Pydantic model/type adapter or JSON Schema
                mapping. If omitted and the first parameter is a Pydantic model
                annotation, that model is used automatically.
            aliases: Optional alternate command names.
            kind: Optional command kind, ``"command"`` or ``"recipe"``.
            timeout_in_sec: Optional per-command timeout hint. Use ``0`` for no
                timeout.

        Returns:
            A decorator that returns the original function unchanged.
        """

        def decorator(func: CommandHandler) -> CommandHandler:
            command_name = name or _callable_name(func)
            command_description = description or inspect.getdoc(func) or command_name
            schema = input_schema if input_schema is not None else infer_schema_from_callable(func)
            self.register_command(
                name=command_name,
                description=command_description,
                input_schema=schema,
                aliases=aliases,
                kind=kind,
                timeout_in_sec=timeout_in_sec,
                execute=func,
            )
            return func

        return decorator

    @overload
    def on(self, event: str, handler: EventHandler, /) -> EventHandler: ...

    @overload
    def on(
        self,
        event: str,
        /,
        *,
        priority: int = 0,
        timeout_in_sec: float | None = None,
    ) -> Callable[[EventHandler], EventHandler]: ...

    def on(
        self,
        event: str,
        handler: EventHandler | None = None,
        /,
        *,
        priority: int = 0,
        timeout_in_sec: float | None = None,
    ) -> EventHandler | Callable[[EventHandler], EventHandler]:
        """Register an event handler.

        Can be used either as ``@ext.on("session.start")`` or as
        ``ext.on("session.start", handler)``. Handlers receive ``(event, ctx)``;
        ``event`` is a dict-like object that also supports attribute access.

        Args:
            event: Dot-separated Kodelet event name, for example
                ``"session.start"`` or ``"tool.call"``.
            handler: Optional handler callable. If omitted, this method returns
                a decorator.
            priority: Higher-priority handlers run before lower-priority
                handlers for the same event. Ties preserve registration order.
            timeout_in_sec: Optional per-event timeout hint. Use ``0`` for no
                timeout.

        Returns:
            The registered handler when ``handler`` is supplied, otherwise a
            decorator.
        """

        def decorator(func: EventHandler) -> EventHandler:
            self._handlers.append(
                EventHandlerRegistration(
                    event=event,
                    priority=priority,
                    timeout_in_sec=timeout_in_sec,
                    order=self._order,
                    handler=func,
                )
            )
            self._order += 1
            return func

        if handler is not None:
            return decorator(handler)
        return decorator

    def initialize(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Handle Kodelet's ``extension.initialize`` JSON-RPC request.

        Args:
            params: Raw initialization parameters from the Kodelet host.

        Returns:
            JSON-serializable extension metadata, tool registrations, command
            registrations, and event subscriptions.
        """

        self._init_params = params
        extension = params.get("extension")
        extension_id = extension.get("id") if isinstance(extension, Mapping) else None
        result: dict[str, Any] = {
            "name": self._metadata.get("name") or extension_id or "extension",
            "tools": [self._tool_to_json(registration) for registration in self._tools.values()],
            "commands": [
                self._command_to_json(registration)
                for registration in self._command_registrations
            ],
            "subscriptions": self._subscriptions(),
        }
        if version := self._metadata.get("version"):
            result["version"] = version
        return result

    async def execute_tool(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Handle Kodelet's ``extension.tool.execute`` JSON-RPC request.

        Args:
            params: Raw tool execution parameters containing ``name``,
                ``input``, and optional call ``context``.

        Returns:
            A JSON-serializable tool result. String handler results are wrapped
            as ``{"content": result}``.
        """

        name = str(params.get("name", ""))
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown extension tool: {name}")
        input_value = tool.input_schema.validate(params.get("input"))
        context = _mapping_or_empty(params.get("context"))
        result = await _call_with_input_and_context(
            tool.handler,
            input_value,
            create_tool_context(self._init_params, context),
        )
        if isinstance(result, str):
            return {"content": result}
        return to_plain(result)

    async def execute_command(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Handle Kodelet's ``extension.command.execute`` JSON-RPC request.

        Args:
            params: Raw command parameters containing ``name``, optional
                ``input``, optional call ``context``, and ``invocation`` data.

        Returns:
            A command-result mapping. Missing handler results or validation
            failures return ``{"action": "pass"}``.
        """

        name = normalize_command_name(str(params.get("name", "")))
        command = self._commands_by_name.get(name)
        if command is None:
            raise ValueError(f"Unknown extension command: {name}")
        input_value: Any = params.get("input") or {}
        if command.input_schema is not None:
            try:
                input_value = command.input_schema.validate(input_value)
            except ValidationError:
                return {"action": "pass"}
        context = _mapping_or_empty(params.get("context"))
        invocation = _mapping_or_empty(params.get("invocation"))
        result = await _call_with_input_and_context(
            command.handler,
            input_value,
            create_command_context(self._init_params, context, invocation),
        )
        return {"action": "pass"} if result is None else to_plain(result)

    async def handle_event(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Handle Kodelet's ``extension.event.handle`` JSON-RPC request.

        Args:
            params: Raw event parameters containing ``id``, ``event``, optional
                ``payload``, and optional call ``context``.

        Returns:
            Aggregated event mutations from all matching handlers. Input/output
            mutations are visible to later handlers for the same event; a
            ``block`` result stops further handler execution.
        """

        event_name = str(params.get("event", ""))
        handlers = sorted(
            (handler for handler in self._handlers if handler.event == event_name),
            key=lambda handler: (-handler.priority, handler.order),
        )
        payload = json_clone(params.get("payload") or {})
        if not isinstance(payload, dict):
            payload = {}
        payload["id"] = params.get("id")
        payload["event"] = event_name
        event = _to_attr_dict(payload)
        ctx = create_event_context(self._init_params, _mapping_or_empty(params.get("context")))
        aggregate: dict[str, Any] = {}

        for entry in handlers:
            raw_result = await maybe_await(entry.handler(event, ctx))
            if raw_result is None:
                continue
            result = to_plain(raw_result)
            if not isinstance(result, Mapping):
                continue
            if "input" in result and result["input"] is not None:
                aggregate["input"] = result["input"]
                _set_nested_tool_field(event, "input", result["input"])
            if "output" in result and result["output"] is not None:
                aggregate["output"] = result["output"]
                _set_nested_tool_field(event, "output", result["output"])
            if "message" in result and result["message"] is not None:
                aggregate["message"] = result["message"]
            if "systemPrompt" in result and result["systemPrompt"] is not None:
                aggregate["systemPrompt"] = result["systemPrompt"]
            if "tools" in result and result["tools"] is not None:
                aggregate["tools"] = _merge_tool_patch(aggregate.get("tools"), result["tools"])
            if "followUpMessages" in result and result["followUpMessages"] is not None:
                aggregate["followUpMessages"] = [
                    *aggregate.get("followUpMessages", []),
                    *list(result["followUpMessages"]),
                ]
            if "resources" in result and result["resources"] is not None:
                aggregate["resources"] = result["resources"]
            if result.get("block"):
                aggregate["block"] = result["block"]
                return aggregate

        return aggregate

    async def run(self) -> None:
        """Run this extension as an asyncio JSON-RPC stdio server."""

        from .runtime import run_extension

        await run_extension(self)

    def run_sync(self) -> None:
        """Run this extension from synchronous ``if __name__ == '__main__'`` code."""

        import asyncio

        asyncio.run(self.run())

    def _tool_to_json(self, registration: ToolRegistration) -> dict[str, Any]:
        return {
            "name": registration.name,
            "description": registration.description,
            "inputSchema": registration.input_schema.json_schema(),
            **optional_timeout(registration.timeout_in_sec),
        }

    def _command_to_json(self, registration: CommandRegistration) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": registration.name,
            "description": registration.description,
            **optional_timeout(registration.timeout_in_sec),
        }
        if registration.aliases:
            result["aliases"] = list(registration.aliases)
        if registration.input_schema is not None:
            result["inputSchema"] = registration.input_schema.json_schema()
        if registration.kind is not None:
            result["kind"] = registration.kind
        return result

    def _subscriptions(self) -> list[dict[str, Any]]:
        by_event: dict[str, dict[str, float | int | None]] = {}
        for handler in self._handlers:
            previous = by_event.get(handler.event)
            if previous is None:
                by_event[handler.event] = {
                    "priority": handler.priority,
                    "timeoutInSec": handler.timeout_in_sec,
                }
                continue
            previous_priority = int(previous.get("priority") or 0)
            previous_timeout = previous.get("timeoutInSec")
            by_event[handler.event] = {
                "priority": max(previous_priority, handler.priority),
                "timeoutInSec": merge_timeout_in_sec(
                    previous_timeout if isinstance(previous_timeout, int | float) else None,
                    handler.timeout_in_sec,
                ),
            }
        return [
            {
                "event": event,
                "priority": int(options.get("priority") or 0),
                **optional_timeout(
                    value
                    if isinstance((value := options.get("timeoutInSec")), int | float)
                    else None
                ),
            }
            for event, options in by_event.items()
        ]


def define_extension(entrypoint: Entrypoint) -> Entrypoint:
    """Return an extension entrypoint unchanged.

    This mirrors the TypeScript SDK helper and is useful for readability in
    tests and module-level extension definitions.

    Args:
        entrypoint: Callable that receives an :class:`Extension` and registers
            tools, commands, and event handlers.

    Returns:
        The same callable.
    """

    return entrypoint


async def create_extension_host(entrypoint: Extension | Entrypoint) -> Extension:
    """Create an initialized in-process extension host.

    Args:
        entrypoint: Either an existing :class:`Extension` or a callable that
            registers behavior on a newly created instance.

    Returns:
        The extension host ready to receive initialize/execute/event calls.
    """

    if isinstance(entrypoint, Extension):
        return entrypoint
    extension = Extension()
    await maybe_await(entrypoint(extension))
    return extension


def on(
    extension: Extension,
    event: str,
    handler: EventHandler | None = None,
    /,
    *,
    priority: int = 0,
    timeout_in_sec: float | None = None,
) -> EventHandler | Callable[[EventHandler], EventHandler]:
    """Functional wrapper around :meth:`Extension.on`.

    Args:
        extension: Extension to register the handler on.
        event: Dot-separated event name.
        handler: Optional event handler callable. If omitted, returns a
            decorator.
        priority: Handler priority for ordering.
        timeout_in_sec: Optional timeout hint for the subscription.

    Returns:
        The registered handler or a decorator.
    """

    if handler is not None:
        return extension.on(event, handler)
    return extension.on(event, priority=priority, timeout_in_sec=timeout_in_sec)


async def _call_with_input_and_context(
    handler: Callable[..., Any],
    input_value: Any,
    ctx: Any,
) -> Any:
    try:
        return await maybe_await(handler(input_value, ctx))
    except TypeError as exc:
        # Allow zero/one-argument handlers for small scripts, but avoid hiding TypeErrors
        # raised from inside the handler body when the signature actually accepts two args.
        signature = inspect.signature(handler)
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and parameter.default is inspect.Parameter.empty
        ]
        if len(positional) >= 2:
            raise
        if len(positional) == 1:
            return await maybe_await(handler(input_value))
        if len(positional) == 0:
            return await maybe_await(handler())
        raise exc


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _callable_name(func: Callable[..., Any]) -> str:
    name = getattr(func, "__name__", None)
    if isinstance(name, str):
        return name
    return func.__class__.__name__


def _to_attr_dict(value: Any) -> Any:
    if isinstance(value, Mapping):
        return AttrDict({str(key): _to_attr_dict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_attr_dict(item) for item in value]
    return value


def _set_nested_tool_field(event: AttrDict, field: str, value: Any) -> None:
    tool = event.get("tool")
    if isinstance(tool, Mapping):
        tool[field] = value


def _merge_tool_patch(current: Any, next_patch: Any) -> dict[str, list[Any]]:
    current_mapping = current if isinstance(current, Mapping) else {}
    next_mapping = next_patch if isinstance(next_patch, Mapping) else {}
    return {
        "disable": [
            *list(current_mapping.get("disable") or []),
            *list(next_mapping.get("disable") or []),
        ],
        "enable": [
            *list(current_mapping.get("enable") or []),
            *list(next_mapping.get("enable") or []),
        ],
    }

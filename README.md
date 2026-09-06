# kodelet-sdk

Python SDK for authoring [Kodelet](https://github.com/jingkaihe/kodelet) extensions.

The SDK speaks Kodelet's JSON-RPC extension protocol over stdio and provides an asyncio-first API for registering tools, commands, native TUI shortcuts, and event handlers.

## Quick start

```python
from kodelet_sdk import BaseModel, Extension, ToolContext, ToolExecutionResult

ext = Extension(name="weather", version="0.1.0")


class WeatherInput(BaseModel):
    location: str


@ext.tool("get_weather", description="Get weather", input_schema=WeatherInput)
async def get_weather(input: WeatherInput, ctx: ToolContext) -> ToolExecutionResult:
    return {"content": f"Weather for {input.location}"}


@ext.on("session.start")
async def session_start(event, ctx):
    ctx.log.info("extension started")


if __name__ == "__main__":
    ext.run_sync()
```

## Public API

### Agent sessions

Use `Client` to launch the thin `kodelet acp` daemon client over stdio JSON-RPC. Set `server` and `runner` on `Client`, or use normal daemon selection. Standalone clients authenticate with client credentials such as `KODELET_AUTH_TOKEN`; provider credentials stay on the daemon. Session `cwd` is interpreted on the runner, not used as the local subprocess directory.

```python
from kodelet_sdk import Client

client = Client()
session = await client.create_session()
response = await session.run_and_wait(message="what is the meaning of life?")

print(response.content)
await client.close()
```

Pass a named or inline `Profile` when creating a session, and listen for typed stream events while a run is active:

```python
from kodelet_sdk import Client, Profile

client = Client(command="kodelet")
session = await client.create_session(
    profile=Profile(
        {
            "provider": "openai",
            "model": "gpt-5.5",
        }
    ),
    max_turns=4,
    streaming=True,
)

session.on(
    "assistant.message_delta",
    lambda event: print(event.data.deltaContent, end="", flush=True),
)
session.on(
    "tool.update",
    lambda event: print(f"partial {event.data.toolCallId}: {event.data.result}"),
)
session.on(
    "tool.result",
    lambda event: print(f"final {event.data.toolCallId}: {event.data.result}"),
)

response = await session.run_and_wait(message="help me choose an approach")
print("\nfinal:", response.content)
await client.close()
```

Each `tool.update` contains the latest accumulated output snapshot, not a new delta. Listeners receive every snapshot. To keep completed responses bounded, `response.events` retains only the latest `tool.update` for each `toolCallId`, followed by the authoritative `tool.result`.

`create_session(options=ExecutionOptions(...), environment_profile="workspace")` accepts typed per-session execution settings and a runner-owned environment profile. Named `profile` values select daemon model profiles; inline profiles accept only typed model/resource/restriction options, not arbitrary provider configuration, credentials, endpoints, or local prompt paths. Explicit false, permitted zero, and empty lists are preserved. Temporary config files and inline executable extensions no longer configure remote execution. `session.close()` detaches; explicit `session.cancel()` targets the active turn.

### Extension-owned presets and delegated children

Use `ctx.children`, not a nested `Client` or `inherit_context`, for model work from an extension tool. Register a preset on the parent extension; its name is scoped to that extension/environment and need not exist in daemon YAML. System-prompt paths are resolved relative to the extension directory on the runner and frozen for the child, including later conversation resume.

```python
ext.register_profile({
    "name": "code_search",
    "systemPromptPath": "search-prompt.md",
    "options": {
        "model": "gpt-4o-mini",
        "allowedTools": ["file_read", "grep_tool", "glob_tool"],
        "noExtensions": True, "noSkills": True,
        "enableFSSearchTools": True, "maxTurns": 3,
    },
})

@ext.tool("code_search", description="Search the repository", input_schema=TaskInput)
async def search(input: TaskInput, ctx: ToolContext) -> str:
    child = await ctx.children.start(
        profile="code_search", message=input.task, request_id="this-tool-call-search-1",
    )
    result = await child.wait(on_event=lambda event: ctx.update(event.get("text") or event["kind"]))
    return result["output"]
```

`ExecutionProfile` and `ExecutionOptions` accept Python field names as well as camelCase wire names; registration snapshots them. `child.conversation_id` and `child.run_id` are separate durable identities with parent metadata. `read()` returns status/progress; `cancel()` targets only that child. Cancelling `wait()` cancels that child. Child options cannot widen the parent's effective permissions or resource limits, and model selection is validated centrally. `system_prompt` can supply per-invocation content and `cwd` can select a descendant workspace directory. No administrative token, subprocess model loop, or local-provider fallback is used.

Foreground children end with the owning tool handler. For work that outlives it, acquire an activated runner lease with `await ctx.acquire_background_task(...)`, pass `lease=lease` on the first child submission inside the tool, and release it only after all child work finishes. Retained authority has a non-renewing one-hour maximum lifetime and is revoked on release, cancellation, runner/extension loss, or shutdown. Provisional initialization leases are not authority. Stable `request_id` values reconcile repeated submissions within the same live capability; changed input is rejected. Capabilities and bounded progress caches are not restart-replay credentials. Foreground usage aggregates into the parent; retained child usage remains in its own conversation.

`await ctx.fork_conversation(name="Snapshot")` remains available for taking a history snapshot, but does not authorize execution. `inherit_context` is rejected before spawning; migrate execution to registered presets and `ctx.children`.

Live forks require a persistent in-memory conversation. `fork_conversation()` raises `ConversationForkUnavailableError` when unavailable; other host RPC errors should be surfaced.

An active session can receive additional guidance without starting another run. Call `steer()` only after a streaming event confirms that the run is active:

```python
import asyncio

run_active = asyncio.Event()
session.once("assistant.thinking_start", lambda _event: run_active.set())
run_task = asyncio.create_task(
    session.run_and_wait(message="Review the persistence implementation")
)

await run_active.wait()
steered = await session.steer("Also check transaction boundaries")
response = await run_task
```

`steer()` uses the ACP `_session/steering` extension and returns an outcome such as `{"outcome": "injected"}`. It rejects calls when no run is active or the ACP server does not advertise `_meta.steering.supported`. The SDK requests `idleBehavior: "promptRequired"`, so an end-of-turn race returns `{"outcome": "promptRequired", "reason": "noRunningTurn"}` rather than silently starting another turn. `injected` means Kodelet queued the message, not that the model consumed it before the prompt ended; guidance left unconsumed remains on the conversation for a later run. Blank steering messages are rejected locally.

Install executable extensions on the selected runner, where tools, skills, and lifecycle handlers execute. Inline `create_session(extensions=..., extension_transport=..., ui=...)` callbacks are explicitly rejected before spawning rather than silently ignored.

```python
from kodelet_sdk import BaseModel, Extension


ext = Extension(name="workspace", version="0.1.0")


class AskInput(BaseModel):
    question: str
    options: list[str]


@ext.tool("ask_user_question", description="Ask the user", input_schema=AskInput)
async def ask_user_question(input: AskInput, ctx):
    choice = await ctx.ui.select({"title": input.question, "options": input.options})
    return choice or "dismissed"


ext.run_sync()  # Invoke from a runner-installed kodelet-extension-* executable.
```

### Extension registration

- `Extension(name=None, version=None)` creates an extension host.
- `ext.register_profile(ExecutionProfile(...))` or `ext.register_profile({...})` registers an extension-owned child execution preset.
- `@ext.tool(name=None, description=None, input_schema=None, timeout_in_sec=None)` registers a tool.
- `@ext.command(name=None, description=None, input_schema=None, aliases=None, kind=None, timeout_in_sec=None)` registers a command.
- `@ext.shortcut(shortcut, description=None)` registers a native TUI keyboard shortcut handler; `ext.register_shortcut(shortcut, handler=..., description=None)` is the explicit form.
- `@ext.on(event, priority=0, timeout_in_sec=None)` registers an event handler such as `session.start`, `tool.call`, `tool.update`, or `agent.end`.
- `await ext.run()` starts the async stdio runtime; `ext.run_sync()` is a synchronous entrypoint convenience.

Handlers may be synchronous or asynchronous. Tool handlers may return a string, which is converted to `{ "content": ... }`, or a protocol-shaped mapping. Command handlers return `{ "action": "pass" }`, `{ "action": "respond", "response": ... }`, or `{ "action": "runAgent", "prompt": ... }`. A `runAgent` result may include optional `display` text to replace the slash command in the visible and persisted user message while keeping `prompt` as the LLM input.

Tool results may include host-facing presentation metadata under `data["presentation"]`:

```python
from kodelet_sdk import ToolExecutionResult, ToolPresentation

presentation: ToolPresentation = {
    "summary": "Found 2 matches",
    "body": "- `src/api.py`\n- `tests/test_api.py`",
    "format": "markdown",
}
result: ToolExecutionResult = {
    "content": "Found 2 matches.",
    "data": {"presentation": presentation},
}
```

`summary` is required and supplies the complete compact label. `body` optionally provides expanded details; only an omitted body falls back to the ordinary tool content. `format`, when present, declares the body as plain `text` or `markdown`. The SDK forwards this advisory object unchanged inside the generic `data` mapping, and it does not replace model-facing `content`, status, errors, or provenance. Hosts validate presentation metadata as untrusted input, sanitize Markdown, ignore malformed values, and may truncate bodies to their configured extension output limit.

Shortcut handlers receive a `ShortcutContext` and may return `{"action": "submit", "message": "/dictate"}` when the host advertises `capabilities.shortcuts.submit`. Validated shortcuts appear in the native TUI's shortcut help.

Supported shortcut identifiers are case-insensitive ASCII single chords: `ctrl+<ASCII letter>`, `alt+<ASCII letter-or-digit>`, `ctrl+alt+<ASCII letter>`, and unmodified `f1` through `f12`. `control` aliases `ctrl`, `option` aliases `alt`, and modifier order does not matter. `ctrl+i` and `ctrl+m`, including Ctrl+Alt variants, are rejected because terminals report them as Tab and Enter. Shift, Command/Meta/Super, modified function keys, punctuation, spaces, non-ASCII characters, and navigation-key combinations are unsupported. The native TUI skips reserved host bindings, reports overrides and extension-to-extension conflicts, and shows only effective registrations. Shortcuts currently execute only in local native `kodelet chat` sessions.

```python
from kodelet_sdk import ShortcutContext


@ext.shortcut("ctrl+alt+r", description="Refresh project context")
async def refresh(ctx: ShortcutContext) -> None:
    await ctx.ui.notify("Project context refreshed")
```

Long-running tool handlers can publish transient accumulated snapshots through their context. Each update replaces the previous snapshot for that tool call; only the handler's return value is persisted or sent back to the model:

```python
@ext.tool("search", description="Search a project", input_schema=SearchInput)
async def search(input: SearchInput, ctx: ToolContext) -> ToolExecutionResult:
    await ctx.update(
        "Searching code",
        {"filesScanned": 12},
    )
    return {"content": "Search complete"}
```

`ctx.update(...)` is capability-gated and is a no-op when the connected Kodelet host does not support live extension-tool updates.

When the host cancels an active request or disconnects, async handlers receive `asyncio.CancelledError`. Any late `ctx.update(...)` or transient `ctx.ui.input/confirm/select/notify(...)` call from that cancelled request is rejected rather than being routed to a later call. Persistent transcript, widget, and surface APIs retain the originating conversation's opaque UI scope and remain usable after their opening handler returns.

For long-running tasks with multiple activities, `TaskProgress` publishes a bounded `taskRun` snapshot and can either be updated directly or attached to a child Kodelet session:

```python
progress = TaskProgress(
    ctx,
    kind="code_search",
    task=input.query,
    cwd=ctx.cwd,
    running_title="Searching code",
    completed_title="Searched code",
    failed_title="Code search failed",
    responding_detail="writing summary",
)
await progress.start()
progress.attach(session)
```

Calling `await progress.finish(...)` returns the terminal snapshot and detaches the child-session listeners automatically.

The decorators preserve concrete function signatures for type checkers, so handlers can annotate their inputs and contexts directly:

```python
from kodelet_sdk import (
    CommandContext,
    CommandResult,
    EventContext,
    ToolCallEvent,
    ToolUpdateEvent,
)


@ext.command("doctor", description="Check health", input_schema=WeatherInput)
async def doctor(input: WeatherInput, ctx: CommandContext) -> CommandResult:
    return {"action": "respond", "response": ctx.input["commandName"]}


@ext.on("tool.call")
def approve(event: ToolCallEvent, ctx: EventContext):
    return {"message": event.tool.name}


@ext.on("tool.update")
def sanitize_partial_output(event: ToolUpdateEvent, ctx: EventContext):
    return {"output": event.tool.output}
```

`tool.update` handlers receive transient accumulated structured-result snapshots and may replace the snapshot by returning `{"output": ...}`. An extension that sanitizes `tool.result` should apply the same policy in `tool.update`; Kodelet suppresses partial snapshots when a result-subscribing extension does not also subscribe to updates.

### Pydantic and Jinja2 bridge dependencies

`kodelet-sdk` depends on Pydantic and Jinja2 and re-exports common entry points so extensions can be self-contained:

```python
from kodelet_sdk import BaseModel, Field, Jinja2, Pydantic, render_template


class ReviewInput(BaseModel):
    target: str = Field(min_length=1)


assert render_template("Review {{ target }}", {"target": "main"}) == "Review main"
assert Jinja2.Template("Hello {{ name }}").render(name="Kodelet") == "Hello Kodelet"
assert Pydantic.TypeAdapter(int).validate_python("1") == 1
```

Pydantic input schemas are converted to JSON Schema during initialization and validate incoming tool/command inputs before handlers run. Commands with validation failures return `{"action": "pass"}` so another command route can handle the invocation.

Tools also accept arbitrary raw `JSONSchema` mappings. Raw schemas are forwarded unchanged to Kodelet and inputs are passed directly to the handler; use a Pydantic schema when the Python extension should perform local validation.

### Context helpers

Handlers receive `ctx` with Kodelet call metadata and helper namespaces:

- `ctx.storage.read_text/write_text/read_json/write_json(...)` for extension data files.
- `ctx.path.resolve_workspace_path(...)` and `ctx.path.relative_to_workspace(...)`.
- `ctx.fs.exists/read_text/write_text/list(...)` for workspace file access.
- `ctx.process.exec(...)` and `ctx.process.spawn(...)` for async process execution.
- `ctx.env.get(...)` for environment access.
- `ctx.log.debug/info/warn/error(...)` for JSON logs to stderr.
- `await ctx.acquire_background_task(...)` for a host lifetime lease around work that may outlive the current handler.
- `ctx.ui.input/confirm/select/notify(...)` for host UI reverse-RPC calls.
- `ctx.ui.append_transcript(...)`, `ctx.ui.set_widget(...)`, and `ctx.ui.open_surface(...)` for capability-gated persistent native-TUI content.

UI helpers accept protocol-shaped typed requests: `UIInputRequest`, `UIConfirmRequest`, `UISelectRequest`, and `UINotifyRequest`. The stdio runtime dispatches independent extension requests concurrently and includes the originating request's `parentId` on reverse-RPC calls so Kodelet can route UI interactions to the correct call context.

```python
from kodelet_sdk import UIInputRequest, UISelectRequest

input_request: UIInputRequest = {"title": "Branch name", "required": True}
select_request: UISelectRequest = {"title": "Mode", "options": ["fast", "thorough"]}

branch = await ctx.ui.input(input_request)
mode = await ctx.ui.select(select_request)
```

Background leases retain host runtime resources; they do not persist extension-specific task state. Acquire the lease before returning from the originating handler and close it after the worker and its final state or UI updates complete. Persistent local hosts return a no-op lease, while runner-backed hosts retain the conversation's extension runtime and execution instance until the last lease is released.

```python
lease = await ctx.acquire_background_task("index repository")
try:
    await run_background_worker()
finally:
    await lease.close()
```

The native Kodelet TUI can advertise persistent transcript, widget, and interactive-surface support. `append_transcript(...)` and `set_widget(...)` are no-ops when unavailable; `open_surface(...)` raises `RuntimeError` when surfaces are unavailable. Persistent UI requests retain the originating request's `parentId` while a tool, command, or event handler is active and always carry `ctx.ui_scope_id` as an opaque durable scope, including an explicit empty string for host-global UI. This lets one extension reuse the same widget or surface ID independently in multiple conversations while returned surface handles continue receiving correctly scoped events and publishing frames through the persistent connection.

```python
import asyncio


await ctx.ui.append_transcript({"title": "Drawing saved", "message": "./drawing.png"})

await ctx.ui.set_widget(
    "status",
    [
        "Extension state",
        {"spans": [{"text": " ready", "style": {"foreground": "#00ff00", "bold": True}}]},
    ],
)
await ctx.ui.set_widget("status", ["Moved"], {"placement": "belowComposer"})
await ctx.ui.set_widget("status", None)

surface = await ctx.ui.open_surface(
    {
        "id": "game",
        "initialLines": ["Loading…"],
        "width": "75%",
        "height": "80%",
        "anchor": "center",
        "margin": {"top": 1, "right": 1, "bottom": 1, "left": 1},
    }
)

surface.on_resize(
    lambda event: surface.update([f"Surface size: {event['width']}×{event['height']}"])
)


def handle_input(event):
    if event["kind"] == "key" and event.get("key") == "q":
        asyncio.create_task(surface.close())


surface.on_input(handle_input)
```

`surface.update(...)` is synchronous and replace-in-place. The SDK keeps at most one frame transport write in flight and one replaceable latest pending frame per surface. Input, mouse, focus, blur, and resize notifications share an ordered host-event sequence; stale events are discarded. Surface dimensions accept positive terminal-cell counts or percentage strings such as `"75%"`, anchors cover all corners, edges, and center, and `nonCapturing: True` leaves keyboard focus with the underlying TUI. A failed `await surface.close()` keeps the handle owned and retryable; the ID is released only after the host acknowledges a successful close.

### Testing extensions

Use `create_test_harness` to exercise registrations without spawning a subprocess:

```python
from kodelet_sdk import Extension, create_test_harness


async def test_tool():
    ext = Extension(name="example")

    @ext.tool("echo", description="Echo", input_schema={"type": "object"})
    async def echo(input, ctx):
        return {"content": input["text"]}

    harness = await create_test_harness(ext)
    result = await harness.execute_tool({"name": "echo", "input": {"text": "hi"}})
    assert result == {"content": "hi"}
```

Use `await harness.execute_shortcut({"key": "ctrl+r", "context": {...}})` to invoke a registered shortcut handler in-process.

## Examples

Runnable example extensions live in `examples/`:

- `examples/review/kodelet-extension-review` is a review command extension.
- `examples/workspace/kodelet-extension-workspace` is a workspace helper/policy extension.

From a checked-out SDK repository, run an example with:

```bash
uv run -s examples/review/kodelet-extension-review
```

The `kodelet-extension-*` files are executable wrappers so Kodelet can discover and launch them directly.

## Releases

Package versions are read from `VERSION.txt`. To publish a release, configure PyPI Trusted Publishing for the `Release` workflow, then update and commit `VERSION.txt` manually:

```bash
git add VERSION.txt pyproject.toml uv.lock
git commit -m "chore: release v0.1.0"
make release
```

Pushing the `vX.Y.Z` tag runs the GitHub Actions release workflow, builds the package, and publishes to PyPI using OIDC trusted publishing.

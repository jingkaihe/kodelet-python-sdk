# kodelet-sdk

Run [Kodelet](https://github.com/jingkaihe/kodelet) sessions and write tools, commands, and event handlers in Python.

## Quick start

```bash
uv add kodelet-sdk
```

An executable extension registers handlers and serves them over stdio:

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

## Agent sessions

`Client` uses your configured `kodelet` CLI to connect to a daemon and runner. Set `server` and `runner` on `Client` to select them explicitly. Provider credentials stay on the daemon; session `cwd` refers to the runner's workspace.

Run the following snippets inside an async function. The inline-extension example below includes a complete script.

```python
from kodelet_sdk import Client

client = Client()
session = await client.create_session()
response = await session.run_and_wait(message="what is the meaning of life?")

print(response.content)
await client.close()
```

### Streaming

Choose a profile and subscribe to session events:

```python
from kodelet_sdk import Client

client = Client()
session = await client.create_session(
    profile="work",  # A model profile configured on your daemon.
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

`tool.update` replaces the previous snapshot for that tool call; it is not a delta. Listeners receive every update, while `response.events` keeps only the latest snapshot and the final result.

Session options include:

| Option | Purpose |
| --- | --- |
| `profile` | A daemon profile name, `Profile`, or inline model settings |
| `options` | An `ExecutionOptions` instance or mapping of execution limits and restrictions |
| `environment_profile` | A runner-owned environment profile |
| `cwd` | Working directory on the runner |
| `resume` | An existing conversation ID |

Pass an `ExecutionOptions` instance directly as `create_session(options=...)` for model settings, execution limits, and tool selection. The optional `profile="work"` selects a model profile already configured on the daemon; omit it to use the daemon's default. Inline settings cannot include provider secrets, endpoints, or local prompt paths.

### Extension model profiles

```python
ext = Extension()
SEARCH_PROFILE = ext.register_profile(
    "code-search",
    provider="openai",
    model="gpt-5.6-luna",
    reasoning_effort="none",
    hidden=True,
)
CLAUDE_PROFILE = ext.register_profile(
    "claude",
    provider="anthropic",
    model="claude-sonnet-4-6",
    anthropic={"platform": "anthropic"},
    anthropic_api_access="subscription",
)
session = await client.create_session(profile=SEARCH_PROFILE)
```

`register_profile(name, *, provider, model, hidden=False, **options)` accepts native profile JSON with snake_case keys (no camelCase conversion), using built-in defaults rather than inheriting daemon model/provider settings. Credentials remain daemon-resolved and host restrictions still apply. It returns the name unchanged: an ASCII slug of 1–128 characters (`default` is reserved). `hidden=True` hides the profile from pickers.

### Inline extensions

Pass `extensions=[ext]` to expose local Python callbacks as agent tools. This calculator runs in your Python process; the agent runs on the selected daemon and runner.

```python
import asyncio

from kodelet_sdk import BaseModel, Client, Extension


ext = Extension(name="calculator", version="0.1.0")


class CalculatorInput(BaseModel):
    a: int
    b: int


@ext.tool("calculator", description="Add two integers", input_schema=CalculatorInput)
async def calculator(input: CalculatorInput) -> str:
    return str(input.a + input.b)


async def main() -> None:
    client = Client()
    try:
        session = await client.create_session(extensions=[ext])
        response = await session.run_and_wait("Use calculator to add 123 and 456")
        print(response.content)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
```

Inline extensions require ACP session-extension protocol v1; older hosts fail with a clear error. On resume, reattach extensions in the same order. Callbacks are not saved or replayed, and pending work is cancelled when the channel or session closes.

To customize the prompt, attach an inline `agent.init` handler that returns a `systemPrompt` patch and leave extensions enabled. Prompt hooks work alongside typed session options.

Host calls such as `ctx.update()` and `ctx.fork_conversation()` go to the runner. File, process, and storage helpers remain local. Legacy `extension_transport="unix"` and `"tcp"` options are accepted but ignored; ACP manages the connection.

### Steering

Call `session.steer()` after an event confirms the run is active:

```python
import asyncio

run_active = asyncio.Event()
session.once("assistant.thinking_start", lambda _event: run_active.set())
run_task = asyncio.create_task(session.run_and_wait("Review the persistence implementation"))

await run_active.wait()
steered = await session.steer("Also check transaction boundaries")
response = await run_task
```

Steering requires host support. `injected` means guidance was queued, not necessarily consumed. If the turn has just ended, the result is `promptRequired`; steering never starts a new turn automatically.

## Extension registration

Create an `Extension(name=..., version=...)`, then register synchronous or asynchronous handlers:

| API | Purpose |
| --- | --- |
| `@ext.tool(...)` | Model-callable tool with an input schema |
| `@ext.command(...)` | Slash command or recipe, with optional aliases |
| `@ext.on(event, ...)` | Lifecycle handler, such as `session.start`, `tool.call`, or `agent.end` |
| `@ext.shortcut(...)` | Native TUI keyboard shortcut |
| `ext.run_sync()` / `await ext.run()` | Serve an executable extension over stdio |

### Tool results and progress

Return a string or a mapping with `content` and optional `data` and `error` fields. Use `data["presentation"]` to customize the displayed result without changing what the model receives:

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

`summary` is required; `body` is optional and supports `text` or `markdown`. Hosts may sanitize or truncate display content.

For live progress, call `ctx.update()`. Updates replace earlier snapshots; only the final return value is persisted and sent to the model:

```python
@ext.tool("search", description="Search a project")
async def search(input, ctx: ToolContext) -> str:
    await ctx.update("Searching code", {"filesScanned": 12})
    return "Search complete"
```

`ctx.update()` is a no-op on hosts without progress support. For multi-step tasks, `TaskProgress` tracks activities and can attach to session events; `await progress.finish(...)` ends tracking and detaches listeners.

### Commands and events

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

Commands return one of:

- `{"action": "pass"}` — let another route handle the command.
- `{"action": "respond", "response": "..."}` — respond directly to the user.
- `{"action": "runAgent", "prompt": "..."}` — run the agent with a replacement prompt; optional `display` controls the visible user message.

If you sanitize `tool.result`, apply the same policy to `tool.update` so partial output is also safe to display. Without an update handler, Kodelet suppresses partial output for result-subscribing extensions.

### Keyboard shortcuts

Shortcuts run in local native `kodelet chat` sessions:

```python
from kodelet_sdk import ShortcutContext


@ext.shortcut("ctrl+alt+r", description="Refresh project context")
async def refresh(ctx: ShortcutContext) -> None:
    await ctx.ui.notify("Project context refreshed")
```

Supported chords are `ctrl+<letter>`, `alt+<letter-or-digit>`, `ctrl+alt+<letter>`, and `f1`–`f12` (ASCII, case-insensitive). `ctrl+i` and `ctrl+m` are excluded because terminals treat them as Tab and Enter. Reserved host bindings take precedence.

Handlers can return `{"action": "submit", "message": "/dictate"}` when the host supports shortcut submission.

## Schemas and templates

The SDK re-exports Pydantic types and provides Jinja2 rendering:

```python
from kodelet_sdk import BaseModel, Field, render_template


class ReviewInput(BaseModel):
    target: str = Field(min_length=1)


assert render_template("Review {{ target }}", {"target": "main"}) == "Review main"
```

Pydantic schemas validate inputs before handlers run and generate JSON Schema for the host. Invalid command inputs return `{"action": "pass"}`. Raw JSON Schema mappings are also accepted, but do not perform local input validation.

## Context helpers

Handlers receive `ctx` with call metadata and these helpers:

- `ctx.storage.read_text/write_text/read_json/write_json(...)` for extension data files.
- `ctx.path.resolve_workspace_path(...)` and `ctx.path.relative_to_workspace(...)`.
- `ctx.fs.exists/read_text/write_text/list(...)` for workspace file access.
- `ctx.process.exec(...)` and `ctx.process.spawn(...)` for async process execution.
- `ctx.env.get(...)` for environment access.
- `ctx.log.debug/info/warn/error(...)` for JSON logs to stderr.
- `await ctx.fork_conversation(name=...)` to snapshot the active tool's conversation for an ACP session.
- `await ctx.acquire_background_task(...)` to keep runtime resources alive after a handler returns.
- `ctx.ui.input/confirm/select/notify(...)` for host UI reverse-RPC calls.
- `ctx.ui.append_transcript(...)`, `ctx.ui.set_widget(...)`, and `ctx.ui.open_surface(...)` for capability-gated persistent native-TUI content.

### Agent work inside tools

Use `Client.create_session(options=ExecutionOptions(...))` with normal client credentials and server/runner targeting. Attach `TaskProgress` for activity updates and await client/progress cleanup on every exit path. A cancelled response is not success, even with partial text.

Sessions start fresh. For inherited context, call `ctx.fork_conversation(name="worker-name")` inside the active tool, then pass the returned ID as `resume`. `inherit_context` remains unsupported. Reuse or resume the session for follow-ups; steering only affects a running turn.

### Background work

To let extension work outlive its handler, acquire a lease with `await ctx.acquire_background_task(...)` while the handler is active. Retain and manage the background task yourself, close any ACP client it owns, and release the lease after the work and final UI updates finish. A lease keeps extension resources alive; it does not authorize ACP sessions or keep a completed tool's `ctx.update()` channel open.

Runner leases last at most one hour and end on release, cancellation, or runner/extension shutdown. They keep resources alive, not Python task state; background tasks must be recreated after a restart.

### User input

Executable extensions use the host's UI. For inline extensions, pass `ui={"select": handler, ...}` to `create_session()` to handle requests locally. A local handler needs no terminal unless it uses one; requests without a handler depend on the runner's UI support.

```python
from kodelet_sdk import UIInputRequest, UISelectRequest

input_request: UIInputRequest = {"title": "Branch name", "required": True}
select_request: UISelectRequest = {"title": "Mode", "options": ["fast", "thorough"]}

branch = await ctx.ui.input(input_request)
mode = await ctx.ui.select(select_request)
```

### Persistent TUI content

Use transcript entries, widgets, and surfaces for longer-lived UI. Widget and surface IDs are scoped to the originating conversation.

```python
await ctx.ui.append_transcript({"title": "Drawing saved", "message": "./drawing.png"})
await ctx.ui.set_widget("status", ["Indexing repository…"])

surface = await ctx.ui.open_surface(
    {
        "id": "preview",
        "initialLines": ["Loading…"],
        "width": "75%",
        "height": "80%",
        "anchor": "center",
    }
)

surface.on_resize(
    lambda event: surface.update([f"Surface size: {event['width']}×{event['height']}"])
)
surface.update(["Preview ready"])
await surface.close()
await ctx.ui.set_widget("status", None)  # Remove the widget.
```

`append_transcript()` and `set_widget()` are no-ops without host support; `open_surface()` raises `RuntimeError`. Surface updates replace existing content. If `surface.close()` fails, the handle remains valid for retry.

## Runtime behavior

- Requests run concurrently. Cancellation or disconnection raises `asyncio.CancelledError` in async handlers; late request-scoped UI calls and updates are rejected.
- Persistent widgets and surfaces can outlive the handler that opened them, but remain scoped to their conversation.
- ACP messages are limited to 64 MiB each. Pipe failures and oversized messages fail pending requests and stop the subprocess.

## Testing extensions

Test handlers without a subprocess:

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

## Examples

Run the inline calculator from a checkout with a configured Kodelet daemon:

```bash
uv run -s examples/sdk/inline-extension-session
```

Other examples:

- `examples/sdk/basic-agent-session` — one prompt and its final response.
- `examples/sdk/streaming-agent-session` — live assistant and tool output.
- `examples/review/kodelet-extension-review` — review command extension.
- `examples/workspace/kodelet-extension-workspace` — workspace helper/policy extension.

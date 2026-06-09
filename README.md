# kodelet-sdk

Python SDK for authoring [Kodelet](https://github.com/jingkaihe/kodelet) extensions.

The SDK speaks Kodelet's JSON-RPC extension protocol over stdio and provides an asyncio-first API for registering tools, commands, and event handlers.

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

Use `Client` to launch Kodelet and drive an agent session from Python. The
client speaks to `kodelet acp` over stdio JSON-RPC, so normal profile
resolution, conversation persistence, tools, skills, MCP, and extensions still
come from the Kodelet executable.

```python
from kodelet_sdk import Client

client = Client()
session = await client.create_session()
response = await session.run_and_wait(message="what is the meaning of life?")

print(response.content)
await client.close()
```

Pass a named or inline `Profile` when creating a session, and listen for typed
stream events while a run is active:

```python
from kodelet_sdk import Client, Profile

client = Client(command="kodelet")
session = await client.create_session(
    profile=Profile(
        {
            "provider": "openai",
            "model": "gpt-5.5",
            "openai": {"api_mode": "responses", "service_tier": "fast"},
        }
    ),
    max_turns=4,
    streaming=True,
)

session.on(
    "assistant.message_delta",
    lambda event: print(event.data.deltaContent, end="", flush=True),
)

response = await session.run_and_wait(message="help me choose an approach")
print("\nfinal:", response.content)
await client.close()
```

Agent sessions can expose in-process Python extensions for that session. Inline
extensions are served through a temporary JSON-RPC bridge and are removed when
the session closes.

```python
from kodelet_sdk import BaseModel, Client, Extension


ext = Extension(name="workspace", version="0.1.0")


class AskInput(BaseModel):
    question: str
    options: list[str]


@ext.tool("ask_user_question", description="Ask the user", input_schema=AskInput)
async def ask_user_question(input: AskInput, ctx):
    choice = await ctx.ui.select({"title": input.question, "options": input.options})
    return choice or "dismissed"


client = Client()
session = await client.create_session(
    extensions=[ext],
    ui={"select": lambda request: request["options"][0]},
)
response = await session.run_and_wait(message="ask me to choose")
await client.close()
```

`create_session` accepts either ready-to-use `Extension` instances or entrypoint
callables that receive a fresh `Extension`. Prefer passing an `Extension`
directly for simple scripts and examples; use an entrypoint callable when each
session should build an isolated extension host.

Inline extension bridges use Unix domain sockets by default. If your environment
blocks Unix sockets, use a loopback TCP bridge instead:

```python
session = await client.create_session(
    extensions=[ext],
    extension_transport="tcp",  # binds an ephemeral 127.0.0.1 port
)
```

### Extension registration

- `Extension(name=None, version=None)` creates an extension host.
- `@ext.tool(name=None, description=None, input_schema=None, timeout_in_sec=None)` registers a tool.
- `@ext.command(name=None, description=None, input_schema=None, aliases=None, kind=None, timeout_in_sec=None)` registers a command.
- `@ext.on(event, priority=0, timeout_in_sec=None)` registers an event handler such as `session.start`, `tool.call`, or `agent.end`.
- `await ext.run()` starts the async stdio runtime; `ext.run_sync()` is a synchronous entrypoint convenience.

Handlers may be synchronous or asynchronous. Tool handlers may return a string, which is converted to `{ "content": ... }`, or a protocol-shaped mapping. Command handlers return `{ "action": "pass" }`, `{ "action": "respond", "response": ... }`, or `{ "action": "runAgent", "prompt": ... }`.

The decorators preserve concrete function signatures for type checkers, so handlers can annotate their inputs and contexts directly:

```python
from kodelet_sdk import CommandContext, CommandResult, EventContext, ToolCallEvent


@ext.command("doctor", description="Check health", input_schema=WeatherInput)
async def doctor(input: WeatherInput, ctx: CommandContext) -> CommandResult:
    return {"action": "respond", "response": ctx.input["commandName"]}


@ext.on("tool.call")
def approve(event: ToolCallEvent, ctx: EventContext):
    return {"message": event.tool.name}
```

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

### Context helpers

Handlers receive `ctx` with Kodelet call metadata and helper namespaces:

- `ctx.storage.read_text/write_text/read_json/write_json(...)` for extension data files.
- `ctx.path.resolve_workspace_path(...)` and `ctx.path.relative_to_workspace(...)`.
- `ctx.fs.exists/read_text/write_text/list(...)` for workspace file access.
- `ctx.process.exec(...)` and `ctx.process.spawn(...)` for async process execution.
- `ctx.env.get(...)` for environment access.
- `ctx.log.debug/info/warn/error(...)` for JSON logs to stderr.
- `ctx.ui.input/confirm/select/notify(...)` for host UI reverse-RPC calls.

UI helpers accept protocol-shaped typed requests: `UIInputRequest`, `UIConfirmRequest`, `UISelectRequest`, and `UINotifyRequest`.

```python
from kodelet_sdk import UIInputRequest, UISelectRequest

input_request: UIInputRequest = {"title": "Branch name", "required": True}
select_request: UISelectRequest = {"title": "Mode", "options": ["fast", "thorough"]}

branch = await ctx.ui.input(input_request)
mode = await ctx.ui.select(select_request)
```

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

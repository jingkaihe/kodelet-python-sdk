# kodelet-sdk

Python SDK for authoring [Kodelet](https://github.com/jingkaihe/kodelet) extensions.

The SDK speaks Kodelet's JSON-RPC extension protocol over stdio and provides an asyncio-first API for registering tools, commands, and event handlers.

## Quick start

```python
from kodelet_sdk import BaseModel, Extension

ext = Extension(name="weather", version="0.1.0")


class WeatherInput(BaseModel):
    location: str


@ext.tool("get_weather", description="Get weather", input_schema=WeatherInput)
async def get_weather(input: WeatherInput, ctx):
    return {"content": f"Weather for {input.location}"}


@ext.on("session.start")
async def session_start(event, ctx):
    ctx.log.info("extension started")


if __name__ == "__main__":
    ext.run_sync()
```

## Public API

### Extension registration

- `Extension(name=None, version=None)` creates an extension host.
- `@ext.tool(name=None, description=None, input_schema=None, timeout_in_sec=None)` registers a tool.
- `@ext.command(name=None, description=None, input_schema=None, aliases=None, kind=None, timeout_in_sec=None)` registers a command.
- `@ext.on(event, priority=0, timeout_in_sec=None)` registers an event handler such as `session.start`, `tool.call`, or `agent.end`.
- `await ext.run()` starts the async stdio runtime; `ext.run_sync()` is a synchronous entrypoint convenience.

Handlers may be synchronous or asynchronous. Tool handlers may return a string, which is converted to `{ "content": ... }`, or a protocol-shaped mapping. Command handlers return `{ "action": "pass" }`, `{ "action": "respond", "response": ... }`, or `{ "action": "runAgent", "prompt": ... }`.

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

- `examples/review/extension.py` ports the TypeScript review command extension.
- `examples/workspace/extension.py` ports the TypeScript workspace helper/policy extension.

From a checked-out SDK repository, run an example with:

```bash
uv run -- python examples/review/extension.py
```

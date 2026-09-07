# Python Agent SDK examples

These executable `uv` scripts use the local SDK checkout and your configured Kodelet daemon:

```bash
examples/sdk/basic-agent-session "what is the meaning of life?"
examples/sdk/streaming-agent-session "explain this repository in one paragraph"
examples/sdk/inline-extension-session
```

The inline example registers a local `calculator` tool and asks the agent to add 123 and 456. It logs the callback to stderr and prints the agent's answer. No UI handler is needed. Pass a prompt to try other inputs:

```bash
examples/sdk/inline-extension-session "Use calculator to add 42 and 58"
```

Useful environment variables:

- `KODELET_BIN` — Kodelet executable to launch. Defaults to `kodelet` from `PATH`.
- `KODELET_PROFILE` — optional named profile for the basic and streaming examples.

## Examples

- `basic-agent-session` runs one prompt and prints the final response.
- `streaming-agent-session` streams assistant deltas and accumulated tool-output snapshots as they arrive.
- `inline-extension-session` attaches a local calculator with `create_session(extensions=[ext])`. Requires a daemon/runner with inline-extension support.

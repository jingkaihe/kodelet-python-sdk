# Python Agent SDK examples

These examples show how to use `kodelet_sdk.Client` to launch and drive Kodelet
agent sessions from Python.

They are executable `uv` scripts that use the local SDK checkout:

```bash
examples/sdk/basic-agent-session "what is the meaning of life?"
examples/sdk/streaming-agent-session "explain this repository in one paragraph"
examples/sdk/inline-extension-session
```

Useful environment variables:

- `KODELET_BIN` — Kodelet executable to launch. Defaults to `kodelet` from
  `PATH`.
- `KODELET_PROFILE` — optional named Kodelet profile to use for the session.

## Examples

- `basic-agent-session` runs one prompt and prints the final response.
- `streaming-agent-session` streams assistant deltas as they arrive.
- `inline-extension-session` exposes an in-process Python extension with an
  `sdk_echo` tool for the session.

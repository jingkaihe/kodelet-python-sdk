# kodelet-sdk Python SDK

## Project overview

This repository contains `kodelet-sdk`, a Python SDK for authoring Kodelet extensions. It mirrors the TypeScript SDK protocol shape while using Python idioms: asyncio runtime, decorators, Pydantic schemas, and Jinja2 templates.

## Layout

```text
src/kodelet_sdk/      # SDK source
  api.py              # Extension registration/dispatch API
  context.py          # Tool/command/event context helpers
  runtime.py          # stdio JSON-RPC runtime
  schemas.py          # Pydantic/JSON Schema bridge
  template.py         # Jinja2 rendering helper
  test_harness.py     # In-process extension test harness
examples/            # Runnable example extensions
  review/             # Python port of the review extension
  workspace/          # Python port of the workspace extension
tests/               # pytest coverage
.github/workflows/   # GitHub Actions CI
```

## Tooling

Use `uv` for dependency management and command execution. Do not use raw `pip` or ad-hoc virtualenv commands.

Common commands:

```bash
uv sync
uv run -- ruff check
uv run -- ty check
uv run -- pytest -q
uv build
```

Run all gates before considering a change complete:

```bash
uv run -- ruff check && uv run -- ty check && uv run -- pytest -q && uv build
```

Generated build artifacts under `dist/`, virtualenvs, caches, and `__pycache__` should not be committed.

## Coding conventions

- Keep public APIs documented with useful docstrings and parameter descriptions.
- Prefer asyncio-compatible implementations for runtime/context features.
- Use Pydantic for schema validation and JSON Schema generation.
- Use Jinja2 for template rendering.
- Keep examples importable without side effects; guard runtime startup with `if __name__ == "__main__"`.
- Prefer SDK re-exports in examples (`from kodelet_sdk import BaseModel, Extension, Field, render_template`) so examples are self-contained.
- When adding public functionality, add focused pytest coverage and keep type/lint checks green.

## README guidance

Keep `README.md` focused on users of the SDK: quick start, public API shape, examples, and runtime behavior. Put contributor/agent workflow details here in `AGENTS.md` instead of expanding the README.

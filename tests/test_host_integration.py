"""Opt-in real-host gate: set KODELET_TEST_BINARY to an absolute Kodelet binary.

Only the model HTTP endpoint is a fixture. ACP, daemon, runner, persistence, and
the installed extension all run through real Kodelet/SDK code in isolated state.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

import kodelet_sdk
from kodelet_sdk import Client, ExecutionOptions


@pytest.mark.skipif(not os.getenv("KODELET_TEST_BINARY"), reason="set KODELET_TEST_BINARY")
async def test_real_host_config_follow_ups_reload_and_resume(tmp_path: Path) -> None:
    binary = await asyncio.to_thread(Path(os.environ["KODELET_TEST_BINARY"]).resolve, strict=True)
    sdk_source = (await asyncio.to_thread(Path(kodelet_sdk.__file__).resolve)).parents[1]
    home, state, workspace = (tmp_path / name for name in ("home", "state", "workspace"))
    for directory in (home / ".kodelet/bin", state, workspace / ".kodelet/extensions"):
        directory.mkdir(parents=True)
    # Copy managed search binaries to avoid downloads without touching the live home.
    for name in ("rg", "fd"):
        managed = Path.home() / ".kodelet/bin" / name
        shutil.copy2(managed if managed.is_file() else shutil.which(name) or name,
                     home / ".kodelet/bin" / name)
    receipts = workspace / "receipts.jsonl"
    extension = workspace / ".kodelet/extensions/kodelet-extension-proof"
    extension.write_text(
        f"#!{sys.executable}\nimport sys\nsys.path.insert(0, {str(sdk_source)!r})\n"
        + r'''
import json
import os
from pathlib import Path
from kodelet_sdk import BaseModel, Extension

ext = Extension(name="proof")
settings = None

class Settings(BaseModel):
    marker: str
    hidden: str
    visible: str
    revision: int
    nested: dict

def record(event, ctx):
    with (Path(ctx.cwd) / "receipts.jsonl").open("a") as output:
        output.write(json.dumps({"event": dict(event), "pid": os.getpid(),
                                 "config": getattr(ctx, "config", None)}) + "\n")

@ext.on("session.start")
def configure(event, ctx):
    global settings
    record(event, ctx)
    settings = Settings.model_validate(ctx.config)

@ext.tool("configured_value", description="Read initialized extension settings")
def configured_value(_input, ctx):
    assert settings.marker == ctx.config["marker"]
    return {"content": json.dumps({"marker": settings.marker, "nested": settings.nested})}

@ext.on("turn.end")
def turn_end(event, ctx):
    record(event, ctx)
    return {"followUps": [{"text": settings.hidden, "visibility": "hidden",
                           "key": "hidden-stream", "revision": settings.revision}]}

@ext.on("agent.end")
def agent_end(event, ctx):
    record(event, ctx)
    return {"followUps": [{"text": settings.visible,
                           "key": "visible-stream", "revision": settings.revision}]}

ext.run_sync()
''',
    )
    extension.chmod(0o700)
    requests: list[dict[str, Any]] = []

    def user_texts(request: dict[str, Any]) -> list[str]:
        return ["".join(part["text"] for part in msg["content"])
                for msg in request["messages"] if msg["role"] == "user"]

    async def provider(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            headers = (await reader.readuntil(b"\r\n\r\n")).decode()
            size = next(int(line.split(":", 1)[1]) for line in headers.split("\r\n")
                        if line.lower().startswith("content-length:"))
            request = json.loads(await reader.readexactly(size))
            streaming = request.get("stream", False)
            if streaming:
                requests.append(request)
                (tmp_path / "model-requests.json").write_text(json.dumps(requests, indent=2))
            message: dict[str, Any] = {"role": "assistant", "content": "fixture answer"}
            finish = "stop"
            if streaming and len(requests) == 1:
                message = {"role": "assistant", "tool_calls": [{
                    "index": 0, "id": "read-config", "type": "function",
                    "function": {"name": "configured_value", "arguments": "{}"},
                }]}
                finish = "tool_calls"
            response = {
                "id": "fixture", "object": "chat.completion.chunk" if streaming
                else "chat.completion", "model": "gpt-4o",
                "choices": [{"index": 0, "delta" if streaming else "message": message,
                             "finish_reason": finish}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
            body = json.dumps(response).encode()
            content_type = "application/json"
            if streaming:
                body = b"data: " + body + b"\n\ndata: [DONE]\n\n"
                content_type = "text/event-stream"
            writer.write(
                f"HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\n"
                f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    model = await asyncio.start_server(provider, "127.0.0.1", 0)
    model_port = model.sockets[0].getsockname()[1]
    settings: dict[str, Any] = {
        "marker": uuid4().hex, "hidden": uuid4().hex, "visible": uuid4().hex,
        "revision": 1, "nested": {"enabled": False, "ratio": 0.5, "items": [1, "two"]},
    }
    config: dict[str, Any] = {
        "profile": "proof",
        "profiles": {"proof": {
            "provider": "openai", "model": "gpt-4o", "weak_model": "gpt-4o",
            "max_tokens": 256, "openai": {
                "platform": "openai", "base_url": f"http://127.0.0.1:{model_port}",
                "api_mode": "chat_completions", "api_key_env_var": "KODELET_PROOF_KEY",
            },
        }},
        "extensions": {"enabled": True, "discover_path": False,
                       "config": {"proof": settings}},
        "skills": {"enabled": False}, "tracing": {"enabled": False},
        "serve": {
            "host": "127.0.0.1", "port": 0, "web_auth_mode": "token",
            "runner_auth_mode": "token", "embedded_runner": True,
            "runner_workspace": str(workspace),
            "runner_settings": {"skills": {"enabled": False}, "code_mode": "off"},
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(config))  # JSON is valid YAML.
    config_path.chmod(0o600)
    env = {
        **os.environ, "HOME": str(home), "KODELET_BASE_PATH": str(state),
        "KODELET_CONFIG_FILE": str(config_path), "KODELET_CONFIG_FILE_MODE": "isolated",
        "KODELET_SERVER": "", "KODELET_AUTH_TOKEN": "",
        "KODELET_PROOF_KEY": "local-fixture-only",
    }

    async def cli(*args: str) -> str:
        process = await asyncio.create_subprocess_exec(
            str(binary), *args, cwd=workspace, env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        assert process.returncode == 0, stderr.decode()
        return stdout.decode()

    client: Client | None = None
    try:
        version = json.loads(await cli("version"))
        await cli("server", "start")
        connection = json.loads((state / "server/connection.json").read_text())
        assert connection["url"].startswith("http://127.0.0.1:")
        env["KODELET_AUTH_TOKEN"] = (state / "server/client-token").read_text().strip()
        client = Client(command=str(binary), cwd=workspace, env=env, server=connection["url"])
        options = ExecutionOptions(allowed_tools=["configured_value"], code_mode="off")
        session = await client.create_session(options=options, max_turns=8)
        response = await asyncio.wait_for(session.run_and_wait("Exercise the extension"), 30)
        recorded = [json.loads(line) for line in receipts.read_text().splitlines()]
        initialized = [entry for entry in recorded if entry["event"]["event"] == "session.start"]
        assert initialized and all(entry["config"] == settings for entry in initialized), (
            "host YAML did not reach session.start as ctx.config", initialized
        )
        assert all(entry["pid"] not in (os.getpid(), connection["pid"]) for entry in recorded)
        assert len(requests) == 3, "tool iteration + hidden continuation + visible continuation"
        assert any(entry["event"]["event"] == "turn.end"
                   and entry["event"].get("response") == "" for entry in recorded)
        tool_messages = [msg for msg in requests[1]["messages"] if msg["role"] == "tool"]
        assert len(tool_messages) == 1
        tool_content = tool_messages[0]["content"].removeprefix("<result>\n")
        assert json.loads(tool_content.removesuffix("\n</result>\n")) == {
            "marker": settings["marker"], "nested": settings["nested"],
        }
        for index, expected in enumerate(([], [settings["hidden"]],
                                          [settings["hidden"], settings["visible"]])):
            actual = [text for text in user_texts(requests[index])
                      if text in (settings["hidden"], settings["visible"])]
            assert actual == expected
        updates = [event.data for event in response.events if event.type == "event"
                   and event.data.get("sessionUpdate") == "user_message_chunk"]
        (tmp_path / "acp-events.json").write_text(json.dumps(response.events, indent=2))
        assert len(updates) == 1
        assert updates[0]["_meta"]["kodelet/extensionId"] == "proof"
        assert updates[0]["content"]["text"] == f"Extension: proof\n\n{settings['visible']}"
        assert settings["hidden"] not in json.dumps(response.events)
        assert settings["visible"] not in response.content
        for format_name in ("text", "markdown", "json", "raw"):
            history = await cli("--server", connection["url"], "conversation", "show",
                                session.id, "--format", format_name)
            (tmp_path / f"history-{format_name}.txt").write_text(history)
            assert settings["visible"] in history
            assert (settings["hidden"] in history) == (format_name == "raw")
        print("PASS: YAML -> session.start/Pydantic -> tool; tool-only turn.end; "
              "both follow-up hooks; ACP attribution; hidden/visible history")

        # Restart both ACP and the daemon: deduplication must come from saved state.
        conversation_id = session.id
        await client.close()
        await cli("server", "restart")
        restarted = json.loads((state / "server/connection.json").read_text())
        assert restarted["instanceId"] != connection["instanceId"]
        env["KODELET_AUTH_TOKEN"] = (state / "server/client-token").read_text().strip()
        client = Client(command=str(binary), cwd=workspace, env=env, server=restarted["url"])
        session = await client.create_session(resume=conversation_id, options=options, max_turns=8)
        await asyncio.wait_for(session.run_and_wait("Resume without duplicate notifications"), 30)
        assert len(requests) == 4, "persisted revisions must suppress both duplicate follow-ups"
        for text in (settings["hidden"], settings["visible"]):
            assert user_texts(requests[-1]).count(text) == 1
        print("PASS: daemon restart + saved resume suppress duplicate revisions (one model call)")

        # Change trusted config without restarting the daemon, then reload resources.
        settings.update(hidden=uuid4().hex, visible=uuid4().hex, revision=2)
        config_path.write_text(json.dumps(config))
        await asyncio.wait_for(session.run_and_wait("/reload"), 30)
        assert len(requests) == 4, "resource reload must not call the model"
        await asyncio.wait_for(session.run_and_wait("Consume the revised notifications"), 30)
        assert len(requests) == 7, "higher revisions must trigger both continuations"
        for text in (settings["hidden"], settings["visible"]):
            assert user_texts(requests[-1]).count(text) == 1
        recorded = [json.loads(line) for line in receipts.read_text().splitlines()]
        refreshed = [entry for entry in recorded if entry["event"]["event"] == "session.start"
                     and entry["config"] == settings]
        assert refreshed, "resource reload must initialize Python with updated config"
        assert refreshed[-1]["pid"] != initialized[0]["pid"]
        current = json.loads((state / "server/connection.json").read_text())
        assert current["instanceId"] == restarted["instanceId"], "reload restarted the daemon"
        print("PASS: /reload refreshes ctx.config without daemon restart; revision 2 accepted once")
        evidence = {
            "host": version, "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "sdk_sha256": {name: hashlib.sha256((sdk_source / "kodelet_sdk" / name)
                                               .read_bytes()).hexdigest()
                           for name in ("__init__.py", "api.py", "context.py")},
            "sdk_source": str(sdk_source), "model_requests": len(requests),
            "conversation_id": conversation_id, "isolation": str(tmp_path),
            "provider": "local deterministic HTTP fixture, not a live LLM",
        }
        (tmp_path / "evidence.json").write_text(json.dumps(evidence, indent=2))
        print(json.dumps(evidence, indent=2))
    finally:
        try:
            if client is not None:
                await client.close()
        finally:
            try:
                await cli("server", "stop")
            finally:
                model.close()
                await model.wait_closed()

from __future__ import annotations

import asyncio
import json
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from .types import BridgeTransport


class BridgeEndpoint:
    def __init__(self, transport: BridgeTransport, *, path: str | None = None) -> None:
        self.transport = transport
        self.path = path
        self.host: str | None = None
        self.port: int | None = None


def _normalize_bridge_transport(value: Any) -> BridgeTransport:
    if value is None:
        return "unix"
    if value in {"unix", "tcp"}:
        return cast(BridgeTransport, value)
    raise ValueError("extension_transport must be 'unix' or 'tcp'")


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _unlink_missing(path: str) -> None:
    Path(path).unlink(missing_ok=True)


async def _read_frame(reader: asyncio.StreamReader) -> bytes | None:
    header_lines: list[bytes] = []
    while True:
        line = await reader.readline()
        if line == b"":
            return None
        if line in (b"\r\n", b"\n"):
            break
        header_lines.append(line)
    header = b"".join(header_lines).decode("ascii", errors="replace")
    content_length = _parse_content_length(header)
    try:
        return await reader.readexactly(content_length)
    except asyncio.IncompleteReadError:
        return None


def _parse_content_length(header: str) -> int:
    for line in header.splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() == "content-length":
            content_length = int(value.strip())
            if content_length >= 0:
                return content_length
    raise ValueError("Missing Content-Length header")


async def _write_json_frame(writer: asyncio.StreamWriter, message: Mapping[str, Any]) -> None:
    await _write_frame(writer, json.dumps(message, separators=(",", ":")).encode("utf-8"))


async def _write_frame(writer: asyncio.StreamWriter, payload: bytes) -> None:
    writer.write(b"Content-Length: " + str(len(payload)).encode("ascii") + b"\r\n\r\n" + payload)
    await writer.drain()


def _extension_bridge_endpoint(
    root_dir: str,
    extension_id: str,
    transport: BridgeTransport,
) -> BridgeEndpoint:
    if transport == "unix":
        return BridgeEndpoint(transport, path=str(Path(root_dir) / f"{extension_id}.sock"))
    return BridgeEndpoint(transport)


def _extension_bridge_executable(endpoint: BridgeEndpoint) -> str:
    if endpoint.transport == "unix":
        if endpoint.path is None:
            raise RuntimeError("Unix extension bridge endpoint is missing a socket path")
        endpoint_config: dict[str, Any] = {"transport": "unix", "path": endpoint.path}
    else:
        if endpoint.host is None or endpoint.port is None:
            raise RuntimeError("TCP extension bridge endpoint is missing host/port")
        endpoint_config = {"transport": "tcp", "host": endpoint.host, "port": endpoint.port}

    return f'''#!/usr/bin/env python3
from __future__ import annotations

import json
import socket
import sys
import threading

ENDPOINT = {endpoint_config!r}


def read_frame(stream):
    header = bytearray()
    while True:
        ch = stream.read(1)
        if ch == b"":
            return None
        header += ch
        if header.endswith(b"\\r\\n\\r\\n") or header.endswith(b"\\n\\n"):
            break
    header_text = header.decode("ascii", errors="replace")
    content_length = parse_content_length(header_text)
    payload = stream.read(content_length)
    if len(payload) != content_length:
        return None
    return payload


def parse_content_length(header):
    for line in header.splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() == "content-length":
            parsed = int(value.strip())
            if parsed >= 0:
                return parsed
    raise RuntimeError("Missing Content-Length header")


def framed(payload):
    return b"Content-Length: " + str(len(payload)).encode("ascii") + b"\\r\\n\\r\\n" + payload


stdout_lock = threading.Lock()


def stdin_to_socket(sock):
    while True:
        payload = read_frame(sys.stdin.buffer)
        if payload is None:
            try:
                sock.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            return
        sock.sendall(framed(payload))


def socket_to_stdout(sock_file):
    while True:
        payload = read_frame(sock_file)
        if payload is None:
            return
        with stdout_lock:
            sys.stdout.buffer.write(framed(payload))
            sys.stdout.buffer.flush()


def main():
    if ENDPOINT["transport"] == "unix":
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        address = ENDPOINT["path"]
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        address = (ENDPOINT["host"], ENDPOINT["port"])
    try:
        sock.connect(address)
        sock_file = sock.makefile("rb")
        thread = threading.Thread(target=stdin_to_socket, args=(sock,), daemon=True)
        thread.start()
        socket_to_stdout(sock_file)
    except Exception as exc:
        error_payload = {{
            "level": "error",
            "message": "kodelet SDK extension bridge failed",
            "error": str(exc),
        }}
        sys.stderr.write(json.dumps(error_payload) + "\\n")
        sys.stderr.flush()
        return 1
    finally:
        try:
            sock.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


__all__ = ["BridgeEndpoint"]

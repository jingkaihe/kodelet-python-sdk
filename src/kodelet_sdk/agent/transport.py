from __future__ import annotations

import asyncio
from collections.abc import Sequence
from contextlib import suppress
from typing import cast

from .types import SpawnedProcess, SpawnOptions

ACP_MESSAGE_LIMIT = 64 * 1024 * 1024


class _ACPProcess:
    def __init__(
        self, process: asyncio.subprocess.Process, transport: asyncio.SubprocessTransport
    ) -> None:
        self._process = process
        self._transport = transport
        self.stdin, self.stdout, self.stderr = process.stdin, process.stdout, process.stderr

    def terminate(self) -> None:
        with suppress(ProcessLookupError):
            self._transport.terminate()

    def kill(self) -> None:
        with suppress(ProcessLookupError):
            self._transport.kill()
        # Process.wait also waits for pipe disconnection. A failed reader can
        # leave a paused/full pipe after SIGKILL; close it to permit reaping.
        for fd in (0, 1, 2):
            pipe = self._transport.get_pipe_transport(fd)
            if pipe is not None:
                pipe.close()

    async def wait(self) -> int:
        return await self._process.wait()


async def spawn_acp(command: str, args: Sequence[str], options: SpawnOptions) -> SpawnedProcess:
    loop = asyncio.get_running_loop()
    transport, protocol = await loop.subprocess_exec(
        lambda: asyncio.subprocess.SubprocessStreamProtocol(limit=ACP_MESSAGE_LIMIT, loop=loop),
        command,
        *args,
        cwd=options.get("cwd"),
        env=dict(options.get("env") or {}),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    process = asyncio.subprocess.Process(transport, protocol, loop)
    return cast(SpawnedProcess, _ACPProcess(process, transport))


__all__ = ["ACP_MESSAGE_LIMIT", "spawn_acp"]

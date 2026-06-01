from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from .api import Entrypoint, Extension, create_extension_host
from .context import HostRPCClient, set_active_host_rpc_client


class ExtensionTestHarness:
    """In-process harness for testing extension registrations and handlers."""

    def __init__(self, host: Extension) -> None:
        self._host = host
        self._initialized = False
        self._default_init: dict[str, Any] = {
            "protocolVersion": "2026-05-30",
            "kodelet": {"version": "test"},
            "extension": {
                "id": "test-extension",
                "cwd": os.getcwd(),
                "dataDir": "",
                "config": {},
            },
            "capabilities": {},
        }

    def initialize(self, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Initialize the extension with test defaults.

        Args:
            params: Partial initialization parameters. Nested ``extension``
                values are merged with sensible defaults.

        Returns:
            The same initialization result Kodelet would receive.
        """

        params = params or {}
        extension_param = params.get("extension")
        extension = extension_param if isinstance(extension_param, Mapping) else {}
        merged = {
            **self._default_init,
            **dict(params),
            "extension": {
                **self._default_init["extension"],
                **{str(key): value for key, value in extension.items()},
            },
        }
        self._initialized = True
        return self._host.initialize(merged)

    async def execute_tool(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Execute a registered tool through the extension host."""

        self._ensure_initialized()
        return await self._host.execute_tool(params)

    async def execute_command(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Execute a registered command through the extension host."""

        self._ensure_initialized()
        return await self._host.execute_command(params)

    async def handle_event(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Dispatch an event to registered event handlers."""

        self._ensure_initialized()
        return await self._host.handle_event(params)

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.initialize()


async def create_test_harness(
    entrypoint: Extension | Entrypoint,
    host_rpc_client: HostRPCClient | None = None,
) -> ExtensionTestHarness:
    """Create a test harness for an extension.

    Args:
        entrypoint: Existing extension or registration callable.
        host_rpc_client: Optional fake reverse-RPC client for testing
            ``ctx.ui`` interactions.

    Returns:
        An :class:`ExtensionTestHarness` ready for initialize/execute/event
        calls.
    """

    host = await create_extension_host(entrypoint)
    set_active_host_rpc_client(host_rpc_client)
    return ExtensionTestHarness(host)

from __future__ import annotations

import jinja2 as jinja2
import pydantic as pydantic
from jinja2 import Environment, Template
from pydantic import BaseModel, Field, TypeAdapter

from .api import Extension, create_extension_host, define_extension, on
from .context import (
    CommandContext,
    EnvContext,
    EventContext,
    ExecResult,
    FileSystemContext,
    HostRPCClient,
    LogContext,
    PathContext,
    ProcessContext,
    SharedContext,
    StorageContext,
    ToolContext,
    UIContext,
    set_active_host_rpc_client,
)
from .runtime import run_extension
from .template import render_template
from .test_harness import ExtensionTestHarness, create_test_harness

ExtensionHost = Extension
Pydantic = pydantic
Jinja2 = jinja2

__all__ = [
    "BaseModel",
    "CommandContext",
    "EnvContext",
    "Environment",
    "EventContext",
    "ExecResult",
    "Extension",
    "ExtensionHost",
    "ExtensionTestHarness",
    "Field",
    "FileSystemContext",
    "HostRPCClient",
    "Jinja2",
    "LogContext",
    "PathContext",
    "ProcessContext",
    "Pydantic",
    "SharedContext",
    "StorageContext",
    "Template",
    "ToolContext",
    "TypeAdapter",
    "UIContext",
    "create_extension_host",
    "create_test_harness",
    "define_extension",
    "jinja2",
    "on",
    "pydantic",
    "render_template",
    "run_extension",
    "set_active_host_rpc_client",
]

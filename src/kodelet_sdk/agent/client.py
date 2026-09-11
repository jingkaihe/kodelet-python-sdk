from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Mapping, Sequence
from typing import Any, Unpack, cast

from ..api import Entrypoint, Extension
from ..execution import ExecutionOptions, execution_args
from .rpc import ACPRPCClient
from .session import Session
from .transport import spawn_acp
from .types import (
    AgentUIHandlers,
    ClientOptions,
    CreateSessionOptions,
    Profile,
    SpawnedProcess,
    SpawnFunction,
    SpawnOptions,
)


class Client:
    """Launch and manage Kodelet agent sessions over ACP JSON-RPC."""

    def __init__(
        self,
        options: ClientOptions | None = None,
        *,
        command: str | None = None,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str | None] | None = None,
        spawn: SpawnFunction | None = None,
        server: str | None = None,
        runner: str | None = None,
    ) -> None:
        resolved_options: dict[str, Any] = dict(options or {})
        if server is not None:
            resolved_options["server"] = server
        if runner is not None:
            resolved_options["runner"] = runner
        self._endpoint_args: list[str] = []
        for flag in ("server", "runner"):
            if value := resolved_options.get(flag):
                self._endpoint_args.extend([f"--{flag}", str(value)])
        if command is not None:
            resolved_options["command"] = command
        if cwd is not None:
            resolved_options["cwd"] = cwd
        if env is not None:
            resolved_options["env"] = env
        if spawn is not None:
            resolved_options["spawn"] = spawn

        self._command = str(resolved_options.get("command") or "kodelet")
        self._cwd = str(resolved_options.get("cwd") or os.getcwd())
        self._env = dict(cast(Mapping[str, str | None], resolved_options.get("env") or {}))
        self._spawn = (
            cast(SpawnFunction | None, resolved_options.get("spawn")) or self._default_spawn
        )
        self._sessions: set[Session] = set()
        self._rpcs: set[ACPRPCClient] = set()

    async def create_session(
        self,
        session_options: CreateSessionOptions | None = None,
        **kwargs: Unpack[CreateSessionOptions],
    ) -> Session:
        """Create a new Kodelet ACP session.

        Keyword arguments mirror :class:`CreateSessionOptions`; passing a mapping
        as the first argument is also accepted for parity with the TypeScript SDK.
        Inline ``extensions`` keep callbacks in this Python process and relay
        host RPC through the selected runner. ``extension_transport`` is a
        compatibility no-op; ACP owns transport. ``ui`` handles local UI requests.
        """

        merged_options: dict[str, Any] = {**dict(session_options or {}), **kwargs}
        parent_conversation_id = merged_options.get("parent_conversation_id")
        if "parent_conversation_id" in merged_options:
            if not isinstance(parent_conversation_id, str) or not parent_conversation_id.strip():
                raise ValueError("parent_conversation_id must be a non-empty conversation ID")
            if merged_options.get("resume"):
                raise ValueError(
                    "parent_conversation_id cannot be combined with resume; "
                    "existing conversations retain their parent"
                )
            parent_conversation_id = parent_conversation_id.strip()
        if merged_options.get("inherit_context") is not None:
            raise ValueError(
                "inherit_context is unsupported; call ctx.fork_conversation() "
                "and pass the returned conversation ID as resume"
            )
        extensions = tuple(
            cast(Sequence[Entrypoint | Extension], merged_options.get("extensions") or ())
        )
        if any(not isinstance(ext, Extension) and not callable(ext) for ext in extensions):
            raise TypeError("extensions must contain Extension objects or extension entrypoints")
        resume = merged_options.get("resume")
        cwd = str(merged_options.get("cwd") or self._cwd)
        profile = _normalize_profile(merged_options.get("profile"))
        inline = {key: value for key, value in (profile.config if profile else {}).items()
                  if key != "name"}
        execution = ExecutionOptions.model_validate(inline).to_wire()
        if "options" in merged_options:
            execution.update(ExecutionOptions.model_validate(merged_options["options"]).to_wire())
        if "max_turns" in merged_options:
            execution["maxTurns"] = merged_options["max_turns"]
        execution = ExecutionOptions.model_validate(execution).to_wire()
        args = ["acp", *self._endpoint_args,
                *execution_args(ExecutionOptions.model_validate(execution))]
        if profile and profile.name and profile.is_named_only():
            args.append(f"--profile={profile.name}")
        if merged_options.get("environment_profile"):
            args.append(f"--runner-profile={merged_options['environment_profile']}")
        rpc: ACPRPCClient | None = None
        try:
            env = _clean_env(self._base_env())
            process = await self._spawn_process(
                args,
                {"cwd": os.getcwd(), "env": env, "stdio": ["pipe"] * 3},
            )
            rpc = ACPRPCClient(
                process,
                extensions=extensions,
                ui=cast(AgentUIHandlers | None, merged_options.get("ui")),
            )
            self._rpcs.add(rpc)
            await rpc.initialize()
            session_id = (
                await rpc.load_session(str(resume), cwd)
                if isinstance(resume, str) and resume
                else await rpc.create_session(cwd, parent_conversation_id=parent_conversation_id)
            )
            session = Session(
                self,
                cwd=cwd,
                session_id=session_id,
                rpc=rpc,
                max_turns=cast(int | None, merged_options.get("max_turns")),
            )
            self._sessions.add(session)
            return session
        except BaseException:
            if rpc is not None:
                await rpc.close()
                self._rpcs.discard(rpc)
            raise

    async def createSession(self, *args: Any, **kwargs: Any) -> Session:
        """CamelCase alias for :meth:`create_session`."""

        return await self.create_session(*args, **kwargs)

    async def close(self) -> None:
        """Close all sessions owned by this client."""

        # Include initializing sessions whose caller was cancelled during
        # cleanup. Their shielded RPC cleanup remains owned until confirmed.
        await asyncio.gather(
            *(session.close() for session in list(self._sessions)),
            *(rpc.close() for rpc in list(self._rpcs)),
        )
        self._sessions.clear()
        self._rpcs.clear()

    def _base_env(self) -> dict[str, str | None]:
        env: dict[str, str | None] = dict(os.environ)
        env.update(self._env)
        return env

    async def _spawn_process(self, args: Sequence[str], options: SpawnOptions) -> SpawnedProcess:
        result = self._spawn(self._command, args, options)
        if inspect.isawaitable(result):
            return cast(SpawnedProcess, await result)
        return result

    async def _default_spawn(
        self,
        command: str,
        args: Sequence[str],
        options: SpawnOptions,
    ) -> SpawnedProcess:
        return await spawn_acp(command, args, options)

    def _delete_session(self, session: Session) -> None:
        self._sessions.discard(session)
        self._rpcs.discard(session._rpc)


def _normalize_profile(profile: Any) -> Profile | None:
    if profile is None:
        return None
    if isinstance(profile, Profile):
        return profile
    if isinstance(profile, str) or isinstance(profile, Mapping):
        return Profile(profile)
    raise TypeError("profile must be a profile name, Profile, or mapping")


def _clean_env(env: Mapping[str, str | None]) -> dict[str, str]:
    return {key: str(value) for key, value in env.items() if value is not None}


__all__ = ["Client"]

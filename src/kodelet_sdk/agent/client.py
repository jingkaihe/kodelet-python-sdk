from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Unpack, cast

from ..execution import ExecutionOptions, execution_args
from .bridge import InMemoryExtensionBridge, TempConfig
from .rpc import ACPRPCClient
from .session import Session
from .types import (
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

    async def create_session(
        self,
        session_options: CreateSessionOptions | None = None,
        **kwargs: Unpack[CreateSessionOptions],
    ) -> Session:
        """Create a new Kodelet ACP session.

        Keyword arguments mirror :class:`CreateSessionOptions`; passing a mapping
        as the first argument is also accepted for parity with the TypeScript SDK.
        """

        merged_options: dict[str, Any] = {**dict(session_options or {}), **kwargs}
        if merged_options.get("extensions") or "extension_transport" in merged_options:
            raise ValueError("Inline executable extensions are unsupported remotely; install on "
                             "the runner and use ctx.children for delegated execution")
        if merged_options.get("ui") is not None:
            raise ValueError("Inline extension UI handlers are unsupported by this ACP adapter")
        if merged_options.get("inherit_context") is not None:
            raise ValueError("Use ctx.children for scoped execution; live forking remains "
                             "available separately through ctx.fork_conversation")
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
            rpc = ACPRPCClient(process)
            await rpc.initialize()
            session_id = (
                await rpc.load_session(str(resume), cwd)
                if isinstance(resume, str) and resume
                else await rpc.create_session(cwd)
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
            raise

    async def createSession(self, *args: Any, **kwargs: Any) -> Session:
        """CamelCase alias for :meth:`create_session`."""

        return await self.create_session(*args, **kwargs)

    async def close(self) -> None:
        """Close all sessions owned by this client."""

        await asyncio.gather(*(session.close() for session in list(self._sessions)))
        self._sessions.clear()

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
        process = await asyncio.create_subprocess_exec(
            command,
            *args,
            cwd=options.get("cwd"),
            env=dict(options.get("env") or {}),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return cast(SpawnedProcess, process)

    def _delete_session(self, session: Session) -> None:
        self._sessions.discard(session)


class LaunchConfig:
    def __init__(
        self,
        *,
        args: Sequence[str],
        env: Mapping[str, str],
        temp_config: TempConfig | None = None,
        config_file_mode: str | None = None,
    ) -> None:
        self.args = list(args)
        self.env = dict(env)
        self.temp_config = temp_config
        self.config_file_mode = config_file_mode


def _normalize_profile(profile: Any) -> Profile | None:
    if profile is None:
        return None
    if isinstance(profile, Profile):
        return profile
    if isinstance(profile, str) or isinstance(profile, Mapping):
        return Profile(profile)
    raise TypeError("profile must be a profile name, Profile, or mapping")


def _option_int(options: Mapping[str, Any], key: str) -> int | None:
    value = options.get(key)
    return value if isinstance(value, int) else None


def _acp_server_args(options: Mapping[str, Any]) -> list[str]:
    max_turns = _option_int(options, "max_turns")
    if max_turns is not None and max_turns > 0:
        return ["--max-turns", str(max_turns)]
    return []


async def _build_launch_config(
    profile: Profile | None,
    bridge: InMemoryExtensionBridge | None,
) -> LaunchConfig:
    resolved = profile.to_launch_config() if profile is not None else None
    profile_config = resolved.get("config") if resolved else None
    config: dict[str, Any] = {}
    if isinstance(profile_config, Mapping):
        config.update(profile_config)
        if profile_config.get("profile") is None:
            config["profile"] = "default"
    if bridge is not None:
        config["extensions"] = bridge.config()
    config = _prune_none(config)
    if not config:
        return LaunchConfig(args=cast(Sequence[str], (resolved or {}).get("args") or []), env={})

    temp_config = await TempConfig.create(config)
    config_file_mode = "isolated" if isinstance(profile_config, Mapping) else "merge"
    return LaunchConfig(
        args=cast(Sequence[str], (resolved or {}).get("args") or []),
        env={"KODELET_CONFIG_FILE": temp_config.path, "KODELET_CONFIG_FILE_MODE": config_file_mode},
        temp_config=temp_config,
        config_file_mode=config_file_mode,
    )


def _prune_none(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if item is None:
            continue
        if isinstance(item, Mapping):
            result[str(key)] = _prune_none(item)
        else:
            result[str(key)] = item
    return result


def _clean_env(env: Mapping[str, str | None]) -> dict[str, str]:
    return {key: str(value) for key, value in env.items() if value is not None}


def _resolve_path(path: str) -> str:
    return str(Path(path).resolve(strict=False))


__all__ = ["Client", "LaunchConfig"]

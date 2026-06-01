from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Mapping
from typing import Any

from pydantic import BaseModel


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def to_plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        return {str(key): to_plain(item) for key, item in value.items() if item is not None}
    if isinstance(value, list | tuple):
        return [to_plain(item) for item in value]
    return value


def json_clone(value: Any) -> Any:
    return json.loads(json.dumps(to_plain(value)))


def normalize_command_name(name: str) -> str:
    return name.strip().lstrip("/")


def optional_timeout(timeout_in_sec: float | None) -> dict[str, float]:
    if timeout_in_sec is None:
        return {}
    return {"timeoutInSec": timeout_in_sec}


def merge_timeout_in_sec(current: float | None, next_value: float | None) -> float | None:
    if current == 0 or next_value == 0:
        return 0
    if current is None:
        return next_value
    if next_value is None:
        return current
    return max(current, next_value)


class AttrDict(dict[str, Any]):
    """Dictionary with attribute access for ergonomic event payload handling."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


ConvertableAwaitable = Awaitable[Any] | Any

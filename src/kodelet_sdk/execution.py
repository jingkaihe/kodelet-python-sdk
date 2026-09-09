"""Typed ACP execution and extension profile options."""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator
from pydantic.alias_generators import to_camel


class ExecutionOptions(BaseModel):
    """Typed daemon contract; unknown settings and explicit null are rejected."""

    model_config = ConfigDict(extra="forbid", strict=True, alias_generator=to_camel,
                              populate_by_name=True)
    provider: Literal["openai", "anthropic"] | None = None
    model: str | None = Field(default=None, min_length=1)
    weak_model: str | None = Field(default=None, min_length=1)
    max_tokens: int | None = Field(default=None, gt=0)
    weak_model_max_tokens: int | None = Field(default=None, gt=0)
    thinking_budget_tokens: int | None = Field(default=None, ge=0)
    reasoning_effort: str | None = Field(default=None, min_length=1)
    max_turns: int | None = Field(default=None, ge=0)
    use_weak_model: bool | None = None
    no_tools: bool | None = None
    no_extensions: bool | None = None
    no_skills: bool | None = None
    allowed_tools: list[Annotated[str, Field(min_length=1)]] | None = None
    allowed_commands: list[Annotated[str, Field(min_length=1)]] | None = None
    enable_fs_search_tools: bool | None = Field(default=None, alias="enableFSSearchTools")

    @model_validator(mode="before")
    @classmethod
    def reject_null(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and any(item is None for item in value.values()):
            raise ValueError("Omit absent execution options; null is not inheritance")
        return value

    def to_wire(self) -> dict[str, Any]:
        """Return an independent camelCase protocol snapshot."""
        return self.model_dump(by_alias=True, exclude_none=True)


class ExtensionProfileOptions(BaseModel):
    """Native profile JSON; configuration semantics are validated by the daemon."""

    model_config = ConfigDict(extra="allow", strict=True, allow_inf_nan=False)

    __pydantic_extra__: dict[str, JsonValue] = Field(init=False)
    provider: Literal["openai", "anthropic"]
    model: str

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if not value.strip() or "\0" in value:
            raise ValueError("Profile model must be nonempty and contain no NUL")
        return value

    def to_wire(self) -> dict[str, Any]:
        """Return an independent snapshot without rewriting profile keys or values."""
        return self.model_dump()


def execution_args(options: ExecutionOptions) -> list[str]:
    """Encode typed options for the daemon-only ACP client, including empty/false values."""
    args: list[str] = []
    for name, value in options.to_wire().items():
        flag = re.sub(r"[A-Z]", lambda match: "-" + match[0].lower(), name)
        if name == "enableFSSearchTools":
            flag = "enable-fs-search-tools"
        if isinstance(value, list):
            output = io.StringIO()
            csv.writer(output, quoting=csv.QUOTE_ALL, lineterminator="").writerow(value)
            encoded = output.getvalue()
        elif isinstance(value, bool):
            encoded = str(value).lower()
        else:
            encoded = str(value)
        args.append(f"--{flag}={encoded}")
    return args

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, TypeAlias

from pydantic import BaseModel, TypeAdapter

SchemaLike: TypeAlias = type[BaseModel] | TypeAdapter[Any] | Mapping[str, Any] | None

_DEFAULT_JSON_SCHEMA: dict[str, Any] = {"type": "object", "additionalProperties": True}


class SchemaAdapter:
    def __init__(self, schema: SchemaLike = None) -> None:
        self._schema = schema
        self._adapter: TypeAdapter[Any] | None = None
        if schema is not None and not isinstance(schema, Mapping):
            if isinstance(schema, TypeAdapter):
                self._adapter = schema
            else:
                self._adapter = TypeAdapter(schema)

    @property
    def has_validation(self) -> bool:
        return self._adapter is not None

    def json_schema(self) -> dict[str, Any]:
        if self._adapter is not None:
            schema = self._adapter.json_schema()
            return dict(schema)
        if isinstance(self._schema, Mapping):
            return {str(key): value for key, value in self._schema.items()}
        return dict(_DEFAULT_JSON_SCHEMA)

    def validate(self, value: Any) -> Any:
        if self._adapter is None:
            return {} if value is None else value
        return self._adapter.validate_python({} if value is None else value)


def infer_schema_from_callable(func: Any) -> SchemaLike:
    signature = inspect.signature(func)
    for parameter in signature.parameters.values():
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            continue
        annotation = parameter.annotation
        if annotation is inspect.Signature.empty:
            return None
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return annotation
        return None
    return None

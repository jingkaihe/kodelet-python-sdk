from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jinja2 import BaseLoader, Environment, StrictUndefined

_DEFAULT_ENV = Environment(
    loader=BaseLoader(),
    autoescape=False,
    undefined=StrictUndefined,
)


def render_template(template: str, view: Mapping[str, Any] | None = None) -> str:
    """Render a Jinja2 template string with the provided view.

    Args:
        template: Jinja2 template source.
        view: Optional mapping of template variables.

    Returns:
        Rendered string.

    Raises:
        jinja2.exceptions.UndefinedError: If the template references a missing
            variable. The SDK uses ``StrictUndefined`` so mistakes fail fast.
    """

    return _DEFAULT_ENV.from_string(template).render(dict(view or {}))

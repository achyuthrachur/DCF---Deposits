"""User input interface components (CLI/GUI placeholders)."""

from __future__ import annotations

from typing import Any, Callable

try:
    from .cli import app  # type: ignore  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != "typer":
        raise

    def _missing_cli(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(
            "The command-line interface is unavailable because the 'typer' package "
            "was not bundled. Reinstall with Typer to enable CLI commands."
        ) from exc

    app: Callable[..., None] = _missing_cli

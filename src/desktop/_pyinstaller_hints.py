"""Import hints so PyInstaller bundles optional runtime dependencies."""

from __future__ import annotations

from importlib import import_module


def _safe_import(name: str) -> None:
    try:  # pragma: no cover - executed during build/packaging
        import_module(name)
    except Exception:
        pass


for _mod in {
    "altair",
    "docx",
    "fredapi",
    "imageio",
    "matplotlib",
    "numpy",
    "openpyxl",
    "pandas",
    "plotly",
    "pyarrow",
    "pydeck",
    "pydantic",
    "pydantic_core",
    "python_docx",
    "requests",
    "rich",
    "scipy",
    "seaborn",
    "streamlit",
    "tenacity",
    "typer",
    "yaml",
}:
    _safe_import(_mod)

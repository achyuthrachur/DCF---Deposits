"""Background job infrastructure with dynamic driver selection.

The module switches between the local multiprocessing worker and the GitHub
Actions remote worker based on environment variables. Streamlit can therefore
inject credentials at runtime and call ``refresh_driver()`` to use the remote
path without restarting the Python interpreter.
"""

from __future__ import annotations

import logging
import os
from types import ModuleType
from typing import Any

from .job_manager import AnalysisJobHandle

LOGGER = logging.getLogger(__name__)

_DRIVER_MODULE: ModuleType | None = None
USING_GITHUB_DRIVER = False
GITHUB_DRIVER_ERROR: str | None = None


def _github_env_present() -> bool:
    return bool(os.environ.get("GH_REPO") and (os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")))


def _load_driver(*, force_reload: bool = False) -> ModuleType:
    """Return the active driver module, reloading if requested."""

    global _DRIVER_MODULE, USING_GITHUB_DRIVER, GITHUB_DRIVER_ERROR

    if not force_reload and _DRIVER_MODULE is not None:
        return _DRIVER_MODULE

    if _github_env_present():
        try:
            from . import github_actions_driver as gh_driver

            _DRIVER_MODULE = gh_driver
            USING_GITHUB_DRIVER = True
            GITHUB_DRIVER_ERROR = None
            LOGGER.info("Using GitHub Actions job driver")
            return _DRIVER_MODULE
        except Exception as exc:  # pragma: no cover - defensive fallback
            LOGGER.warning("GitHub driver unavailable (%s); falling back to local worker.", exc)
            GITHUB_DRIVER_ERROR = str(exc)

    from . import job_manager as local_driver

    _DRIVER_MODULE = local_driver
    USING_GITHUB_DRIVER = False
    if force_reload or _github_env_present():
        LOGGER.info("Using local job driver")
    return _DRIVER_MODULE


def refresh_driver() -> None:
    """Re-evaluate the driver (call after setting GH_* environment variables)."""

    _load_driver(force_reload=True)


def launch_analysis_job(*args: Any, **kwargs: Any):
    return _load_driver().launch_analysis_job(*args, **kwargs)


def read_job_status(*args: Any, **kwargs: Any):
    return _load_driver().read_job_status(*args, **kwargs)


def load_job_results(*args: Any, **kwargs: Any):
    return _load_driver().load_job_results(*args, **kwargs)


def load_job_bundle(*args: Any, **kwargs: Any):
    return _load_driver().load_job_bundle(*args, **kwargs)


def cleanup_job_artifacts(*args: Any, **kwargs: Any):
    return _load_driver().cleanup_job_artifacts(*args, **kwargs)


def cancel_job(*args: Any, **kwargs: Any):
    driver = _load_driver()
    cancel = getattr(driver, "cancel_job", None)
    if cancel is None:
        LOGGER.info("Active driver does not support cancellation.")
        return None
    return cancel(*args, **kwargs)


__all__ = [
    "AnalysisJobHandle",
    "launch_analysis_job",
    "read_job_status",
    "load_job_results",
    "load_job_bundle",
    "cleanup_job_artifacts",
    "cancel_job",
    "refresh_driver",
    "USING_GITHUB_DRIVER",
    "GITHUB_DRIVER_ERROR",
]


# Initialise driver based on current environment (likely local during import).
_load_driver()

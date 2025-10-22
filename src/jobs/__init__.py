"""Background job infrastructure for long-running analyses.

Chooses between local (multiprocess) driver and GitHub Actions remote driver
based on environment variables. If `GH_REPO` and `GH_TOKEN` are present,
the GitHub driver is used; otherwise, falls back to local.
"""

from __future__ import annotations

import logging
import os

from .job_manager import AnalysisJobHandle as _LocalHandle

LOGGER = logging.getLogger(__name__)
USING_GITHUB_DRIVER = False
GITHUB_DRIVER_ERROR: str | None = None


def _use_github() -> bool:
    return bool(os.environ.get("GH_REPO") and (os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")))


if _use_github():
    try:
        from .github_actions_driver import (  # type: ignore
            launch_analysis_job,
            read_job_status,
            load_job_results,
            load_job_bundle,
            cleanup_job_artifacts,
        )
        AnalysisJobHandle = _LocalHandle
        USING_GITHUB_DRIVER = True
    except Exception as exc:  # pragma: no cover - defensive fallback
        LOGGER.warning("GitHub driver unavailable (%s); falling back to local worker.", exc)
        GITHUB_DRIVER_ERROR = str(exc)
        from .job_manager import (
            AnalysisJobHandle,
            launch_analysis_job,
            read_job_status,
            load_job_results,
            load_job_bundle,
            cleanup_job_artifacts,
        )
else:
    from .job_manager import (
        AnalysisJobHandle,
        launch_analysis_job,
        read_job_status,
        load_job_results,
        load_job_bundle,
        cleanup_job_artifacts,
    )


__all__ = [
    "AnalysisJobHandle",
    "launch_analysis_job",
    "read_job_status",
    "load_job_results",
    "load_job_bundle",
    "cleanup_job_artifacts",
    "USING_GITHUB_DRIVER",
    "GITHUB_DRIVER_ERROR",
]

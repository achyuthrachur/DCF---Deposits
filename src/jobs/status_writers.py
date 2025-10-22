"""Status writer that commits updates to a Git repository (results branch)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from .job_manager import JobStatusWriter, JobStatus


class RepoJobStatusWriter(JobStatusWriter):
    """Extends JobStatusWriter to commit and push status updates.

    Assumes `repo_root` points to a checked-out Git repository with the
    results branch already checked out (HEAD).
    """

    def __init__(self, status_path: Path, job_id: str, *, repo_root: Path) -> None:
        super().__init__(status_path, job_id)
        self.repo_root = Path(repo_root)

    def _git(self, *args: str) -> None:
        subprocess.run(["git", "-C", str(self.repo_root), *args], check=True)

    def write(self, **updates: Any) -> JobStatus:
        status = super().write(**updates)
        rel = self._rel_path(self.status_path)
        try:
            self._git("add", rel)
            msg = f"Update status for job {self.job_id}"
            # Commit may fail if no changes; ignore non-zero in that case.
            subprocess.run(
                ["git", "-C", str(self.repo_root), "commit", "-m", msg],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._git("push")
        except Exception:
            # Best-effort; status file still exists locally for next push.
            pass
        return status

    def _rel_path(self, path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(self.repo_root.resolve()))
        except Exception:
            return str(path)


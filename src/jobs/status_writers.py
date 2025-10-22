"""Status writer that commits updates to a Git repository (results branch)."""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Any

from .job_manager import JobStatusWriter, JobStatus

LOGGER = logging.getLogger(__name__)


class RepoJobStatusWriter(JobStatusWriter):
    """Extends JobStatusWriter to commit and push status updates with retries.

    Assumes `repo_root` points to a checked-out Git repository with the
    results branch already checked out (HEAD).
    """

    def __init__(self, status_path: Path, job_id: str, *, repo_root: Path) -> None:
        super().__init__(status_path, job_id)
        self.repo_root = Path(repo_root)
        self.branch = self._git_capture("rev-parse", "--abbrev-ref", "HEAD").strip() or "results"

    def _git(self, *args: str, check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(self.repo_root), *args],
            check=check,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            text=True,
        )

    def _git_capture(self, *args: str) -> str:
        proc = self._git(*args, capture_output=True)
        return (proc.stdout or "").strip()

    def write(self, **updates: Any) -> JobStatus:
        status = super().write(**updates)
        rel = self._rel_path(self.status_path)
        try:
            self._git("add", rel)
            msg = f"Update status for job {self.job_id}"
            commit_proc = self._git("commit", "-m", msg, check=False, capture_output=True)
            if commit_proc.returncode not in (0, 1):  # 1 indicates no changes to commit
                LOGGER.warning("Status commit failed: %s", commit_proc.stderr)
            self._push_with_retry()
        except Exception as exc:
            LOGGER.warning("Unable to push status update for %s: %s", self.job_id, exc)
        return status

    def _push_with_retry(self, attempts: int = 3) -> None:
        for idx in range(attempts):
            push_proc = self._git("push", check=False, capture_output=True)
            if push_proc.returncode == 0:
                return
            LOGGER.info("Push failed for %s (attempt %s/%s): %s", self.job_id, idx + 1, attempts, push_proc.stderr)
            # Reconcile with remote and retry
            self._git("fetch", "origin", self.branch, check=False)
            self._git("rebase", f"origin/{self.branch}", check=False)
            time.sleep(0.5)
        LOGGER.warning("Gave up pushing status for %s after %s attempts", self.job_id, attempts)

    def _rel_path(self, path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(self.repo_root.resolve()))
        except Exception:
            return str(path)

"""Status writer that publishes updates to GitHub via the REST API."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import base64
import requests

from .job_manager import JobStatusWriter, JobStatus


LOGGER = logging.getLogger(__name__)
GH_API = "https://api.github.com"


class RepoJobStatusWriter(JobStatusWriter):
    """Push job status updates directly to GitHub (no git subprocesses)."""

    def __init__(self, status_path: Path, job_id: str, *, repo_root: Path) -> None:
        super().__init__(status_path, job_id)
        self.repo_root = Path(repo_root)
        self.repo_slug = (
            os.environ.get("GH_REPO")
            or os.environ.get("GITHUB_REPOSITORY")
            or ""
        )
        self.token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        self.branch = (
            os.environ.get("GH_RESULTS_BRANCH")
            or os.environ.get("GITHUB_REF_NAME")
            or os.environ.get("GITHUB_REF", "results")
        )
        if self.branch.startswith("refs/heads/"):
            self.branch = self.branch[len("refs/heads/"):]
        self._sha_cache: dict[str, str] = {}
        self._last_push_ts = 0.0

    def write(self, **updates: Any) -> JobStatus:
        status = super().write(**updates)
        if not self.repo_slug or not self.token:
            return status

        now = time.time()
        if now - self._last_push_ts < 0.25:
            return status  # Throttle API calls slightly

        rel_path = self._rel_path(self.status_path)
        payload_bytes = json.dumps(status.to_dict(), indent=2).encode("utf-8")
        try:
            current_sha = self._sha_cache.get(rel_path) or self._get_sha(rel_path)
            new_sha = self._put_contents(
                rel_path,
                payload_bytes,
                message=f"Update status for job {self.job_id}",
                sha=current_sha,
            )
            if new_sha:
                self._sha_cache[rel_path] = new_sha
            self._last_push_ts = now
        except Exception as exc:  # pragma: no cover - best effort
            LOGGER.warning("Unable to publish status for %s: %s", self.job_id, exc)
        return status

    # ------------------------------------------------------------------ Helpers
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _rel_path(self, path: Path) -> str:
        try:
            rel = Path(path).resolve().relative_to(self.repo_root.resolve())
        except Exception:
            rel = Path(path)
        return rel.as_posix()

    def _get_sha(self, rel_path: str) -> Optional[str]:
        url = f"{GH_API}/repos/{self.repo_slug}/contents/{rel_path}"
        params = {"ref": self.branch, "_": str(time.time_ns())}
        resp = requests.get(url, headers=self._headers(), params=params, timeout=10)
        if resp.status_code == 200:
            try:
                return resp.json().get("sha")
            except Exception:  # pragma: no cover
                return None
        return None

    def _put_contents(self, rel_path: str, content: bytes, *, message: str, sha: Optional[str] = None) -> Optional[str]:
        url = f"{GH_API}/repos/{self.repo_slug}/contents/{rel_path}"
        attempts = 3
        last_error: Optional[str] = None
        last_status: Optional[int] = None
        payload_base = {
            "message": message,
            "content": base64.b64encode(content).decode("utf-8"),
            "branch": self.branch,
        }
        for _ in range(attempts):
            payload = dict(payload_base)
            if sha:
                payload["sha"] = sha
            resp = requests.put(url, headers=self._headers(), json=payload, timeout=10)
            last_status = resp.status_code
            if resp.status_code in (200, 201):
                try:
                    data = resp.json()
                    return data.get("content", {}).get("sha")
                except Exception:
                    return None
            if resp.status_code == 409:
                sha = self._get_sha(rel_path)
                last_error = resp.text
                time.sleep(0.3)
                continue
            last_error = resp.text
            break
        raise RuntimeError(f"Failed to update {rel_path}: {last_status} {last_error}")

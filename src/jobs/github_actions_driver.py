"""GitHub Actions remote job driver.

This driver stores job inputs and status in a dedicated results branch,
dispatches a workflow run for each batch, and aggregates results on read.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

from .aggregate import merge_engine_results
from .job_manager import AnalysisJobHandle, JobStatus
from src.models.results import EngineResults


GH_API = "https://api.github.com"


STATUS_CACHE: Dict[str, JobStatus] = {}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(name)
    return val if val is not None else default


@dataclass
class GHConfig:
    owner: str
    repo: str
    token: str
    workflow: str = "analysis.yml"
    workflow_ref: str = "main"
    results_branch: str = "results"
    jobs_root: str = "output/jobs"
    max_sims_per_job: int = 1000000  # default 'no batching'
    max_steps_per_job: Optional[int] = None

    @classmethod
    def from_env(cls) -> "GHConfig":
        repo_full = _env("GH_REPO")
        token = _env("GH_TOKEN") or _env("GITHUB_TOKEN")
        if not repo_full or not token:
            raise RuntimeError("GitHub Actions driver requires GH_REPO and GH_TOKEN env vars")
        if "/" not in repo_full:
            raise RuntimeError("GH_REPO must be in the form 'owner/repo'")
        owner, repo = repo_full.split("/", 1)
        workflow = _env("GH_WORKFLOW", "analysis.yml")
        workflow_ref = _env("GH_WORKFLOW_REF", "main")
        results_branch = _env("GH_RESULTS_BRANCH", "results")
        jobs_root = _env("GH_JOBS_ROOT", "output/jobs")
        sims_limit = int(_env("GH_MAX_SIMS_PER_JOB", "1000000"))
        steps_limit_env = _env("GH_MAX_STEPS_PER_JOB")
        steps_limit = int(steps_limit_env) if steps_limit_env else None
        return cls(
            owner=owner,
            repo=repo,
            token=token,
            workflow=workflow,
            workflow_ref=workflow_ref,
            results_branch=results_branch,
            jobs_root=jobs_root,
            max_sims_per_job=sims_limit,
            max_steps_per_job=steps_limit,
        )


def _gh_headers(cfg: GHConfig) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {cfg.token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _gh_put_contents(
    cfg: GHConfig,
    path: str,
    content_bytes: bytes,
    message: str,
    *,
    sha: Optional[str] = None,
    attempts: int = 3,
) -> Optional[str]:
    url = f"{GH_API}/repos/{cfg.owner}/{cfg.repo}/contents/{path}"
    last_error: Optional[str] = None
    last_status: Optional[int] = None
    for attempt in range(attempts):
        payload = {
            "message": message,
            "content": base64.b64encode(content_bytes).decode("utf-8"),
            "branch": cfg.results_branch,
        }
        if sha:
            payload["sha"] = sha
        r = requests.put(url, headers=_gh_headers(cfg), json=payload)
        last_status = r.status_code
        if r.status_code in (200, 201):
            try:
                data = r.json()
                content = data.get("content") or {}
                return content.get("sha")
            except Exception:
                return None
        if r.status_code == 409 and attempt < attempts - 1:
            sha = _gh_get_file_sha(cfg, path)
            last_error = r.text
            time.sleep(0.3)
            continue
        last_error = r.text
        break
    raise RuntimeError(f"Failed to write {path}: {last_status} {last_error}")


def _ensure_results_branch(cfg: GHConfig) -> None:
    # Ensure results branch exists; if not, create from workflow_ref
    ref_url = f"{GH_API}/repos/{cfg.owner}/{cfg.repo}/git/ref/heads/{cfg.results_branch}"
    r = requests.get(ref_url, headers=_gh_headers(cfg))
    if r.status_code == 200:
        return
    base_ref_url = f"{GH_API}/repos/{cfg.owner}/{cfg.repo}/git/ref/heads/{cfg.workflow_ref}"
    base = requests.get(base_ref_url, headers=_gh_headers(cfg))
    if base.status_code != 200:
        raise RuntimeError(f"Base branch not found: {cfg.workflow_ref}")
    sha = base.json().get("object", {}).get("sha")
    if not sha:
        raise RuntimeError("Unable to resolve base branch sha")
    create_url = f"{GH_API}/repos/{cfg.owner}/{cfg.repo}/git/refs"
    payload = {"ref": f"refs/heads/{cfg.results_branch}", "sha": sha}
    cr = requests.post(create_url, headers=_gh_headers(cfg), json=payload)
    if cr.status_code not in (201,):
        raise RuntimeError(f"Failed to create results branch: {cr.status_code} {cr.text}")


def _gh_get_raw_file(cfg: GHConfig, path: str) -> Optional[bytes]:
    """Fetch file contents for the results branch while bypassing CDN caching."""

    url = f"{GH_API}/repos/{cfg.owner}/{cfg.repo}/contents/{path}"
    headers = _gh_headers(cfg)
    headers["Accept"] = "application/vnd.github.raw"
    params = {"ref": cfg.results_branch, "_": str(time.time_ns())}
    r = requests.get(url, headers=headers, params=params)
    if r.status_code == 200:
        return r.content
    return None


def _gh_get_file_sha(cfg: GHConfig, path: str) -> Optional[str]:
    url = f"{GH_API}/repos/{cfg.owner}/{cfg.repo}/contents/{path}"
    params = {"ref": cfg.results_branch, "_": str(time.time_ns())}
    r = requests.get(url, headers=_gh_headers(cfg), params=params)
    if r.status_code == 200:
        try:
            data = r.json()
            return data.get("sha")
        except Exception:
            return None
    return None


def _gh_dispatch_workflow(cfg: GHConfig, inputs: Dict[str, Any]) -> None:
    url = f"{GH_API}/repos/{cfg.owner}/{cfg.repo}/actions/workflows/{cfg.workflow}/dispatches"
    payload = {"ref": cfg.workflow_ref, "inputs": inputs}
    r = requests.post(url, headers=_gh_headers(cfg), json=payload)
    if r.status_code not in (201, 204):
        raise RuntimeError(f"Failed to dispatch workflow: {r.status_code} {r.text}")


def _job_index_path(cfg: GHConfig, job_id: str) -> str:
    return f"{cfg.jobs_root}/{job_id}/index.json"


def _batch_dir(cfg: GHConfig, job_id: str, batch_idx: int) -> str:
    return f"{cfg.jobs_root}/{job_id}/batch-{batch_idx+1}"


def _status_path_for_batch(cfg: GHConfig, job_id: str, batch_idx: int) -> str:
    return f"{_batch_dir(cfg, job_id, batch_idx)}/status.json"


def _read_status_bytes(cfg: GHConfig, path: str) -> Optional[Dict[str, Any]]:
    content = _gh_get_raw_file(cfg, path)
    if not content:
        return None
    try:
        return json.loads(content.decode("utf-8"))
    except Exception:
        return None


def _read_index(cfg: GHConfig, job_id: str) -> Optional[Dict[str, Any]]:
    data = _read_status_bytes(cfg, _job_index_path(cfg, job_id))
    return data


def _write_index(cfg: GHConfig, job_id: str, data: Dict[str, Any]) -> None:
    path = _job_index_path(cfg, job_id)
    _gh_put_contents(cfg, path, json.dumps(data, indent=2).encode("utf-8"), message=f"Init job {job_id}")


def _serialize_df_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    df.to_parquet(bio)
    return bio.getvalue()


# ----------------------------------------------------------------- Driver API

def launch_analysis_job(payload: Dict[str, Any], dataframe: pd.DataFrame) -> AnalysisJobHandle:
    cfg = GHConfig.from_env()
    job_id = payload.get("job_id") or __import__("uuid").uuid4().hex
    # Prepare batching plan
    mc = payload.get("monte_carlo_config") or {}
    num_sims = int(mc.get("num_simulations", 0) or 0)
    # Estimate total steps based on flags
    flags = payload.get("scenario_flags", {}) or {}
    accounts = int(getattr(dataframe, "shape", (0,))[0] or 0)
    deterministic_scenarios = 0
    if flags.get("base", True):
        deterministic_scenarios += 1
    parallel_ids = [
        "parallel_100",
        "parallel_200",
        "parallel_300",
        "parallel_400",
        "parallel_minus_100",
        "parallel_minus_200",
        "parallel_minus_300",
        "parallel_minus_400",
    ]
    for sid in parallel_ids:
        if flags.get(sid, False):
            deterministic_scenarios += 1
    for sid in ("steepener", "flattener", "short_shock_up"):
        if flags.get(sid, False):
            deterministic_scenarios += 1

    det_steps = accounts * max(deterministic_scenarios, 0)
    mc_steps = accounts * max(num_sims, 0) if flags.get("monte_carlo", False) else 0

    batch_count = 1
    # First, limit by max_sims_per_job
    if num_sims > 0 and num_sims > cfg.max_sims_per_job:
        batch_count = int((num_sims + cfg.max_sims_per_job - 1) // cfg.max_sims_per_job)
    # Then, limit by max_steps_per_job if configured
    if cfg.max_steps_per_job and cfg.max_steps_per_job > 0 and mc_steps > 0:
        allowed_mc_steps = max(cfg.max_steps_per_job - det_steps, accounts)  # at least one simulation chunk
        if allowed_mc_steps < mc_steps:
            per_batch_sims = max(1, allowed_mc_steps // max(accounts, 1))
            batch_count = max(batch_count, int((num_sims + per_batch_sims - 1) // per_batch_sims))
    # Always create at least one batch
    batches: List[Dict[str, Any]] = []
    remaining = num_sims
    for i in range(batch_count):
        sims_for_batch = cfg.max_sims_per_job if i < batch_count - 1 else (remaining or num_sims)
        if remaining:
            sims_for_batch = min(remaining, cfg.max_sims_per_job)
            remaining -= sims_for_batch
        batch_payload = dict(payload)
        batch_payload["job_id"] = job_id
        # Adjust MC config for batch
        if num_sims > 0:
            mc_copy = dict(mc)
            mc_copy["num_simulations"] = int(sims_for_batch)
            # Stagger random seed to avoid overlap if provided
            base_seed = mc.get("random_seed")
            if isinstance(base_seed, int):
                mc_copy["random_seed"] = int(base_seed) + i * 1000003
            batch_payload["monte_carlo_config"] = mc_copy
        batches.append({"index": i, "payload": batch_payload})

    # Upload index and batch inputs to results branch
    index = {
        "job_id": job_id,
        "batches": [
            {"index": b["index"], "dir": _batch_dir(cfg, job_id, b["index"]) } for b in batches
        ],
        "created_at": time.time(),
    }
    _ensure_results_branch(cfg)
    _write_index(cfg, job_id, index)

    parquet_bytes = _serialize_df_to_parquet_bytes(dataframe)
    for b in batches:
        idx = b["index"]
        batch_dir = _batch_dir(cfg, job_id, idx)
        _gh_put_contents(
            cfg,
            f"{batch_dir}/payload.json",
            json.dumps(b["payload"], indent=2).encode("utf-8"),
            message=f"Add payload for job {job_id} batch {idx+1}",
        )
        _gh_put_contents(
            cfg,
            f"{batch_dir}/accounts.parquet",
            parquet_bytes,
            message=f"Add data for job {job_id} batch {idx+1}",
        )
        # Initial status
        init_status = JobStatus(id=f"{job_id}-b{idx+1}").to_dict()
        init_status.setdefault("message", "Pending dispatch")
        _gh_put_contents(
            cfg,
            f"{batch_dir}/status.json",
            json.dumps(init_status, indent=2).encode("utf-8"),
            message=f"Init status for job {job_id} batch {idx+1}",
        )

    # Dispatch workflow for each batch
    for b in batches:
        idx = b["index"]
        _gh_dispatch_workflow(
            cfg,
            inputs={
                "job_id": job_id,
                "batch_dir": _batch_dir(cfg, job_id, idx),
                "results_branch": cfg.results_branch,
            },
        )

    # Return a handle; job_dir here is a placeholder (not used by remote)
    job_local_dir = Path.cwd() / "output" / "jobs" / job_id
    return AnalysisJobHandle(job_id=job_id, job_dir=job_local_dir)


def read_job_status(handle: AnalysisJobHandle) -> JobStatus:
    cfg = GHConfig.from_env()
    index = _read_index(cfg, handle.job_id)
    if not index:
        cached = STATUS_CACHE.get(handle.job_id)
        if cached is not None:
            return cached
        return JobStatus(id=handle.job_id, state="pending", message="Waiting for job index...")
    batches = index.get("batches", [])
    total = 0
    step = 0
    any_failed = False
    all_completed = True
    message_parts: List[str] = []
    extras: Dict[str, Any] = {"batches": []}
    for b in batches:
        dir_path = b.get("dir")
        status_dict = _read_status_bytes(cfg, f"{dir_path}/status.json") or {}
        if not status_dict:
            status_dict = {"id": handle.job_id, "state": "pending", "message": "Awaiting status update"}
        else:
            status_dict.setdefault("id", handle.job_id)
        status_obj = JobStatus.from_dict(status_dict)
        b_state = status_obj.state
        b_step = status_obj.step
        b_total = status_obj.total
        total += max(b_total, 1)
        step += max(min(b_step, b_total), 0)
        if b_state == "failed":
            any_failed = True
        if b_state not in {"completed", "failed"}:
            all_completed = False
        msg = status_obj.message or b_state
        message_parts.append(f"batch {b.get('index', 0)+1}: {msg}")
        run_id = ((status_obj.extras or {}).get("run_id"))
        extras["batches"].append({"dir": dir_path, "state": b_state, "run_id": run_id})

    if any_failed:
        state = "failed"
    elif all_completed:
        state = "completed"
    else:
        state = "running"
    message = "; ".join(message_parts) if message_parts else state
    st = JobStatus(id=handle.job_id, state=state, step=step, total=max(total, 1), message=message, extras=extras)
    STATUS_CACHE[handle.job_id] = st
    return st


def _download_artifact_zip(cfg: GHConfig, run_id: str) -> List[bytes]:
    url = f"{GH_API}/repos/{cfg.owner}/{cfg.repo}/actions/runs/{run_id}/artifacts"
    r = requests.get(url, headers=_gh_headers(cfg))
    if r.status_code != 200:
        raise RuntimeError(f"Failed to list artifacts: {r.status_code} {r.text}")
    data = r.json()
    zips: List[bytes] = []
    for art in data.get("artifacts", []):
        art_id = art.get("id")
        if not art_id:
            continue
        dl = requests.get(f"{GH_API}/repos/{cfg.owner}/{cfg.repo}/actions/artifacts/{art_id}/zip", headers=_gh_headers(cfg))
        if dl.status_code == 200:
            zips.append(dl.content)
    return zips


def _extract_file_from_zips(zips: List[bytes], filename: str) -> List[bytes]:
    items: List[bytes] = []
    for blob in zips:
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            for name in zf.namelist():
                base = name.rsplit("/", 1)[-1]
                if base == filename:
                    items.append(zf.read(name))
    return items


def load_job_results(handle: AnalysisJobHandle) -> EngineResults:
    cfg = GHConfig.from_env()
    index = _read_index(cfg, handle.job_id) or {}
    batches = index.get("batches", [])
    # Collect run_ids
    run_ids: List[str] = []
    for b in batches:
        status_dict = _read_status_bytes(cfg, f"{b.get('dir')}/status.json") or {}
        if not status_dict:
            continue
        status_obj = JobStatus.from_dict({**status_dict, "id": handle.job_id})
        extras_dict = status_obj.extras or {}
        run_id = extras_dict.get("run_id")
        if run_id:
            run_ids.append(str(run_id))
    # Download artifacts for each run
    results: List[EngineResults] = []
    for rid in run_ids:
        zips = _download_artifact_zip(cfg, rid)
        for res_bytes in _extract_file_from_zips(zips, "results.pkl"):
            try:
                results.append(pd.read_pickle(io.BytesIO(res_bytes)))
            except Exception:
                pass
    if not results:
        raise RuntimeError("No results artifacts found for job")
    if len(results) == 1:
        return results[0]
    return merge_engine_results(results)


def load_job_bundle(handle: AnalysisJobHandle) -> Optional[Dict[str, Any]]:
    # We aggregate client-side; build bundle in app if needed.
    return None


def cleanup_job_artifacts(handle: AnalysisJobHandle) -> None:
    # No-op for remote driver; history retained in results branch
    return None


def cancel_job(handle: AnalysisJobHandle) -> None:
    cfg = GHConfig.from_env()
    index = _read_index(cfg, handle.job_id) or {}
    batches = index.get("batches", [])
    for b in batches:
        dir_path = b.get("dir")
        if not dir_path:
            continue
        status_dict = _read_status_bytes(cfg, f"{dir_path}/status.json") or {}
        job_status = JobStatus.from_dict({**status_dict, "id": handle.job_id})
        extras = job_status.extras or {}
        run_id = extras.get("run_id")
        if run_id:
            try:
                requests.post(
                    f"{GH_API}/repos/{cfg.owner}/{cfg.repo}/actions/runs/{run_id}/cancel",
                    headers=_gh_headers(cfg),
                )
            except Exception:
                pass
            # Remove run_id to avoid confusing status updates later
            extras.pop("run_id", None)
        job_status.state = "cancelled"
        job_status.message = "Cancellation requested by user."
        job_status.extras = extras
        payload = job_status.to_dict()
        payload["updated_at"] = _utc_now_iso()
        current_sha = _gh_get_file_sha(cfg, f"{dir_path}/status.json")
        _gh_put_contents(
            cfg,
            f"{dir_path}/status.json",
            json.dumps(payload, indent=2).encode("utf-8"),
            message=f"Cancel job {handle.job_id} ({dir_path})",
            sha=current_sha,
        )
        STATUS_CACHE[handle.job_id] = JobStatus.from_dict(payload)

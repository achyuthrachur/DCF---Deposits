from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Dict, Any

import requests
import streamlit as st

try:  # pragma: no cover - optional auto refresh
    from streamlit import st_autorefresh  # type: ignore
except Exception:  # pragma: no cover
    st_autorefresh = None  # type: ignore


GH_API = "https://api.github.com"


def _gh_cfg() -> dict:
    secrets = getattr(st, "secrets", {})
    gh = secrets.get("github", {}) if hasattr(secrets, "get") else {}
    repo = os.environ.get("GH_REPO") or gh.get("repo")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or gh.get("token")
    workflow = (
        os.environ.get("GH_DESKTOP_WORKFLOW")
        or gh.get("desktop_workflow")
        or gh.get("workflow")
        or os.environ.get("GH_WORKFLOW")
        or "release-desktop.yml"
    )
    ref = os.environ.get("GH_WORKFLOW_REF") or gh.get("workflow_ref", "main")
    if not (repo and token):
        raise RuntimeError("GitHub repo/token not configured.")
    owner, name = repo.split("/", 1)
    return {"owner": owner, "repo": name, "token": token, "workflow": workflow, "ref": ref}


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def dispatch_desktop_build() -> str:
    cfg = _gh_cfg()
    now = datetime.now(timezone.utc)
    version = f"v{now.strftime('%Y%m%d_%H%M%S')}"
    url = f"{GH_API}/repos/{cfg['owner']}/{cfg['repo']}/actions/workflows/{cfg['workflow']}/dispatches"
    payload = {"ref": cfg["ref"], "inputs": {"version": version}}
    r = requests.post(url, headers=_headers(cfg["token"]), json=payload, timeout=10)
    use_run_tag = False
    if r.status_code == 422 and "Unexpected inputs" in r.text:
        # Workflow expects no inputs; dispatch without them and let the
        # workflow compute a tag from run_number.
        r = requests.post(url, headers=_headers(cfg["token"]), json={"ref": cfg["ref"]}, timeout=10)
        version = None
        use_run_tag = True
    if r.status_code not in (201, 204):
        raise RuntimeError(f"workflow_dispatch failed: {r.status_code} {r.text}")
    st.session_state["desktop_build"] = {
        "version": version,
        "requested_at": now.isoformat(),
        "run_id": None,
        "run_info": None,
        "use_run_tag": use_run_tag,
    }
    return version or "auto-tag"


def _find_recent_run(build: dict) -> Optional[dict]:
    cfg = _gh_cfg()
    url = (
        f"{GH_API}/repos/{cfg['owner']}/{cfg['repo']}/actions/workflows/"
        f"{cfg['workflow']}/runs?event=workflow_dispatch&per_page=5"
    )
    r = requests.get(url, headers=_headers(cfg["token"]), timeout=10)
    if r.status_code != 200:
        return None
    runs = r.json().get("workflow_runs", [])
    if not runs:
        return None
    target_time = datetime.fromisoformat(build["requested_at"]) - timedelta(seconds=5)
    if target_time.tzinfo is None:
        target_time = target_time.replace(tzinfo=timezone.utc)
    for run in runs:
        created = datetime.strptime(run.get("created_at"), "%Y-%m-%dT%H:%M:%SZ")
        created = created.replace(tzinfo=timezone.utc)
        if created >= target_time:
            return run
    return runs[0]


def _poll_run_status(run_id: int) -> Tuple[str, Optional[str]]:
    cfg = _gh_cfg()
    r = requests.get(
        f"{GH_API}/repos/{cfg['owner']}/{cfg['repo']}/actions/runs/{run_id}",
        headers=_headers(cfg["token"]),
        timeout=10,
    )
    if r.status_code != 200:
        return "unknown", None
    data = r.json()
    return data.get("status", "unknown"), data.get("conclusion")


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None


def _latest_release_asset() -> Optional[Dict[str, Any]]:
    cfg = _gh_cfg()
    r = requests.get(
        f"{GH_API}/repos/{cfg['owner']}/{cfg['repo']}/releases/latest",
        headers=_headers(cfg["token"]),
        timeout=10,
    )
    if r.status_code != 200:
        return None
    release = r.json()
    published = _parse_iso(release.get("published_at") or release.get("created_at"))
    for asset in release.get("assets", []):
        nm = (asset.get("name") or "").lower()
        if nm.endswith(".exe") and ("dcf" in nm or "deposits" in nm):
            return {
                "name": asset.get("name") or "Windows Installer",
                "url": asset.get("browser_download_url"),
                "tag": release.get("tag_name"),
                "published_at": published,
            }
    return None


def _release_asset_by_tag(tag: str) -> Optional[Dict[str, Any]]:
    cfg = _gh_cfg()
    r = requests.get(
        f"{GH_API}/repos/{cfg['owner']}/{cfg['repo']}/releases/tags/{tag}",
        headers=_headers(cfg["token"]),
        timeout=10,
    )
    if r.status_code != 200:
        return None
    release = r.json()
    published = _parse_iso(release.get("published_at") or release.get("created_at"))
    for asset in release.get("assets", []):
        nm = (asset.get("name") or "").lower()
        if nm.endswith(".exe") and ("dcf" in nm or "deposits" in nm):
            return {
                "name": asset.get("name") or "Windows Installer",
                "url": asset.get("browser_download_url"),
                "tag": tag,
                "published_at": published,
            }
    return None


def render_desktop_build_expander() -> None:
    with st.expander("Build / Download Desktop App (Windows)", expanded=False):
        latest: Optional[Dict[str, Any]] = None
        try:
            latest = _latest_release_asset()
        except Exception:
            latest = None
        if latest:
            name, url = latest["name"], latest["url"]
            try:
                st.link_button(f"Download {name}", url)
            except Exception:
                st.markdown(f"[Download {name}]({url})")

        if st.button("Build latest installer", type="primary"):
            try:
                version = dispatch_desktop_build()
                st.success(f"Build dispatched for {version}.")
            except Exception as exc:
                st.error(f"Dispatch failed: {exc}")

        build = st.session_state.get("desktop_build")
        if build:
            run_id = build.get("run_id")
            if not run_id:
                run = _find_recent_run(build)
                if run:
                    build["run_id"] = run.get("id")
                    build["run_info"] = run
                    run_id = build["run_id"]
            status, conclusion = ("unknown", None)
            asset: Optional[Dict[str, Any]] = None
            tag: Optional[str] = build.get("version")
            if run_id:
                status, conclusion = _poll_run_status(run_id)
            if tag:
                asset = _release_asset_by_tag(tag)
            if status != "completed":
                if asset is not None:
                    status, conclusion = "completed", "success"
                else:
                    # Fall back to latest release when newer than dispatch time.
                    candidate = latest
                    requested_at = build.get("requested_at")
                    if candidate and requested_at:
                        cand_time = candidate.get("published_at")
                        req_time = _parse_iso(requested_at)
                        if cand_time and req_time and cand_time >= req_time:
                            asset = candidate
                            status, conclusion = "completed", "success"
            label = f"Build status: {status}" + (f" / {conclusion}" if conclusion else "")
            st.write(label)
            pct = {"queued": 5, "in_progress": 60, "completed": 100}.get(status, 10)
            st.progress(pct)
            if status == "completed":
                if conclusion == "success":
                    tag = build.get("version")
                    if not tag:
                        run_info = build.get("run_info") or {}
                        tag = f"v{run_info.get('run_number')}"
                    # reuse asset we may have already fetched while polling
                    if not asset:
                        asset = _release_asset_by_tag(tag) if tag else None
                    if not asset:
                        asset = _latest_release_asset()
                    if asset:
                        nm, url = asset["name"], asset["url"]
                        st.success("Installer ready.")
                        try:
                            st.link_button(f"Download {nm}", url)
                        except Exception:
                            st.markdown(f"[Download {nm}]({url})")
                else:
                    st.error("Build failed. Check GitHub Actions logs.")
                st.session_state.pop("desktop_build", None)
            else:
                if st_autorefresh is not None:
                    st_autorefresh(interval=5000, key="desktop_build_refresh")

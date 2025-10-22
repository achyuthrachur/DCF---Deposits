"""GitHub Actions worker entrypoint to execute a single analysis batch.

This runner reads payload and data from a job directory checked out from the
results branch, writes status updates back to that branch, and saves results
artefacts for upload by the workflow.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .job_manager import JobStatusWriter
from .status_writers import RepoJobStatusWriter
from .job_manager import _execute_analysis  # type: ignore


def _load_json(path: Path) -> Dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def run_batch(jobs_root: Path, batch_dir: str, job_id: str) -> None:
    repo_root = jobs_root
    batch_path = (jobs_root / batch_dir).resolve()
    status_path = batch_path / "status.json"
    payload_path = batch_path / "payload.json"
    data_path = batch_path / "accounts.parquet"

    batch_path.mkdir(parents=True, exist_ok=True)

    # Status writer that commits to the results branch repository.
    status_writer: JobStatusWriter = RepoJobStatusWriter(status_path, job_id, repo_root=repo_root)
    status_writer.initialize()
    gh_run_id = os.environ.get("GITHUB_RUN_ID")
    status_writer.write(state="running", message="Batch initialised.", extras={"run_id": gh_run_id})

    # Execute analysis
    dataframe = pd.read_parquet(data_path)
    payload = _load_json(payload_path)
    try:
        results, bundle_info, analysis_metadata = _execute_analysis(
            payload, dataframe, status_writer, batch_path
        )
        result_path = batch_path / "results.pkl"
        with result_path.open("wb") as fh:
            pickle.dump(results, fh)

        bundle_zip_path: Optional[Path] = None
        if bundle_info is not None and isinstance(bundle_info, dict) and bundle_info.get("zip_bytes"):
            bundle_zip_path = batch_path / "bundle.zip"
            bundle_zip_path.write_bytes(bundle_info["zip_bytes"])  # type: ignore[index]

        extras: Dict[str, Any] = {"analysis_metadata": analysis_metadata}
        status_writer.write(
            state="completed",
            message="Analysis complete.",
            result_path=result_path.name,
            bundle_path=bundle_zip_path.name if bundle_zip_path else None,
            extras=extras,
        )
    except Exception as exc:
        import traceback

        status_writer.write(state="failed", message="Analysis failed.", error=traceback.format_exc())
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single ALM analysis batch.")
    parser.add_argument("--jobs-root", required=True, help="Path to results branch checkout (repo root)")
    parser.add_argument("--batch-dir", required=True, help="Relative job directory for this batch")
    parser.add_argument("--job-id", required=True, help="Main job id")
    args = parser.parse_args()

    run_batch(Path(args.jobs_root), args.batch_dir, args.job_id)


if __name__ == "__main__":
    main()


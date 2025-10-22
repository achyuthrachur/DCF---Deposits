"""Background job launcher and worker for long-running ALM analyses."""

from __future__ import annotations

import json
import os
import pickle
import signal
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.engine import ALMEngine, DiscountConfig, MonteCarloConfig
from src.reporting import InMemoryReportBuilder
from src.models.results import EngineResults

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(os.environ.get("APP_OUTPUT_ROOT", PROJECT_ROOT / "output"))
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
JOB_ROOT = OUTPUT_ROOT / "jobs"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        # File may be in the middle of being written by another process; treat as empty.
        return {}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    temp_path.replace(path)


@dataclass
class JobStatus:
    """Serializable snapshot of a background job."""

    id: str
    state: str = "pending"
    message: str = ""
    step: int = 0
    total: int = 1
    started_at: Optional[str] = None
    updated_at: Optional[str] = None
    pid: Optional[int] = None
    error: Optional[str] = None
    result_path: Optional[str] = None
    bundle_path: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "id": self.id,
            "state": self.state,
            "message": self.message,
            "step": self.step,
            "total": self.total,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "pid": self.pid,
            "error": self.error,
            "result_path": self.result_path,
            "bundle_path": self.bundle_path,
            "extras": self.extras,
        }
        return {key: value for key, value in payload.items() if value is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobStatus":
        return cls(
            id=str(data.get("id", "")),
            state=str(data.get("state", "pending")),
            message=str(data.get("message", "")),
            step=int(data.get("step", 0)),
            total=int(data.get("total", 1)),
            started_at=data.get("started_at"),
            updated_at=data.get("updated_at"),
            pid=data.get("pid"),
            error=data.get("error"),
            result_path=data.get("result_path"),
            bundle_path=data.get("bundle_path"),
            extras=data.get("extras", {}),
        )


class JobStatusWriter:
    """Helper to persist status updates safely to disk."""

    def __init__(self, status_path: Path, job_id: str) -> None:
        self.status_path = status_path
        self.job_id = job_id

    def _load(self) -> JobStatus:
        data = _read_json(self.status_path)
        if not data:
            return JobStatus(id=self.job_id)
        return JobStatus.from_dict(data)

    def write(self, **updates: Any) -> JobStatus:
        status = self._load()
        for key, value in updates.items():
            setattr(status, key, value)
        now = _utc_now_iso()
        status.updated_at = now
        if status.started_at is None:
            status.started_at = now
        _write_json(self.status_path, status.to_dict())
        return status

    def initialize(self) -> JobStatus:
        return self.write(state="pending", message="Job created", step=0, total=1)

    def mark_dispatched(self, pid: int) -> JobStatus:
        return self.write(
            state="dispatched",
            message="Job dispatched to worker process.",
            pid=pid,
            step=0,
            total=1,
        )

    def mark_running(self, message: str = "Starting analysis...") -> JobStatus:
        return self.write(state="running", message=message)

    def update_progress(
        self,
        step: int,
        total: int,
        message: str,
        *,
        extras: Optional[Dict[str, Any]] = None,
    ) -> JobStatus:
        return self.write(
            state="running",
            message=message,
            step=step,
            total=total,
            extras=extras or {},
        )

    def mark_complete(
        self,
        *,
        message: str = "Analysis complete.",
        result_path: Optional[str] = None,
        bundle_path: Optional[str] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> JobStatus:
        return self.write(
            state="completed",
            message=message,
            result_path=result_path,
            bundle_path=bundle_path,
            extras=extras or {},
        )

    def mark_failed(self, message: str, error: str) -> JobStatus:
        return self.write(state="failed", message=message, error=error)


@dataclass
class AnalysisJobHandle:
    job_id: str
    job_dir: Path

    @property
    def status_path(self) -> Path:
        return self.job_dir / "status.json"

    @property
    def payload_path(self) -> Path:
        return self.job_dir / "payload.json"

    @property
    def data_path(self) -> Path:
        return self.job_dir / "accounts.parquet"

    @property
    def result_path(self) -> Path:
        return self.job_dir / "results.pkl"

    @property
    def bundle_meta_path(self) -> Path:
        return self.job_dir / "bundle.json"

    @property
    def bundle_zip_path(self) -> Path:
        return self.job_dir / "bundle.zip"


def launch_analysis_job(payload: Dict[str, Any], dataframe: pd.DataFrame) -> AnalysisJobHandle:
    """Persist analysis inputs and launch a background worker."""
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    job_dir = JOB_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=False)

    handle = AnalysisJobHandle(job_id=job_id, job_dir=job_dir)

    # Persist inputs
    dataframe.to_parquet(handle.data_path)
    payload = dict(payload)
    payload["job_id"] = job_id
    payload["data_path"] = handle.data_path.name
    _write_json(handle.payload_path, payload)

    status_writer = JobStatusWriter(handle.status_path, job_id)
    status_writer.initialize()

    ctx = get_context("spawn")
    process = ctx.Process(target=_job_worker_entry, args=(str(job_dir),), daemon=True)
    process.start()
    status_writer.mark_dispatched(process.pid or 0)
    return handle


def read_job_status(handle: AnalysisJobHandle) -> JobStatus:
    return JobStatus.from_dict(_read_json(handle.status_path))


def load_job_results(handle: AnalysisJobHandle) -> EngineResults:
    with handle.result_path.open("rb") as fh:
        return pickle.load(fh)


def load_job_bundle(handle: AnalysisJobHandle) -> Optional[Dict[str, Any]]:
    if not handle.bundle_meta_path.exists() or not handle.bundle_zip_path.exists():
        return None
    metadata = _read_json(handle.bundle_meta_path)
    zip_bytes = handle.bundle_zip_path.read_bytes()
    metadata["zip_bytes"] = zip_bytes
    return metadata


def cleanup_job_artifacts(handle: AnalysisJobHandle) -> None:
    """Remove job directory to reclaim disk space."""
    if not handle.job_dir.exists():
        return
    for path in sorted(handle.job_dir.glob("**/*"), reverse=True):
        if path.is_file():
            try:
                path.unlink()
            except OSError:
                pass
        else:
            try:
                path.rmdir()
            except OSError:
                pass
    try:
        handle.job_dir.rmdir()
    except OSError:
        pass


def _job_worker_entry(job_dir_str: str) -> None:
    job_dir = Path(job_dir_str)
    payload = _read_json(job_dir / "payload.json")
    job_id = payload.get("job_id", job_dir.name)
    status_writer = JobStatusWriter(job_dir / "status.json", job_id)
    status_writer.mark_running("Worker process initialised.")
    try:
        dataframe = pd.read_parquet(job_dir / payload["data_path"])
        results, bundle_info, analysis_metadata = _execute_analysis(payload, dataframe, status_writer, job_dir)
        result_path = job_dir / "results.pkl"
        with result_path.open("wb") as fh:
            pickle.dump(results, fh)

        extras: Dict[str, Any] = {"analysis_metadata": analysis_metadata}
        bundle_path: Optional[str] = None
        if bundle_info is not None:
            bundle_zip_path = job_dir / "bundle.zip"
            bundle_meta_path = job_dir / "bundle.json"
            bundle_zip_path.write_bytes(bundle_info["zip_bytes"])
            bundle_meta = dict(bundle_info)
            bundle_meta.pop("zip_bytes", None)
            _write_json(bundle_meta_path, bundle_meta)
            bundle_path = bundle_zip_path.name
            extras["bundle"] = bundle_meta

        status_writer.mark_complete(
            message="Analysis complete.",
            result_path=result_path.name,
            bundle_path=bundle_path,
            extras=extras,
        )
    except Exception as exc:  # pragma: no cover - defensive
        status_writer.mark_failed("Analysis failed.", traceback.format_exc())
        raise


def cancel_job(handle: AnalysisJobHandle) -> None:
    """Attempt to stop the local worker process for the given job."""

    status = read_job_status(handle)
    pid = status.pid
    if pid:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except Exception:
            pass
    status_writer = JobStatusWriter(handle.status_path, handle.job_id)
    status_writer.write(state="cancelled", message="Cancellation requested by user.")


def _execute_analysis(
    payload: Dict[str, Any],
    dataframe: pd.DataFrame,
    status_writer: JobStatusWriter,
    job_dir: Path,
) -> Tuple[Any, Optional[Dict[str, Any]], Dict[str, Any]]:
    """Recreate engine configuration from payload and execute analysis."""
    engine = ALMEngine()
    field_map = payload["field_map"]
    optional_fields = payload.get("optional_fields", [])
    engine.load_dataframe(
        dataframe=dataframe,
        field_map=field_map,
        optional_fields=optional_fields,
    )
    engine.set_segmentation(payload.get("segmentation", "all"))

    assumptions: Dict[str, Dict[str, Any]] = payload.get("assumptions", {})
    for segment_key, values in assumptions.items():
        engine.set_assumptions(
            segment_key=segment_key,
            decay_rate=values.get("decay_rate"),
            wal_years=values.get("wal_years"),
            deposit_beta_up=values.get("deposit_beta_up"),
            deposit_beta_down=values.get("deposit_beta_down"),
            repricing_beta_up=values.get("repricing_beta_up"),
            repricing_beta_down=values.get("repricing_beta_down"),
            decay_priority=values.get("decay_priority", "auto"),
        )

    discount_config = payload.get("discount_config", {})
    mode = discount_config.get("mode")
    if mode == "single":
        engine.set_discount_single_rate(float(discount_config["rate"]))
    elif mode == "manual":
        tenor_rates = {
            int(key): float(value)
            for key, value in discount_config.get("tenor_rates", {}).items()
        }
        engine.set_discount_yield_curve(
            tenor_rates,
            interpolation_method=discount_config.get("interpolation", "linear"),
            source="manual",
        )
    elif mode == "fred":
        curve_snapshot = discount_config.get("curve_snapshot")
        if curve_snapshot:
            tenors = [int(t) for t in curve_snapshot.get("tenors", ())]
            rates = [float(r) for r in curve_snapshot.get("rates", ())]
            interpolation = curve_snapshot.get("interpolation", "linear")
            engine.set_discount_yield_curve(dict(zip(tenors, rates)), interpolation_method=interpolation, source="fred")
        else:
            engine.set_discount_curve_from_fred(
                discount_config.get("api_key", ""),
                interpolation_method=discount_config.get("interpolation", "linear"),
                target_date=discount_config.get("target_date"),
            )

    base_market_path = [float(value) for value in payload.get("base_market_path", [])]
    if base_market_path:
        engine.set_base_market_rate_path(base_market_path)

    projection_settings = payload.get("projection_settings", {})

    monte_carlo_metadata = payload.get("monte_carlo_config")
    if monte_carlo_metadata:
        engine.set_monte_carlo_config(MonteCarloConfig.from_metadata(monte_carlo_metadata))

    scenario_flags = payload.get("scenario_flags", {})
    engine.configure_standard_scenarios(scenario_flags)

    def progress_callback(step: int, total: int, message: str) -> None:
        status_writer.update_progress(step, total, message)

    results = engine.run_analysis(
        projection_months=int(projection_settings.get("projection_months", engine.projection_months)),
        materiality_threshold=float(projection_settings.get("materiality_threshold", engine.materiality_threshold)),
        max_projection_months=int(projection_settings.get("max_projection_months", engine.max_projection_months)),
        progress_callback=progress_callback,
    )

    analysis_metadata: Dict[str, Any] = {
        "run_timestamp": datetime.utcnow().isoformat(),
        "segmentation_method": engine.segmentation_method,
        "projection_settings": projection_settings,
        "scenario_flags": scenario_flags,
        "monte_carlo_config": monte_carlo_metadata,
        "assumptions": assumptions,
        "field_map": field_map,
        "optional_fields": optional_fields,
        "discount_selection": discount_config,
        "base_market_path": base_market_path,
    }
    try:
        analysis_metadata["portfolio"] = {
            "account_count": len(engine.accounts),
            "total_balance": float(sum(acc.balance for acc in engine.accounts)),
        }
    except Exception:
        pass

    selected_curve = discount_config.get("curve_snapshot")
    if selected_curve:
        analysis_metadata["selected_curve"] = selected_curve

    discount_config_obj: Optional[DiscountConfig]
    try:
        discount_config_obj = engine.discount_configuration()
    except Exception:
        discount_config_obj = None

    bundle_info: Optional[Dict[str, Any]] = None
    package_options = payload.get("package_options", {}) or {}
    if package_options.get("enabled", False):
        builder = InMemoryReportBuilder(base_title=payload.get("report_title", "Deposit Analysis"))
        bundle = builder.build(
            results,
            discount_config=discount_config_obj,
            analysis_metadata=analysis_metadata,
            export_cashflows=package_options.get("export_cashflows", False),
            cashflow_mode=package_options.get("cashflow_mode", "sample"),
            cashflow_sample_size=int(package_options.get("cashflow_sample_size", 20)),
            cashflow_random_state=42,
        )
        bundle_info = {
            "zip_bytes": bundle.zip_bytes,
            "zip_name": bundle.zip_name,
            "excel_name": bundle.excel_name,
            "word_name": bundle.word_name,
            "manifest": bundle.manifest,
            "created_at": bundle.created_at.isoformat(),
            "token": bundle.created_at.isoformat(),
        }

    extras = {
        "analysis_metadata": analysis_metadata,
    }
    status_writer.update_progress(
        results.summary_frame().shape[0] or 1,
        results.summary_frame().shape[0] or 1,
        "Finalising analysis outputs...",
        extras=extras,
    )

    return results, bundle_info, analysis_metadata

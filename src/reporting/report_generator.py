"""Report generation utilities."""

from __future__ import annotations

import io
import json
import logging
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..models.results import EngineResults
from ..visualization import (
    create_monte_carlo_dashboard,
    create_rate_path_animation,
    extract_monte_carlo_data,
    extract_shock_data,
    plot_percentile_ladder,
    plot_portfolio_pv_distribution,
    plot_rate_confidence_fan,
    plot_rate_path_spaghetti,
    plot_shock_group_summary,
    plot_shock_magnitude,
    plot_shock_pv_delta,
    plot_shock_rate_paths,
    plot_shock_tenor_comparison,
)
from ..visualization.monte_carlo_plots import save_figure

if TYPE_CHECKING:
    from ..engine import DiscountConfig

LOGGER = logging.getLogger(__name__)


def _json_default(obj: object) -> object:
    """JSON serializer that handles numpy/pandas/path objects gracefully."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class ReportGenerator:
    """Persist ALM engine outputs to disk (tables, charts, bundles)."""

    def __init__(
        self,
        output_dir: str | Path = "output",
        *,
        timestamped: bool = False,
        run_label: Optional[str] = None,
    ) -> None:
        base_dir = Path(output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        if timestamped:
            run_label = run_label or datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
            self.output_dir = base_dir / run_label
        else:
            self.output_dir = base_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir = base_dir
        self.run_label = self.output_dir.name

    # ------------------------------------------------------------------ helpers
    def _write_json(self, payload: Dict[str, object], filename: str) -> Path:
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=_json_default)
        return path

    def _export_tables(
        self, tables: Optional[Dict[str, pd.DataFrame]], directory: Path, prefix: str = ""
    ) -> Dict[str, Path]:
        output: Dict[str, Path] = {}
        if not tables:
            return output
        directory.mkdir(parents=True, exist_ok=True)
        for key, table in tables.items():
            if table is None or not isinstance(table, pd.DataFrame) or table.empty:
                continue
            filename = f"{prefix}{key}.csv" if prefix else f"{key}.csv"
            path = directory / filename
            table.to_csv(path, index=False)
            output[key] = path
        return output

    @staticmethod
    def _build_shock_figures(
        results: EngineResults,
        scenario_id: str,
    ) -> Dict[str, Figure]:
        """Prepare deterministic shock figures in memory."""
        data = extract_shock_data(results, scenario_id)
        if not data:
            return {}

        figures: Dict[str, Figure] = {}
        label = data["scenario_label"]
        curve_df = data["curve_comparison"]
        tenor_df = data.get("tenor_comparison")

        try:
            figures["rate_paths"] = plot_shock_rate_paths(curve_df, label)
            figures["shock_magnitude"] = plot_shock_magnitude(curve_df, label)

            tenor_fig = plot_shock_tenor_comparison(tenor_df, label)
            if tenor_fig is not None:
                figures["tenor_comparison"] = tenor_fig

            pv_fig = plot_shock_pv_delta(
                data.get("base_pv"),
                data.get("scenario_pv"),
                label,
                data.get("delta"),
                data.get("delta_pct"),
            )
            if pv_fig is not None:
                figures["pv_delta"] = pv_fig
        except Exception as exc:  # pragma: no cover - visualisations are best effort
            LOGGER.warning("Failed to prepare shock visuals for %s: %s", scenario_id, exc)
            for fig in figures.values():
                plt.close(fig)
            return {}

        return figures

    def _export_shock_visuals(
        self,
        results: EngineResults,
        scenario_id: str,
        scenario_dir: Path,
    ) -> Dict[str, Path]:
        """Generate deterministic shock figures."""
        figures = self._build_shock_figures(results, scenario_id)
        if not figures:
            return {}

        output: Dict[str, Path] = {}
        try:
            if "rate_paths" in figures:
                output["rate_paths"] = save_figure(
                    figures["rate_paths"], scenario_dir / "shock_rate_paths.png"
                )
            if "shock_magnitude" in figures:
                output["shock_magnitude"] = save_figure(
                    figures["shock_magnitude"], scenario_dir / "shock_magnitude.png"
                )
            if "tenor_comparison" in figures:
                output["tenor_comparison"] = save_figure(
                    figures["tenor_comparison"], scenario_dir / "shock_tenor_comparison.png"
                )
            if "pv_delta" in figures:
                output["pv_delta"] = save_figure(
                    figures["pv_delta"], scenario_dir / "shock_pv_delta.png"
                )
        finally:
            for fig in figures.values():
                plt.close(fig)

        return output

    def _export_monte_carlo_animation(
        self,
        results: EngineResults,
        scenario_id: str,
        output_dir: Path,
        *,
        animation_format: str = "html",
    ) -> Dict[str, Path]:
        data = extract_monte_carlo_data(results, scenario_id=scenario_id)
        if not data:
            return {}

        rate_summary = data.get("rate_summary")
        if rate_summary is None or rate_summary.empty:
            return {}

        figure = create_rate_path_animation(
            data.get("rate_sample"),
            rate_summary,
        )
        output: Dict[str, Path] = {}
        output_dir.mkdir(parents=True, exist_ok=True)
        html_path = output_dir / "rate_path_animation.html"
        try:
            figure.write_html(html_path, include_plotlyjs="cdn")
            output["animation_html"] = html_path
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Unable to save Monte Carlo animation HTML: %s", exc)
            return output

        if animation_format.lower() not in {"gif", "mp4"}:
            return output

        try:
            import imageio.v2 as imageio  # type: ignore

            frames = []
            original_data = figure.data
            for frame in figure.frames:
                figure.update(data=frame.data)
                png_bytes = figure.to_image(format="png", width=1280, height=720)
                frames.append(imageio.imread(png_bytes))
            figure.update(data=original_data)
            if not frames:
                return output
            fps = 12
            if animation_format.lower() == "gif":
                gif_path = output_dir / "rate_path_animation.gif"
                imageio.mimsave(gif_path, frames, fps=fps)
                output["animation_gif"] = gif_path
            else:
                try:
                    video_path = output_dir / "rate_path_animation.mp4"
                    writer = imageio.get_writer(video_path, fps=fps, format="FFMPEG")
                    for frame in frames:
                        writer.append_data(frame)
                    writer.close()
                    output["animation_mp4"] = video_path
                except Exception as exc:  # pragma: no cover
                    LOGGER.warning("Unable to encode animation as MP4: %s", exc)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Unable to export Monte Carlo animation frames: %s", exc)

        return output

    def _create_archive(self, files: Iterable[Path], filename: str = "analysis_bundle.zip") -> Path:
        archive_path = self.output_dir / filename
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in files:
                file_path = Path(file_path)
                if not file_path.exists() or file_path == archive_path:
                    continue
                arcname = file_path.relative_to(self.output_dir)
                zf.write(file_path, arcname=str(arcname))
        return archive_path

    # ------------------------------------------------------------------- exports
    def export_summary(self, results: EngineResults, filename: str = "eve_summary.csv") -> Path:
        path = self.output_dir / filename
        results.summary_frame().to_csv(path, index=False)
        return path

    def export_parameter_summary(
        self, results: EngineResults, filename: str = "parameter_summary.json"
    ) -> Path:
        payload = results.parameter_summary or {}
        return self._write_json(payload, filename)

    @staticmethod
    def sample_cashflows(
        cashflows: pd.DataFrame,
        sample_size: int = 20,
        random_state: Optional[int] = None,
    ) -> pd.DataFrame:
        if sample_size <= 0:
            return cashflows
        unique_accounts = cashflows["account_id"].nunique()
        if unique_accounts <= sample_size:
            return cashflows
        starting_balances = (
            cashflows.sort_values("month")
            .groupby("account_id", as_index=False)
            .first()[["account_id", "beginning_balance"]]
        )
        candidate_count = min(unique_accounts, max(sample_size * 3, sample_size))
        top_candidates = starting_balances.nlargest(candidate_count, "beginning_balance")
        sampled_ids = top_candidates["account_id"].sample(
            n=min(sample_size, len(top_candidates)),
            replace=False,
            random_state=random_state,
        )
        return cashflows[cashflows["account_id"].isin(sampled_ids.tolist())]

    def export_validation_summary(
        self,
        results: EngineResults,
        filename: str = "validation_summary.json",
    ) -> Optional[Path]:
        if not results.validation_summary:
            return None
        return self._write_json(results.validation_summary, filename)

    def export_discount_configuration(
        self,
        discount_config: "DiscountConfig",
        filename: str = "discount_curve.json",
    ) -> Path:
        payload = {
            "method": discount_config.method,
            "source": discount_config.source,
            "metadata": discount_config.metadata,
            "curve": discount_config.curve.to_dict(),
        }
        return self._write_json(payload, filename)

    def export_monte_carlo_tables(
        self,
        results: EngineResults,
        scenario_id: str = "monte_carlo",
        prefix: str = "monte_carlo",
    ) -> Dict[str, Path]:
        scenario = results.scenario_results.get(scenario_id)
        if scenario is None:
            return {}
        tables = scenario.extra_tables or {}
        directory = self.output_dir / "monte_carlo" / "tables"
        directory.mkdir(parents=True, exist_ok=True)
        output: Dict[str, Path] = self._export_tables(tables, directory, prefix=f"{prefix}_")
        metadata = scenario.metadata.get("config") if scenario.metadata else None
        if metadata:
            config_path = self._write_json(metadata, f"{prefix}_config.json")
            output["config"] = config_path
        return output

    @staticmethod
    def _build_monte_carlo_figures(
        results: EngineResults,
        scenario_id: str = "monte_carlo",
        prefix: str = "monte_carlo",
    ) -> Dict[str, Figure]:
        data = extract_monte_carlo_data(results, scenario_id=scenario_id)
        if not data:
            return {}

        figures: Dict[str, Figure] = {}
        try:
            figures[f"{prefix}_rate_spaghetti"] = plot_rate_path_spaghetti(
                data["rate_sample"], data["rate_summary"]
            )
            figures[f"{prefix}_rate_fan"] = plot_rate_confidence_fan(data["rate_summary"])
            figures[f"{prefix}_pv_distribution"] = plot_portfolio_pv_distribution(
                data["pv_distribution"],
                book_value=data.get("book_value"),
                base_case_pv=data.get("base_case_pv"),
                percentiles=data.get("percentiles"),
            )
            figures[f"{prefix}_percentiles"] = plot_percentile_ladder(
                data.get("percentiles", {}),
                book_value=data.get("book_value"),
                base_case_pv=data.get("base_case_pv"),
            )
            figures[f"{prefix}_dashboard"] = create_monte_carlo_dashboard(data)
        except Exception as exc:  # pragma: no cover - visual generation best effort
            LOGGER.warning("Failed to prepare Monte Carlo visuals: %s", exc)
            for fig in figures.values():
                plt.close(fig)
            return {}

        return figures

    def export_monte_carlo_visuals(
        self,
        results: EngineResults,
        scenario_id: str = "monte_carlo",
        prefix: str = "monte_carlo",
    ) -> Dict[str, Path]:
        figures = self._build_monte_carlo_figures(results, scenario_id=scenario_id, prefix=prefix)
        if not figures:
            return {}

        output_paths: Dict[str, Path] = {}
        directory = self.output_dir / "monte_carlo" / "figures"
        directory.mkdir(parents=True, exist_ok=True)
        try:
            for key, figure in figures.items():
                cleaned_key = key.replace(f"{prefix}_", "")
                output_paths[cleaned_key] = save_figure(
                    figure, directory / f"{prefix}_{cleaned_key}.png"
                )
        finally:
            for fig in figures.values():
                plt.close(fig)
        return output_paths

    # ----------------------------------------------------------- comprehensive
    def export_analysis_bundle(
        self,
        results: EngineResults,
        *,
        discount_config: Optional["DiscountConfig"] = None,
        analysis_metadata: Optional[Dict[str, object]] = None,
        export_cashflows: bool = False,
        cashflow_mode: str = "sample",
        cashflow_sample_size: int = 20,
        cashflow_random_state: Optional[int] = 42,
        include_animation: bool = False,
        animation_format: str = "html",
    ) -> Dict[str, object]:
        """Export a complete analysis bundle and return artifact metadata."""

        exported: Dict[str, Path] = {}

        exported["summary"] = self.export_summary(results)
        exported["parameter_summary"] = self.export_parameter_summary(results)

        if analysis_metadata:
            exported["analysis_metadata"] = self.export_metadata(
                analysis_metadata, filename="analysis_metadata.json"
            )

        validation_path = self.export_validation_summary(results)
        if validation_path:
            exported["validation_summary"] = validation_path

        if discount_config is not None:
            exported["discount_configuration"] = self.export_discount_configuration(
                discount_config
            )

        scenario_dirs: Dict[str, Path] = {}
        shock_scenarios: list[str] = []
        for scenario_id, scenario in results.scenario_results.items():
            scenario_dir = self.output_dir / "scenarios" / scenario_id
            scenario_dir.mkdir(parents=True, exist_ok=True)
            scenario_dirs[scenario_id] = scenario_dir

            account_pv_path = scenario_dir / "account_pv.csv"
            scenario.account_level_pv.to_csv(account_pv_path, index=False)
            exported[f"{scenario_id}_account_pv"] = account_pv_path

            metadata_path = scenario_dir / "metadata.json"
            scenario_metadata = scenario.metadata or {}
            with metadata_path.open("w", encoding="utf-8") as fh:
                json.dump(scenario_metadata, fh, indent=2, default=_json_default)
            exported[f"{scenario_id}_metadata"] = metadata_path

            tables_dir = scenario_dir / "tables"
            table_paths = self._export_tables(
                scenario.extra_tables,
                tables_dir,
            )
            for key, path in table_paths.items():
                exported[f"{scenario_id}_table_{key}"] = path

            if export_cashflows:
                cash_df = (
                    scenario.cashflows
                    if cashflow_mode == "full"
                    else self.sample_cashflows(
                        scenario.cashflows,
                        sample_size=cashflow_sample_size,
                        random_state=cashflow_random_state,
                    )
                )
                cash_path = scenario_dir / "cashflows.csv"
                cash_df.to_csv(cash_path, index=False)
                exported[f"{scenario_id}_cashflows"] = cash_path

            method = (scenario.metadata or {}).get("method")
            if method not in {None, "base", "monte_carlo"}:
                shock_paths = self._export_shock_visuals(results, scenario_id, scenario_dir)
                if shock_paths:
                    shock_scenarios.append(scenario_id)
                    for key, path in shock_paths.items():
                        exported[f"{scenario_id}_visual_{key}"] = path

        if shock_scenarios:
            group_fig = plot_shock_group_summary(
                results,
                scenario_ids=shock_scenarios,
                title="Deterministic Scenario Comparison",
            )
            if group_fig is not None:
                path = save_figure(
                    group_fig, self.output_dir / "scenarios" / "shock_group_summary.png"
                )
                exported["shock_group_summary"] = path

        monte_carlo_tables = self.export_monte_carlo_tables(results)
        exported.update({f"monte_carlo_table_{k}": p for k, p in monte_carlo_tables.items()})

        monte_carlo_visuals = self.export_monte_carlo_visuals(results)
        exported.update({f"monte_carlo_visual_{k}": p for k, p in monte_carlo_visuals.items()})

        if include_animation:
            animation_dir = self.output_dir / "monte_carlo" / "animation"
            animation_paths = self._export_monte_carlo_animation(
                results,
                scenario_id="monte_carlo",
                output_dir=animation_dir,
                animation_format=animation_format,
            )
            exported.update({f"monte_carlo_animation_{k}": p for k, p in animation_paths.items()})

        all_files = {Path(p) for p in exported.values() if p is not None}
        archive_path = self._create_archive(all_files)

        return {
            "output_dir": self.output_dir,
            "files": sorted(all_files),
            "archive": archive_path,
            "artifacts": exported,
        }

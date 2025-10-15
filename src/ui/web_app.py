"""Streamlit web interface for the NMD ALM engine."""

from __future__ import annotations

import sys
import os
from datetime import date
from base64 import b64encode
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import streamlit as st

st.cache_data.clear()
st.cache_resource.clear()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.validator import ValidationError
from src.engine import ALMEngine
from src.core.fred_loader import FREDYieldCurveLoader
from src.core.yield_curve import YieldCurve
from src.core.monte_carlo import MonteCarloConfig, MonteCarloLevel, VasicekParams
from src.reporting import ReportGenerator
from src.visualization import (
    create_monte_carlo_dashboard,
    create_rate_path_animation,
    extract_monte_carlo_data,
    plot_percentile_ladder,
    plot_portfolio_pv_distribution,
    plot_rate_confidence_fan,
    plot_rate_path_spaghetti,
)

# Crowe logo SVG fallback for hosting environments where the asset file isn't available.
CROWE_LOGO_SVG = r"""<svg id="Pinnacle_Crowe_2017" data-name="Pinnacle Crowe 2017" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 79.72 22.5"><defs><style>.cls-1{fill:#f5a81c;}.cls-2{fill:#fff;}</style></defs><title>Crowe_Logo_2c_w</title><path class="cls-1" d="M26.32,14.83a.19.19,0,0,0-.15-.11c-.06,0-.12,0-.16.11L12.78,37.09c0,.08-.06.12,0,.13l.1-.1L25.9,21.32c.06-.07.09-.1.1-.09s0,0-.05.12L16.89,37.09c0,.08-.06.12-.05.13s0,0,.1-.1L27.65,24.18c.06-.07.09-.1.1-.09s0,0,0,.12l-7,12.88c0,.08-.06.12-.05.12s.05,0,.1-.09L29.26,26.8a.25.25,0,0,1,.4,0L38,37.12c.06.07.09.1.1.09s0-.05,0-.12L26.32,14.83" transform="translate(-12.73 -14.72)"/><path class="cls-2" d="M53.51,21.7a6.33,6.33,0,0,0-3.6-1.2c-3.4,0-5.4,2.44-5.4,5.66,0,3.48,2.4,5.82,5.4,5.82a6.14,6.14,0,0,0,3.48-1l.75,1.14a8.24,8.24,0,0,1-4.33,1.25c-4.91,0-7.4-3.36-7.4-7.15,0-3.46,2.71-7.07,7.64-7.07a6.74,6.74,0,0,1,4.24,1.39l-.78,1.18" transform="translate(-12.73 -14.72)"/><path class="cls-2" d="M56.81,24.51l.7-.74A2.45,2.45,0,0,1,59,23.06a2.42,2.42,0,0,1,1.63,1l-.81,1.14a3,3,0,0,0-1.31-.41c-.91,0-1.65.9-1.65,3.08v5.29H55.08V23.29h1.73v1.22" transform="translate(-12.73 -14.72)"/><path class="cls-2" d="M60,28.09a4.85,4.85,0,1,1,9.69,0c0,3-2,5.26-4.86,5.26S60,31.13,60,28.09m1.91,0c0,1.84.62,4,2.92,4s3-2.17,3-4-.73-3.77-3-3.77-2.92,2-2.92,3.77" transform="translate(-12.73 -14.72)"/><path class="cls-2" d="M77.48,23.29c1,2.34,1.92,4.7,2.86,7.51h0c.77-2.89,1.57-5.29,2.36-7.7l1.63.27-3.46,9.75H79.68c-1-2.44-2-4.87-3-7.66h0c-.94,2.79-1.91,5.22-2.87,7.66H72.66l-3.4-9.59L71,23.1c.81,2.41,1.63,4.81,2.42,7.7h0c.94-2.81,1.9-5.17,2.87-7.51h1.16" transform="translate(-12.73 -14.72)"/><path class="cls-2" d="M85.8,28.32a3.57,3.57,0,0,0,3.32,3.78,4.94,4.94,0,0,0,2.7-.85l.51,1a5.88,5.88,0,0,1-3.26,1.14C86,33.35,84,31.47,84,28a4.61,4.61,0,0,1,4.65-5c3,0,4,2.69,3.8,5.26H85.8m5-1.18c0-1.53-.65-2.82-2.21-2.82a2.72,2.72,0,0,0-2.69,2.82Z" transform="translate(-12.73 -14.72)"/></svg>"""

REQUIRED_FIELDS = {
    "account_id": "Unique account identifier",
    "balance": "Current balance",
    "interest_rate": "Current interest rate (decimal)",
}
OPTIONAL_FIELDS = {
    "account_type": "Account type (for segmentation)",
    "customer_segment": "Customer segment (for segmentation)",
    "rate_type": "Rate type metadata (optional)",
}

DEFAULT_ASSUMPTIONS = {
    "checking": {
        "decay_rate": 0.05,
        "wal_years": 5.0,
        "deposit_beta_up": 0.40,
        "deposit_beta_down": 0.25,
        "repricing_beta_up": 1.00,
        "repricing_beta_down": 1.00,
    },
    "savings": {
        "decay_rate": 0.08,
        "wal_years": 3.5,
        "deposit_beta_up": 0.55,
        "deposit_beta_down": 0.35,
        "repricing_beta_up": 1.00,
        "repricing_beta_down": 1.00,
    },
    "money market": {
        "decay_rate": 0.20,
        "wal_years": 1.5,
        "deposit_beta_up": 0.75,
        "deposit_beta_down": 0.60,
        "repricing_beta_up": 1.00,
        "repricing_beta_down": 1.00,
    },
}

SCENARIO_OPTIONS = [
    ("parallel_100", "+100 bps parallel shock", True),
    ("parallel_200", "+200 bps parallel shock", True),
    ("parallel_300", "+300 bps parallel shock", False),
    ("parallel_400", "+400 bps parallel shock", False),
    ("parallel_minus_100", "-100 bps parallel shock", True),
    ("parallel_minus_200", "-200 bps parallel shock", False),
    ("parallel_minus_300", "-300 bps parallel shock", False),
    ("parallel_minus_400", "-400 bps parallel shock", False),
    ("steepener", "Curve steepener (short down, long up)", False),
    ("flattener", "Curve flattener (short up, long down)", False),
    ("short_shock_up", "Short-rate shock up (+200 bps front-end)", False),
]

TENOR_LABELS: List[Tuple[int, str]] = [
    (3, "3 Months"),
    (6, "6 Months"),
    (12, "1 Year"),
    (24, "2 Years"),
    (36, "3 Years"),
    (60, "5 Years"),
    (84, "7 Years"),
    (120, "10 Years"),
]

CASHFLOW_SAMPLE_SIZE = 20


def _reset_state_on_upload() -> None:
    """Clear downstream state when a new file is uploaded."""
    for key in [
        "field_map",
        "optional_fields",
        "mapping_confirmed",
        "mapped_df",
        "assumptions",
        "run_results",
    ]:
        st.session_state.pop(key, None)


def _infer_defaults(columns: Iterable[str]) -> Dict[str, str]:
    """Suggest mappings based on simple column name matching."""
    mapping: Dict[str, str] = {}
    lower = {col.lower(): col for col in columns}
    for canonical in list(REQUIRED_FIELDS) + list(OPTIONAL_FIELDS):
        canonical_no_underscore = canonical.replace("_", "")
        for candidate, original in lower.items():
            if candidate == canonical or candidate.replace("_", "") == canonical_no_underscore:
                mapping[canonical] = original
                break
    return mapping


def _default_for_segment(segment: str) -> Dict[str, float]:
    """Return suggested assumptions for a segment."""
    key = segment.lower()
    for pattern, defaults in DEFAULT_ASSUMPTIONS.items():
        if pattern in key:
            return defaults
    return DEFAULT_ASSUMPTIONS["checking"]


def _decimalize(value: float) -> float:
    """Convert percentage-based inputs to decimals."""
    if value is None:
        return 0.0
    if value > 1.5:
        return value / 100.0
    return value


@st.cache_data(show_spinner=False)
def _brand_logo_html() -> Optional[str]:
    """Return the Crowe SVG markup wrapped for top-right display.

    Prefers reading the repo asset; falls back to embedded SVG constant.
    """
    search_paths = [
        PROJECT_ROOT / "assets" / "crowe_logo.svg",
        REPO_ROOT / "assets" / "crowe_logo.svg",
    ]
    for path in search_paths:
        try:
            if path.exists():
                svg = path.read_text(encoding="utf-8")
                return f"<div class='brand-badge'>{svg}</div>"
        except Exception:
            continue
    if CROWE_LOGO_SVG:
        return f"<div class='brand-badge'>{CROWE_LOGO_SVG}</div>"
    return None


def _derive_segments(
    dataframe: pd.DataFrame, segmentation: str
) -> Tuple[List[str], pd.DataFrame]:
    """Determine segment keys and produce a summary table."""
    if segmentation == "all":
        total_accounts = int(len(dataframe))
        total_balance = float(dataframe["balance"].sum())
        summary = pd.DataFrame(
            [{
                "segment": "ALL",
                "accounts": total_accounts,
                "balance": total_balance,
            }]
        )
        return ["ALL"], summary

    if segmentation == "by_account_type":
        column = "account_type"
    elif segmentation == "by_customer_segment":
        column = "customer_segment"
    elif segmentation == "cross":
        column = None
    else:
        raise ValueError(f"Unsupported segmentation method: {segmentation}")

    df = dataframe.copy()
    if segmentation == "cross":
        missing_mask = df["account_type"].isna() | df["customer_segment"].isna()
        if missing_mask.any():
            st.warning(
                "Rows with missing account type or customer segment are excluded "
                "from cross segmentation."
            )
        df = df.loc[~missing_mask].copy()
        df["segment_key"] = (
            df["account_type"].astype(str) + "::" + df["customer_segment"].astype(str)
        )
        summary = (
            df.groupby("segment_key")
            .agg(accounts=("account_id", "count"), balance=("balance", "sum"))
            .reset_index()
            .rename(columns={"segment_key": "segment"})
        )
        segments = summary["segment"].tolist()
        return segments, summary

    missing_mask = df[column].isna()
    if missing_mask.any():
        st.warning(
            f"Rows with missing values in '{column}' are excluded from segmentation calculations."
        )
    df = df.loc[~missing_mask].copy()
    df[column] = df[column].astype(str)
    summary = (
        df.groupby(column)
        .agg(accounts=("account_id", "count"), balance=("balance", "sum"))
        .reset_index()
        .rename(columns={column: "segment"})
    )
    segments = summary["segment"].tolist()
    return segments, summary


def _validate_mapping(unique_columns: List[str], mapping: Dict[str, str]) -> Optional[str]:
    """Ensure field mapping selections are valid."""
    chosen = [value for value in mapping.values() if value]
    duplicates = {col for col in chosen if chosen.count(col) > 1}
    if duplicates:
        return (
            "Each column can only be mapped once. Duplicate selections: "
            + ", ".join(sorted(duplicates))
        )
    required_missing = [field for field in REQUIRED_FIELDS if not mapping.get(field)]
    if required_missing:
        return "Missing required field mapping(s): " + ", ".join(required_missing)
    return None


def _prepare_assumption_inputs(segments: List[str]) -> Dict[str, Dict[str, float]]:
    """Collect assumption inputs for each segment from the UI."""
    st.markdown("#### Assumption Entry")
    assumption_values: Dict[str, Dict[str, float]] = {}
    for segment in segments:
        defaults = _default_for_segment(segment)
        decay = st.number_input(
            f"{segment} ' Decay / Runoff Rate (annual decimal)",
            min_value=0.0,
            max_value=1.0,
            value=defaults["decay_rate"],
            step=0.01,
            key=f"decay_{segment}",
        )
        wal = st.number_input(
            f"{segment} ' Weighted Average Life (years)",
            min_value=0.1,
            max_value=15.0,
            value=defaults["wal_years"],
            step=0.1,
            key=f"wal_{segment}",
        )
        col1, col2 = st.columns(2)
        with col1:
            deposit_beta_up = st.number_input(
                f"{segment} ' Deposit Beta (rising rates)",
                min_value=0.0,
                max_value=1.5,
                value=defaults["deposit_beta_up"],
                step=0.05,
                key=f"deposit_beta_up_{segment}",
            )
            repricing_beta_up = st.number_input(
                f"{segment} ' Repricing Beta (rising rates)",
                min_value=0.0,
                max_value=2.0,
                value=defaults["repricing_beta_up"],
                step=0.05,
                key=f"repricing_beta_up_{segment}",
            )
        with col2:
            deposit_beta_down = st.number_input(
                f"{segment} ' Deposit Beta (falling rates)",
                min_value=0.0,
                max_value=1.5,
                value=defaults["deposit_beta_down"],
                step=0.05,
                key=f"deposit_beta_down_{segment}",
            )
            repricing_beta_down = st.number_input(
                f"{segment} ' Repricing Beta (falling rates)",
                min_value=0.0,
                max_value=2.0,
                value=defaults["repricing_beta_down"],
                step=0.05,
                key=f"repricing_beta_down_{segment}",
            )
        assumption_values[segment] = {
            "decay_rate": decay,
            "wal_years": wal,
            "deposit_beta_up": deposit_beta_up,
            "deposit_beta_down": deposit_beta_down,
            "repricing_beta_up": repricing_beta_up,
            "repricing_beta_down": repricing_beta_down,
        }
    return assumption_values



def _collect_scenarios(
    projection_months: int,
    base_curve: Optional[YieldCurve],
) -> Tuple[Dict[str, bool], Optional[MonteCarloConfig]]:
    """Capture scenario selections from the user."""
    st.markdown("#### Scenario Selection")
    scenario_flags = {"base": True}
    st.checkbox("Base case (no shock)", value=True, disabled=True, key="scenario_base")
    cols = st.columns(2)
    for idx, (scenario_id, label, default) in enumerate(SCENARIO_OPTIONS):
        with cols[idx % 2]:
            scenario_flags[scenario_id] = st.checkbox(
                label, value=default, key=f"scenario_{scenario_id}"
            )

    short_anchor = float(base_curve.get_rate(3)) if base_curve else 0.03
    long_anchor = float(base_curve.get_rate(120)) if base_curve else short_anchor
    default_interp = st.session_state.get("discount_interpolation_method", "linear")

    monte_carlo_config: Optional[MonteCarloConfig] = None
    with st.expander("Monte Carlo simulation", expanded=False):
        enable_mc = st.checkbox("Run Monte Carlo simulation", value=False, key="scenario_monte_carlo")
        st.caption(
            "Level 1 keeps today's discount curve fixed and only simulates the 3M rate used for deposit repricing. "
            "Level 2 evolves both the 3M and 10Y anchors so the entire curve can steepen or flatten over time."
        )
        if enable_mc:
            scenario_flags["monte_carlo"] = True
            st.caption(
                f"Current curve anchors: 3M = {short_anchor * 100:.2f}%, 10Y = {long_anchor * 100:.2f}%."
            )

            st.caption("Level 1 keeps today's discount curve fixed; Level 2 simulates both the 3M and 10Y anchors so the curve itself can evolve.")
            level_choice = st.radio(
                "Implementation level",
                options=[
                    "Level 1 - Static yield curve",
                    "Level 2 - Two-factor correlated curve",
                ],
                index=1,
                key="mc_level",
            )
            level = (
                MonteCarloLevel.TWO_FACTOR
                if level_choice.startswith("Level 2")
                else MonteCarloLevel.STATIC_CURVE
            )

            num_sim = st.number_input(
                "Number of simulations",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                key="mc_simulations",
            )
            seed_input = st.text_input("Random seed (optional)", value="", key="mc_seed")
            random_seed = int(seed_input.strip()) if seed_input.strip().isdigit() else None

            short_col = st.columns(3)
            with short_col[0]:
                short_a = st.number_input(
                    "Short mean reversion (a)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.15,
                    step=0.01,
                    key="mc_short_a",
                )
            with short_col[1]:
                short_b = st.number_input(
                    "Short anchor (b)",
                    min_value=0.0,
                    max_value=0.20,
                    value=float(short_anchor),
                    step=0.001,
                    format="%.3f",
                    key="mc_short_b",
                )
            with short_col[2]:
                short_sigma = st.number_input(
                    "Short volatility (sigma)",
                    min_value=0.0,
                    max_value=0.10,
                    value=0.01,
                    step=0.001,
                    format="%.3f",
                    key="mc_short_sigma",
                )
            short_params = VasicekParams(
                mean_reversion=float(short_a),
                long_term_mean=float(short_b),
                volatility=float(short_sigma),
            )

            long_params = None
            correlation = 0.70
            if level == MonteCarloLevel.TWO_FACTOR:
                long_col = st.columns(3)
                with long_col[0]:
                    long_a = st.number_input(
                        "Long mean reversion (a)",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.07,
                        step=0.01,
                        key="mc_long_a",
                    )
                with long_col[1]:
                    long_b = st.number_input(
                        "Long anchor (b)",
                        min_value=0.0,
                        max_value=0.20,
                        value=float(long_anchor),
                        step=0.001,
                        format="%.3f",
                        key="mc_long_b",
                    )
                with long_col[2]:
                    long_sigma = st.number_input(
                        "Long volatility (sigma)",
                        min_value=0.0,
                        max_value=0.10,
                        value=0.008,
                        step=0.001,
                        format="%.3f",
                        key="mc_long_sigma",
                    )
                long_params = VasicekParams(
                    mean_reversion=float(long_a),
                    long_term_mean=float(long_b),
                    volatility=float(long_sigma),
                )
                correlation = st.slider(
                    "Correlation between 3M and 10Y shocks",
                    min_value=-0.95,
                    max_value=0.95,
                    value=0.70,
                    step=0.05,
                    key="mc_correlation",
                )

            save_paths = st.checkbox("Save rate path samples", value=True, key="mc_save_paths")
            sample_size = 0
            if save_paths:
                sample_size = int(
                    st.number_input(
                        "Rate path sample size",
                        min_value=10,
                        max_value=500,
                        value=100,
                        step=10,
                        key="mc_sample_size",
                    )
                )

            generate_reports = st.checkbox(
                "Generate detailed Monte Carlo reports", value=True, key="mc_reports"
            )
            interp_options = ["linear", "log-linear", "cubic"]
            interp_index = interp_options.index(default_interp) if default_interp in interp_options else 0
            interpolation_method = st.selectbox(
                "Yield curve interpolation (for discounting)",
                options=interp_options,
                index=interp_index,
                key="mc_interp",
            )

            monte_carlo_config = MonteCarloConfig(
                level=level,
                num_simulations=int(num_sim),
                projection_months=projection_months,
                random_seed=random_seed,
                short_rate=short_params,
                long_rate=long_params,
                correlation=float(correlation),
                save_rate_paths=save_paths,
                generate_reports=generate_reports,
                sample_size=sample_size if save_paths else 0,
                interpolation_method=interpolation_method,
            )

    return scenario_flags, monte_carlo_config



def _download_button(label: str, dataframe: pd.DataFrame, filename: str) -> None:
    """Render a reusable download button for CSV exports."""
    csv_bytes = dataframe.to_csv(index=False).encode("utf-8")
    st.download_button(
        label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )


def _load_uploaded_file(uploaded_file: "st.runtime.uploaded_file_manager.UploadedFile") -> pd.DataFrame:
    """Read uploaded CSV or Excel file into a dataframe."""
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)


def main() -> None:
    """Streamlit application entry point."""
    st.set_page_config(
        page_title="ALM Validation - Deposit Accounts: DCF Calculation Engine",
        page_icon="",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #0f2d63 0%, #133f7f 45%, #0f2d63 100%);
            color: #ffffff;
        }
        .brand-badge {
            position: fixed;
            top: 18px;
            right: 28px;
            z-index: 1100;
            padding: 6px 10px;
        }
        .brand-badge img {
            max-width: 150px;
            height: auto;
            filter: drop-shadow(0 4px 10px rgba(0,0,0,0.35));
        }
        .hero-card {
            background: rgba(19, 63, 127, 0.70);
            border: 1px solid rgba(255, 194, 61, 0.5);
            box-shadow: 0 18px 45px rgba(3, 14, 40, 0.35);
            border-radius: 24px;
            padding: 42px 50px;
            margin-bottom: 28px;
        }
        .hero-card h1 {
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 16px;
            color: #ffffff;
        }
        .hero-card p {
            font-size: 1.05rem;
            line-height: 1.6;
            color: #e9f0ff;
            margin-bottom: 24px;
        }
        .accent-line {
            width: 140px;
            height: 5px;
            border-radius: 4px;
            background: #ffc94b;
            margin: 14px 0 28px 0;
        }
        .flow-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 18px;
        }
        .flow-step {
            background: rgba(12, 36, 82, 0.85);
            border: 1px solid rgba(255, 201, 75, 0.4);
            border-radius: 18px;
            padding: 18px 20px;
            color: #ffffff;
            font-size: 0.95rem;
            min-height: 140px;
        }
        .flow-step span {
            display: inline-flex;
            justify-content: center;
            align-items: center;
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: #ffc94b;
            color: #0f2d63;
            font-weight: 700;
            margin-bottom: 12px;
        }
        .section-header {
            color: #ffffff;
            margin-top: 12px;
            margin-bottom: 6px;
            font-weight: 600;
        }
        .info-band {
            background: rgba(9, 32, 72, 0.75);
            border: 1px solid rgba(255, 201, 75, 0.4);
            border-radius: 22px;
            padding: 30px 36px;
            margin-bottom: 32px;
        }
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 18px;
        }
        .info-card {
            background: rgba(12, 36, 82, 0.8);
            border-radius: 16px;
            padding: 18px 20px;
            border: 1px solid rgba(255, 201, 75, 0.4);
        }
        .info-card h3 {
            font-size: 1.1rem;
            font-weight: 600;
            color: #ffd66b;
            margin-bottom: 10px;
        }
        .info-card p {
            font-size: 0.95rem;
            line-height: 1.5;
            color: #f2f6ff;
            margin: 0;
        }
        .stApp label,
        .stApp label span,
        .stApp [data-testid="stMarkdown"] p,
        .stApp [data-testid="stMarkdown"] li,
        .stApp [data-testid="stMarkdown"] a,
        .stApp [data-testid="stCaption"],
        .stApp [data-testid="stNumberInputLabel"] p,
        .stApp [data-testid="stRadio"] label div p,
        .stApp [data-testid="stRadio"] label div span,
        .stApp [data-testid="stCheckbox"] label div p,
        .stApp [data-testid="stCheckbox"] label div span,
        .stApp [data-testid="stSelectbox"] div p,
        .stApp [data-testid="stExpander"] button p,
        .stApp [data-testid="stExpander"] button span {
            color: #f6f9ff !important;
        }
        .stApp .stNumberInput input,
        .stApp .stTextInput input,
        .stApp textarea {
            color: #0f2d63 !important;
        }
        .stApp div[data-baseweb="select"] span,
        .stApp .stMultiSelect div[data-baseweb="tag"] span,
        .stApp div[data-baseweb="select"] input {
            color: #0f2d63 !important;
        }
        .stApp .stFormSubmitButton > button {
            background: linear-gradient(135deg, #ffc94b, #f0a500);
            color: #0b1d3a !important;
            font-weight: 700 !important;
            border: none;
            border-radius: 999px;
            padding: 0.5rem 1.6rem;
        }
        .stApp .stButton > button,
        .stApp .stDownloadButton > button {
            background: linear-gradient(135deg, #ffc94b, #f0a500);
            color: #0b1d3a !important;
            font-weight: 700 !important;
            border: none;
            border-radius: 999px;
            padding: 0.6rem 1.8rem;
        }
        .stApp .stButton > button:hover,
        .stApp .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #ffd66b, #ffb400);
            color: #04122a !important;
        }
        .stDataFrame, .stTable {
            background: rgba(255, 255, 255, 0.95) !important;
            border-radius: 12px;
            padding: 10px;
        }
        .footer-credit {
            position: fixed;
            bottom: 24px;
            right: 28px;
            background: rgba(8, 30, 68, 0.9);
            color: #ffffff;
            padding: 10px 20px;
            border-radius: 999px;
            border: 1px solid #ffc94b;
            font-size: 0.85rem;
            z-index: 999;
            box-shadow: 0 8px 20px rgba(0,0,0,0.35);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    brand_html = _brand_logo_html()
    if brand_html:
        st.markdown(brand_html, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero-card">
            <h1>ALM Validation - Deposit Accounts: DCF Calculation Engine</h1>
            <div class="accent-line"></div>
            <p>
                This interactive engine empowers ALM teams to validate non-maturity deposit assumptions,
                project monthly cash flows, and quantify economic value of equity (EVE) impacts under
                deterministic shocks or Monte Carlo-generated rate paths. Upload raw portfolio extracts,
                tailor behavioral assumptions, and instantly compare scenario-driven valuations.
            </p>
            <div class="flow-grid">
                <div class="flow-step">
                    <span>1</span>
                    Upload portfolio data (CSV or Excel) and map fields to the required schema.
                </div>
                <div class="flow-step">
                    <span>2</span>
                    Choose segmentation (portfolio, account type, customer, or cross) and enter
                    decay, WAL, and beta assumptions for each segment.
                </div>
                <div class="flow-step">
                    <span>3</span>
                    Configure discounting (single rate or term structure) and base market rate path.
                </div>
                <div class="flow-step">
                    <span>4</span>
                    Run parallel shocks from 100 bps to 400 bps or launch Monte Carlo simulations with
                    user-defined volatility and drift.
                </div>
                <div class="flow-step">
                    <span>5</span>
                    Review deterministic and stochastic outputs, download detailed cash flow & PV exhibits,
                    and share results.
                </div>
            </div>
        </div>
        <div class="info-band">
            <div class="info-grid">
                <div class="info-card">
                    <h3>Behavioral Controls</h3>
                    <p>Segment balances by product or customer cohort, validate decay and WAL coherence, and ensure beta inputs pass reasonableness checks before simulations run.</p>
                </div>
                <div class="info-card">
                    <h3>Valuation Analytics</h3>
                    <p>Produce monthly principal and interest cash flows, calculate discounted PV and EVE impacts versus the base curve, and export ready-to-share CSV outputs.</p>
                </div>
                <div class="info-card">
                    <h3>Monte Carlo Insights</h3>
                    <p>Generate stochastic market paths, observe expected cash flow trajectories, and review PV distributions with percentile statistics and raw simulation detail.</p>
                </div>
            </div>
        </div>
        <div class="footer-credit">Created by Achyuth Rachur</div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload account-level data (CSV or Excel)", type=["csv", "xlsx", "xls"]
    )
    if uploaded_file is not None:
        if "uploaded_filename" not in st.session_state or (
            st.session_state.get("uploaded_filename") != uploaded_file.name
        ):
            st.session_state["uploaded_filename"] = uploaded_file.name
            st.session_state["uploaded_df"] = _load_uploaded_file(uploaded_file)
            _reset_state_on_upload()
    else:
        st.session_state.pop("uploaded_df", None)

    df_raw: Optional[pd.DataFrame] = st.session_state.get("uploaded_df")
    if df_raw is None:
        st.info("Awaiting CSV upload to begin.")
        return

    st.markdown("### Step 1 - Preview Data & Map Fields")
    st.dataframe(df_raw.head(10))
    columns = list(df_raw.columns)
    defaults = _infer_defaults(columns)

    with st.form("mapping_form"):
        st.write("Select which source columns correspond to the required fields.")
        mapping_inputs: Dict[str, str] = {}
        for field, description in REQUIRED_FIELDS.items():
            default_column = defaults.get(field)
            mapping_inputs[field] = st.selectbox(
                f"{field} ({description})",
                options=columns,
                index=columns.index(default_column)
                if default_column in columns
                else 0,
                key=f"map_{field}",
            )

        st.write("Optional mappings (leave blank to skip).")
        optional_inputs: Dict[str, Optional[str]] = {}
        for field, description in OPTIONAL_FIELDS.items():
            options = ["(Not mapped)"] + columns
            default_label = defaults.get(field, "(Not mapped)")
            optional_inputs[field] = st.selectbox(
                f"{field} ({description})",
                options=options,
                index=options.index(default_label) if default_label in options else 0,
                key=f"map_optional_{field}",
            )

        mapping_submitted = st.form_submit_button("Confirm Field Mapping")

    if mapping_submitted:
        field_map = dict(mapping_inputs)
        optional_fields = [
            field for field, value in optional_inputs.items() if value and value != "(Not mapped)"
        ]
        for field in optional_fields:
            field_map[field] = optional_inputs[field]  # type: ignore[index]

        error = _validate_mapping(columns, field_map)
        if error:
            st.error(error)
        else:
            loader = ALMEngine()
            try:
                load_result = loader.load_dataframe(df_raw, field_map, optional_fields=optional_fields)
            except ValidationError as exc:
                st.error(f"Validation error: {exc}")
            else:
                st.session_state["field_map"] = field_map
                st.session_state["optional_fields"] = optional_fields
                st.session_state["mapping_confirmed"] = True
                st.session_state["mapped_df"] = load_result.dataframe
                st.success("Field mapping saved. Proceed to configure assumptions below.")

    if not st.session_state.get("mapping_confirmed"):
        return

    mapped_df: pd.DataFrame = st.session_state["mapped_df"]

    st.markdown("### Step 2 - Configure Segmentation & Assumptions")
    segmentation_friendly = {
        "All accounts as one segment": "all",
        "Segment by account type": "by_account_type",
        "Segment by customer segment": "by_customer_segment",
        "Cross segmentation (account  customer)": "cross",
    }
    segmentation_choice = st.selectbox(
        "Segmentation method",
        options=list(segmentation_friendly.keys()),
        index=0,
        key="segmentation_choice",
    )
    segmentation = segmentation_friendly[segmentation_choice]

    optional_fields = st.session_state.get("optional_fields", [])
    if segmentation in {"by_account_type", "cross"} and "account_type" not in st.session_state["field_map"]:
        st.error("Segmentation by account type requires the Account Type field to be mapped.")
        return
    if segmentation in {"by_customer_segment", "cross"} and "customer_segment" not in st.session_state["field_map"]:
        st.error("Segmentation by customer segment requires the Customer Segment field to be mapped.")
        return

    segments, segment_summary = _derive_segments(mapped_df, segmentation)
    if not segments:
        st.error("No segments detected for the selected segmentation method.")
        return

    st.write("Detected segments:")
    st.dataframe(segment_summary)

    assumptions = _prepare_assumption_inputs(segments)

    st.markdown("### Step 3 - Projection Settings")
    projection_months = st.number_input(
        "Projection horizon (months)",
        min_value=12,
        max_value=360,
        step=12,
        value=120,
    )

    discount_method = st.radio(
        "Discount rate configuration",
        options=["Single rate", "Fetch from FRED", "Manual yield curve"],
        index=1,
        horizontal=True,
    )

    discount_config: Dict[str, object]
    selected_curve: Optional[YieldCurve] = st.session_state.get("selected_discount_curve")
    selected_interpolation = st.session_state.get("discount_interpolation_method", "linear")
    if discount_method == "Single rate":
        discount_rate_input = st.number_input(
            "Discount rate (annual decimal, e.g., 0.035 for 3.5%)",
            min_value=0.0,
            max_value=0.20,
            value=0.035,
            step=0.005,
        )
        discount_config = {"mode": "single", "rate": discount_rate_input, "interpolation": "linear"}
        flat_tenors = [3, 6, 12, 24, 36, 60, 84, 120]
        flat_rates = [float(discount_rate_input)] * len(flat_tenors)
        selected_curve = YieldCurve(flat_tenors, flat_rates, metadata={"source": "single_rate"})
        selected_interpolation = "linear"
    elif discount_method == "Fetch from FRED":
        fred_api_key_default = os.environ.get("FRED_API_KEY", "")
        fred_api_key = st.text_input(
            "FRED API key",
            value=fred_api_key_default,
            type="password",
            help="Store a FRED API key in the FRED_API_KEY environment variable for convenience.",
            key="fred_api_key_input",
        )
        use_specific_date = st.checkbox(
            "Use specific valuation date",
            value=bool(st.session_state.get("fred_target_date")),
            key="fred_use_specific_date",
        )
        target_date_iso: Optional[str] = st.session_state.get("fred_target_date")
        if use_specific_date:
            default_date = (
                date.fromisoformat(target_date_iso) if target_date_iso else date.today()
            )
            target_date_input = st.date_input(
                "Valuation date",
                value=default_date,
                key="fred_target_date_input",
            )
            target_date_iso = target_date_input.isoformat()
        else:
            target_date_iso = None
        st.session_state["fred_target_date"] = target_date_iso

        interpolation_method = st.selectbox(
            "Interpolation method",
            options=["linear", "log-linear", "cubic"],
            index=0,
            key="fred_interpolation",
        )
        selected_interpolation = interpolation_method
        if st.button("Fetch current curve", use_container_width=True):
            if not fred_api_key:
                st.error("FRED API key is required to fetch the Treasury curve.")
            else:
                with st.spinner("Fetching latest Treasury curve from FRED..."):
                    try:
                        loader = FREDYieldCurveLoader(fred_api_key)
                        curve_snapshot = loader.get_current_yield_curve(
                            interpolation_method=interpolation_method,
                            target_date=target_date_iso,
                        )
                        st.session_state["fred_curve_snapshot"] = curve_snapshot
                        st.session_state["selected_discount_curve"] = curve_snapshot
                        st.success(
                            f"Curve loaded (as of {curve_snapshot.metadata.get('as_of', 'latest')})."
                        )
                    except Exception as exc:  # pragma: no cover - network failure
                        st.error(f"Unable to fetch curve: {exc}")

        curve_snapshot: Optional[YieldCurve] = st.session_state.get("fred_curve_snapshot")
        if curve_snapshot:
            st.caption(
                "Latest fetched curve: "
                + ", ".join(
                    f"{int(tenor)}M {rate * 100:.2f}%"
                    for tenor, rate in zip(curve_snapshot.tenors, curve_snapshot.rates)
                )
            )
            selected_curve = curve_snapshot
        else:
            st.info("Click 'Fetch current curve' after entering your API key.")

        discount_config = {
            "mode": "fred",
            "api_key": fred_api_key,
            "interpolation": interpolation_method,
            "target_date": target_date_iso,
        }
    else:
        st.write("Enter annualised rates for each tenor (leave blank to omit).")
        tenor_values: Dict[int, float] = {}
        columns = st.columns(2)
        for idx, (tenor, label) in enumerate(TENOR_LABELS):
            with columns[idx % 2]:
                value = st.text_input(label, value="", key=f"tenor_{tenor}")
            if value.strip():
                try:
                    tenor_values[tenor] = _decimalize(float(value.strip()))
                except ValueError:
                    st.error(f"Invalid numeric value for {label}.")
                    return
        if not tenor_values:
            st.error("At least one tenor rate is required for manual yield curve configuration.")
            return
        interpolation_method = st.selectbox(
            "Interpolation method",
            options=["linear", "log-linear", "cubic"],
            index=0,
            key="manual_curve_interpolation",
        )
        discount_config = {
            "mode": "manual",
            "tenor_rates": tenor_values,
            "interpolation": interpolation_method,
        }
        sorted_tenors = sorted(tenor_values.keys())
        manual_rates = [tenor_values[t] for t in sorted_tenors]
        selected_curve = YieldCurve(sorted_tenors, manual_rates, interpolation_method=interpolation_method, metadata={"source": "manual"})
        selected_interpolation = interpolation_method

    st.session_state["selected_discount_curve"] = selected_curve
    st.session_state["discount_interpolation_method"] = selected_interpolation

    if selected_curve:
        base_market_path = [
            float(selected_curve.get_rate(month))
            for month in range(1, int(projection_months) + 1)
        ]
        st.caption(
            f"Base market rate path derived from the selected curve (month 1: {base_market_path[0] * 100:.2f}%)."
        )
    else:
        base_market_path = [0.03] * int(projection_months)
        st.caption(
            "No yield curve selected yet  using a flat 3% base market path until a curve is configured."
        )
    st.session_state["base_market_path"] = base_market_path

    scenario_flags, monte_carlo_config = _collect_scenarios(
        int(projection_months),
        selected_curve,
    )

    run_clicked = st.button("Run Analysis", type="primary")

    results = st.session_state.get("run_results")
    progress_bar = None
    progress_text = None

    if run_clicked:
        engine = ALMEngine()
        try:
            engine.load_dataframe(
                dataframe=df_raw,
                field_map=st.session_state["field_map"],
                optional_fields=st.session_state.get("optional_fields", []),
            )
            engine.set_segmentation(segmentation)
            for segment_key, values in assumptions.items():
                engine.set_assumptions(
                    segment_key=segment_key,
                    decay_rate=_decimalize(values["decay_rate"]),
                    wal_years=values["wal_years"],
                    deposit_beta_up=_decimalize(values["deposit_beta_up"]),
                    deposit_beta_down=_decimalize(values["deposit_beta_down"]),
                    repricing_beta_up=_decimalize(values["repricing_beta_up"]),
                    repricing_beta_down=_decimalize(values["repricing_beta_down"]),
                )
            mode = discount_config.get("mode")
            if mode == "single":
                engine.set_discount_single_rate(_decimalize(discount_config["rate"]))
            elif mode == "manual":
                engine.set_discount_yield_curve(
                    {int(k): _decimalize(v) for k, v in discount_config["tenor_rates"].items()},
                    interpolation_method=discount_config.get("interpolation", "linear"),
                    source="manual",
                )
            elif mode == "fred":
                fred_api_key = discount_config.get("api_key") or os.environ.get("FRED_API_KEY", "")
                interpolation = discount_config.get("interpolation", "linear")
                target_date = discount_config.get("target_date")
                if fred_api_key:
                    engine.set_discount_curve_from_fred(
                        fred_api_key,
                        interpolation_method=interpolation,
                        target_date=target_date,
                    )
                else:
                    curve_snapshot = st.session_state.get("fred_curve_snapshot")
                    if not curve_snapshot:
                        st.error("Fetch a FRED curve or provide an API key before running analysis.")
                        return
                    engine.set_discount_curve(curve_snapshot, source="fred")
            base_market_path = st.session_state.get(
                "base_market_path",
                [0.03] * int(projection_months),
            )
            engine.set_base_market_rate_path(base_market_path)
            if monte_carlo_config:
                engine.set_monte_carlo_config(monte_carlo_config)
            engine.configure_standard_scenarios(scenario_flags)
            progress_text = st.empty()
            progress_bar = st.progress(0)

            def progress_callback(current: int, total: int, message: str) -> None:
                pct = int((current / max(total, 1)) * 100)
                progress_text.markdown(f"**{message}**")
                progress_bar.progress(min(100, pct))

            with st.spinner("Executing cash flow projections..."):
                results = engine.run_analysis(
                    projection_months=int(projection_months),
                    progress_callback=progress_callback,
                )
        except ValidationError as exc:
            st.error(f"Validation error during execution: {exc}")
            return
        except Exception as exc:  # pragma: no cover - guard rail
            st.error(f"Unexpected error: {exc}")
            return

        st.session_state["run_results"] = results
        st.success("Analysis complete! Scroll down to review and download outputs.")

        if progress_text and progress_bar:
            progress_text.markdown("**Preparing summary metrics...**")
            progress_bar.progress(80)
    else:
        if results is None:
            st.info("Configure inputs and click **Run Analysis** to generate results.")
            return
        else:
            st.caption(
                "Displaying previously generated results. Run the analysis again to refresh."
            )

    summary_df = results.summary_frame()

    st.markdown("### Scenario Present Value Summary")
    st.dataframe(summary_df)
    _download_button("Download summary CSV", summary_df, "eve_summary.csv")

    scenario_ids = list(results.scenario_results.keys())
    selected_scenario = st.selectbox(
        "Select scenario for detailed cash flows",
        options=scenario_ids,
        index=0,
    )
    scenario_result = results.scenario_results[selected_scenario]
    scenario_method = scenario_result.metadata.get("method", "")

    if scenario_method == "monte_carlo":
        st.markdown(f"### Expected Cash Flow Detail ' {selected_scenario}")
        st.dataframe(scenario_result.cashflows)
        _download_button(
            f"Download expected cashflows ({selected_scenario})",
            scenario_result.cashflows,
            f"cashflows_{selected_scenario}.csv",
        )

        st.markdown("### PV Distribution Statistics")
        st.dataframe(scenario_result.account_level_pv)
        _download_button(
            "Download PV statistics",
            scenario_result.account_level_pv,
            f"pv_stats_{selected_scenario}.csv",
        )

        sim_table = scenario_result.extra_tables.get("simulation_pv")
        if sim_table is not None:
            st.markdown("### Simulation-Level PV Distribution")
            st.dataframe(sim_table)
            _download_button(
                "Download simulation PV distribution",
                sim_table,
                f"pv_distribution_{selected_scenario}.csv",
            )

        viz_data = extract_monte_carlo_data(results, scenario_id=selected_scenario)
        if viz_data:
            st.markdown("### Monte Carlo Visualisations")
            col_a, col_b = st.columns(2)
            with col_a:
                fig = plot_rate_path_spaghetti(viz_data["rate_sample"], viz_data["rate_summary"])
                st.pyplot(fig)
            with col_b:
                fig = plot_rate_confidence_fan(viz_data["rate_summary"])
                st.pyplot(fig)
            fig = plot_portfolio_pv_distribution(
                viz_data["pv_distribution"],
                book_value=viz_data.get("book_value"),
                base_case_pv=viz_data.get("base_case_pv"),
                percentiles=viz_data.get("percentiles"),
            )
            st.pyplot(fig)
            fig = plot_percentile_ladder(
                viz_data.get("percentiles", {}),
                book_value=viz_data.get("book_value"),
                base_case_pv=viz_data.get("base_case_pv"),
            )
            st.pyplot(fig)
            dashboard = create_monte_carlo_dashboard(viz_data)
            st.pyplot(dashboard)
            try:
                animation_fig = create_rate_path_animation(
                    viz_data["rate_sample"], viz_data["rate_summary"]
                )
                st.plotly_chart(animation_fig, use_container_width=True, key="mc_animation")
            except Exception:
                pass
    else:
        cashflows = scenario_result.cashflows
        monthly_summary = (
            cashflows.groupby("month")[["principal", "interest", "total_cash_flow"]]
            .sum()
            .reset_index()
        )

        st.markdown(f"### Cash Flow Detail ' {selected_scenario}")
        st.dataframe(monthly_summary)
        sampled_cashflows = ReportGenerator.sample_cashflows(
            cashflows, sample_size=CASHFLOW_SAMPLE_SIZE
        )
        _download_button(
            f"Download cashflows ({selected_scenario})",
            sampled_cashflows,
            f"cashflows_{selected_scenario}.csv",
        )

        account_pv = scenario_result.account_level_pv
        _download_button(
            f"Download account-level PV ({selected_scenario})",
            account_pv,
            f"account_pv_{selected_scenario}.csv",
        )

    if progress_bar:
        progress_bar.progress(100)
    if progress_text:
        progress_text.empty()

if __name__ == "__main__":
    main()




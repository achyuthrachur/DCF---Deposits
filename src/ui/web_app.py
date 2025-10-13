"""Streamlit web interface for the NMD ALM engine."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import streamlit as st

st.cache_data.clear()
st.cache_resource.clear()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.validator import ValidationError
from src.engine import ALMEngine

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
    "checking": {"decay_rate": 0.05, "wal_years": 5.0, "beta_up": 0.40, "beta_down": 0.25},
    "savings": {"decay_rate": 0.08, "wal_years": 3.5, "beta_up": 0.55, "beta_down": 0.35},
    "money market": {"decay_rate": 0.20, "wal_years": 1.5, "beta_up": 0.75, "beta_down": 0.60},
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
            f"{segment} – Decay / Runoff Rate (annual decimal)",
            min_value=0.0,
            max_value=1.0,
            value=defaults["decay_rate"],
            step=0.01,
            key=f"decay_{segment}",
        )
        wal = st.number_input(
            f"{segment} – Weighted Average Life (years)",
            min_value=0.1,
            max_value=15.0,
            value=defaults["wal_years"],
            step=0.1,
            key=f"wal_{segment}",
        )
        beta_up = st.number_input(
            f"{segment} – Deposit Beta (rising rates)",
            min_value=0.0,
            max_value=1.5,
            value=defaults["beta_up"],
            step=0.05,
            key=f"beta_up_{segment}",
        )
        beta_down = st.number_input(
            f"{segment} – Repricing Beta (falling rates)",
            min_value=0.0,
            max_value=1.5,
            value=defaults["beta_down"],
            step=0.05,
            key=f"beta_down_{segment}",
        )
        assumption_values[segment] = {
            "decay_rate": decay,
            "wal_years": wal,
            "beta_up": beta_up,
            "beta_down": beta_down,
        }
    return assumption_values


def _collect_scenarios() -> Tuple[Dict[str, bool], Optional[Dict[str, float]]]:
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
    monte_carlo_config: Optional[Dict[str, float]] = None
    with st.expander("Monte Carlo simulation", expanded=False):
        enable_mc = st.checkbox("Run Monte Carlo simulation", value=False, key="scenario_monte_carlo")
        st.caption(
            "Monte Carlo draws apply monthly rate shocks (decimal). "
            "Results include expected cash flows and PV distribution statistics."
        )
        if enable_mc:
            scenario_flags["monte_carlo"] = True
            mc_cols = st.columns(4)
            with mc_cols[0]:
                num_sim = st.number_input(
                    "Simulations",
                    min_value=100,
                    max_value=10000,
                    value=1000,
                    step=100,
                )
            with mc_cols[1]:
                monthly_vol_bps = st.number_input(
                    "Volatility (bps / month)",
                    min_value=0.0,
                    max_value=500.0,
                    value=25.0,
                    step=5.0,
                )
            with mc_cols[2]:
                monthly_drift_bps = st.number_input(
                    "Drift (bps / month)",
                    min_value=-200.0,
                    max_value=200.0,
                    value=0.0,
                    step=1.0,
                )
            with mc_cols[3]:
                seed_input = st.text_input("Random seed (optional)", value="")
            monte_carlo_config = {
                "num_simulations": float(num_sim),
                "monthly_volatility": monthly_vol_bps / 10000,
                "monthly_drift": monthly_drift_bps / 10000,
            }
            if seed_input.strip():
                try:
                    monte_carlo_config["random_seed"] = float(int(seed_input.strip()))
                except ValueError:
                    st.warning("Random seed must be an integer. Ignoring input.")
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
        page_icon="📊",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #0f2d63 0%, #133f7f 45%, #0f2d63 100%);
            color: #ffffff;
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
        .stApp [data-testid="stCaption"],
        .stApp [data-testid="stNumberInputLabel"] p,
        .stApp [data-testid="stRadio"] label div p,
        .stApp [data-testid="stRadio"] label div span,
        .stApp [data-testid="stCheckbox"] label div p,
        .stApp [data-testid="stCheckbox"] label div span,
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
        .stButton > button {
            background: linear-gradient(135deg, #ffc94b, #f0a500);
            color: #0f2d63;
            font-weight: 700;
            border: none;
            border-radius: 999px;
            padding: 0.6rem 1.8rem;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #ffd66b, #ffb400);
            color: #08214a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero-card">
            <h1>ALM Validation – Deposit Accounts: DCF Calculation Engine</h1>
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
                    Run parallel shocks from ±100 bps to ±400 bps or launch Monte Carlo simulations with
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

    st.markdown("### Step 1 – Preview Data & Map Fields")
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

    st.markdown("### Step 2 – Configure Segmentation & Assumptions")
    segmentation_friendly = {
        "All accounts as one segment": "all",
        "Segment by account type": "by_account_type",
        "Segment by customer segment": "by_customer_segment",
        "Cross segmentation (account × customer)": "cross",
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

    st.markdown("### Step 3 – Projection Settings")
    projection_months = st.number_input(
        "Projection horizon (months)",
        min_value=12,
        max_value=360,
        step=12,
        value=120,
    )

    discount_method = st.radio(
        "Discount rate configuration",
        options=["Single rate", "Yield curve"],
        index=0,
        horizontal=True,
    )
    discount_config: Dict[int, float] | float
    if discount_method == "Single rate":
        discount_rate_input = st.number_input(
            "Discount rate (annual decimal, e.g., 0.035 for 3.5%)",
            min_value=0.0,
            max_value=0.15,
            value=0.035,
            step=0.005,
        )
        discount_config = discount_rate_input
    else:
        st.write("Enter rates for each tenor. Leave blank to omit a point.")
        tenor_values: Dict[int, float] = {}
        for tenor, label in TENOR_LABELS:
            value = st.text_input(label, value="", key=f"tenor_{tenor}")
            if value.strip():
                try:
                    tenor_values[tenor] = _decimalize(float(value.strip()))
                except ValueError:
                    st.error(f"Invalid numeric value for {label}.")
                    return
        if not tenor_values:
            st.error("At least one tenor rate is required for yield curve configuration.")
            return
        discount_config = tenor_values

    base_rate_input = st.number_input(
        "Base market rate (annual decimal, e.g., 0.03 for 3%)",
        min_value=0.0,
        max_value=0.15,
        value=0.03,
        step=0.005,
    )

    scenario_flags, monte_carlo_config = _collect_scenarios()

    run_clicked = st.button("Run Analysis", type="primary")
    if not run_clicked:
        return

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
                beta_up=_decimalize(values["beta_up"]),
                beta_down=_decimalize(values["beta_down"]),
            )
        if isinstance(discount_config, dict):
            engine.set_discount_yield_curve(discount_config)
        else:
            engine.set_discount_single_rate(_decimalize(discount_config))
        engine.set_base_market_rate_path(_decimalize(base_rate_input))
        if monte_carlo_config:
            engine.set_monte_carlo_config(
                num_simulations=int(monte_carlo_config["num_simulations"]),
                monthly_volatility=float(monte_carlo_config["monthly_volatility"]),
                monthly_drift=float(monte_carlo_config["monthly_drift"]),
                random_seed=int(monte_carlo_config["random_seed"])
                if "random_seed" in monte_carlo_config
                else None,
            )
        engine.configure_standard_scenarios(scenario_flags)
        results = engine.run_analysis(projection_months=int(projection_months))
    except ValidationError as exc:
        st.error(f"Validation error during execution: {exc}")
        return
    except Exception as exc:  # pragma: no cover - guard rail
        st.error(f"Unexpected error: {exc}")
        return


    st.session_state["run_results"] = results

    # Progress indicator for analysis runtime
    progress_text = st.empty()
    progress_bar = st.progress(0)

    progress_text.markdown("**Running deterministic scenarios...**")
    summary_df = results.summary_frame()
    progress_bar.progress(50)

    progress_text.markdown("**Preparing detailed outputs...**")
    st.markdown("### Scenario Present Value Summary")
    st.dataframe(summary_df)
    _download_button("Download summary CSV", summary_df, "eve_summary.csv")
    progress_bar.progress(75)

    scenario_ids = list(results.scenario_results.keys())
    selected_scenario = st.selectbox(
        "Select scenario for detailed cash flows",
        options=scenario_ids,
        index=0,
    )
    scenario_result = results.scenario_results[selected_scenario]
    scenario_method = scenario_result.metadata.get("method", "")

    if scenario_method == "monte_carlo":
        st.markdown(f"### Expected Cash Flow Detail – {selected_scenario}")
        st.dataframe(scenario_result.cashflows)
        _download_button(
            f"Download expected cashflows ({selected_scenario})",
            scenario_result.cashflows,
            f"cashflows_{selected_scenario}.csv",
        )
        progress_bar.progress(85)

        st.markdown("### PV Distribution Statistics")
        st.dataframe(scenario_result.account_level_pv)
        _download_button(
            "Download PV statistics",
            scenario_result.account_level_pv,
            f"pv_stats_{selected_scenario}.csv",
        )
        progress_bar.progress(92)

        sim_table = scenario_result.extra_tables.get("simulation_pv")
        if sim_table is not None:
            st.markdown("### Simulation-Level PV Distribution")
            st.dataframe(sim_table)
            _download_button(
                "Download simulation PV distribution",
                sim_table,
                f"pv_distribution_{selected_scenario}.csv",
            )
        progress_bar.progress(100)
    else:
        cashflows = scenario_result.cashflows
        monthly_summary = (
            cashflows.groupby("month")[["principal", "interest", "total_cash_flow"]]
            .sum()
            .reset_index()
        )

        st.markdown(f"### Cash Flow Detail – {selected_scenario}")
        st.dataframe(monthly_summary)
        _download_button(
            f"Download cashflows ({selected_scenario})",
            cashflows,
            f"cashflows_{selected_scenario}.csv",
        )

        account_pv = scenario_result.account_level_pv
        _download_button(
            f"Download account-level PV ({selected_scenario})",
            account_pv,
            f"account_pv_{selected_scenario}.csv",
        )
        progress_bar.progress(100)

    progress_text.empty()
    st.success("Analysis complete!")


if __name__ == "__main__":
    main()

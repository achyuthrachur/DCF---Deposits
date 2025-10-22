# -*- coding: utf-8 -*-
"""Streamlit web interface for the NMD ALM engine."""

from __future__ import annotations

import sys
import os
import logging
import base64
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import requests

IS_DESKTOP_MODE = os.environ.get("APP_DESKTOP_MODE") == "1"


try:
    from streamlit import st_autorefresh
except ImportError:  # pragma: no cover - older Streamlit
    st_autorefresh = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.ui.gh_build import render_desktop_build_expander

from src.config import FRED_API_KEY
import src.jobs as jobs
from src.security.auth import AuthManager

st.cache_data.clear()
st.cache_resource.clear()

# Configure GitHub driver from Streamlit secrets if provided
try:
    raw_secrets = getattr(st, "secrets", {})
    gh_secrets = raw_secrets.get("github") if hasattr(raw_secrets, "get") else None
    if gh_secrets:
        os.environ.setdefault("GH_REPO", str(gh_secrets.get("repo", "")))
        os.environ.setdefault("GH_TOKEN", str(gh_secrets.get("token", "")))
        if gh_secrets.get("workflow"):
            os.environ.setdefault("GH_WORKFLOW", str(gh_secrets.get("workflow")))
        if gh_secrets.get("workflow_ref"):
            os.environ.setdefault("GH_WORKFLOW_REF", str(gh_secrets.get("workflow_ref")))
        if gh_secrets.get("results_branch"):
            os.environ.setdefault("GH_RESULTS_BRANCH", str(gh_secrets.get("results_branch")))
        if gh_secrets.get("max_sims_per_job"):
            os.environ.setdefault("GH_MAX_SIMS_PER_JOB", str(gh_secrets.get("max_sims_per_job")))
    # Allow flat secrets too
    for key in ("GH_REPO", "GH_TOKEN", "GH_WORKFLOW", "GH_WORKFLOW_REF", "GH_RESULTS_BRANCH", "GH_MAX_SIMS_PER_JOB"):
        if key in raw_secrets:
            os.environ.setdefault(key, str(raw_secrets[key]))
    jobs.refresh_driver()
except Exception:
    pass

REPO_ROOT = PROJECT_ROOT.parent
AUTH_CONFIG_PATH = PROJECT_ROOT / "config" / "auth.yaml"
LOGGER = logging.getLogger(__name__)


def _trigger_auto_download(zip_bytes: bytes, file_name: str, token: str) -> None:
    """Inject a one-time auto-download script into the parent Streamlit document."""
    encoded = base64.b64encode(zip_bytes).decode()
    payload = f"""
    <html>
    <body>
    <script>
    (function() {{
        const token = {json.dumps(token)};
        const parentWindow = window.parent;
        if (!parentWindow) {{
            return;
        }}
        if (parentWindow.__streamlit_auto_download_token === token) {{
            return;
        }}
        const doc = parentWindow.document;
        const link = doc.createElement("a");
        link.href = "data:application/zip;base64,{encoded}";
        link.download = {json.dumps(file_name)};
        link.style.display = "none";
        doc.body.appendChild(link);
        link.click();
        doc.body.removeChild(link);
        parentWindow.__streamlit_auto_download_token = token;
    }})();
    </script>
    </body>
    </html>
    """
    components.html(payload, height=0, width=0)


def _get_auth_manager() -> AuthManager:
    return st.session_state.setdefault("auth_manager", AuthManager(AUTH_CONFIG_PATH))


def _ensure_authenticated() -> bool:
    auth_manager = _get_auth_manager()
    if "auth_user" not in st.session_state:
        st.session_state["auth_user"] = None
    if st.session_state["auth_user"]:
        return True

    users = auth_manager.users()
    notifier = auth_manager.notifier()
    active_users = [user for user in users if user.active]

    st.markdown("### Secure Sign-in")

    if not active_users:
        st.warning(
            "No active user accounts were found. Initialise the administrator account below to continue."
        )
        default_user = users[0] if users else None
        with st.form("initialise_admin", clear_on_submit=True):
            username_field = st.text_input(
                "Administrator username",
                value=default_user.username if default_user else "admin",
                disabled=bool(default_user and default_user.username),
            )
            display_name = st.text_input(
                "Display name",
                value=default_user.name if default_user else "Administrator",
            )
            email_value = st.text_input(
                "Administrator email",
                value=default_user.email if default_user else "",
            )
            password = st.text_input("New password", type="password")
            confirm = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Activate administrator")
        if submitted:
            if password != confirm:
                st.error("Passwords do not match.")
            elif len(password) < 8:
                st.error("Please choose a password with at least 8 characters.")
            else:
                username_value = username_field or (default_user.username if default_user else "admin")
                success, message = auth_manager.set_password_direct(
                    username_value,
                    password,
                    display_name=display_name or username_value,
                    email=email_value or (default_user.email if default_user else ""),
                )
                if success:
                    st.success("Administrator account initialised. Please sign in below.")
                else:
                    st.error(message)
        return False

    login_tab, activate_tab, reset_tab = st.tabs(["Sign in", "Activate account", "Reset password"])

    with login_tab.form("signin_form", clear_on_submit=True):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        remember_me = st.checkbox("Remember me", value=False)
        submitted = st.form_submit_button("Sign in")
    if submitted:
        user = auth_manager.authenticate(username, password)
        if user:
            st.session_state["auth_user"] = {
                "username": user.username,
                "name": user.name,
                "email": user.email,
                "remember": remember_me,
            }
            st.success(f"Welcome back, {user.name}!")
            st.rerun()
        else:
            st.error("Invalid username or password.")

    with activate_tab.expander("Request activation link", expanded=True):
        with st.form("activation_request", clear_on_submit=True):
            req_email = st.text_input("Email", key="activation_email")
            send_request = st.form_submit_button("Generate activation code")
        if send_request:
            if not req_email:
                st.error("Provide an email address to request activation.")
            else:
                result = auth_manager.request_activation_by_email(req_email)
                if result.sent:
                    st.session_state["activation_last_token"] = result.token
                    st.success(result.message)
                else:
                    st.warning(result.message)
        generated_token = st.session_state.get("activation_last_token")
        if generated_token:
            st.caption("Activation token (share with the requester; valid for one hour):")
            st.code(generated_token, language="text")

    with activate_tab.form("activation_complete", clear_on_submit=True):
        token_value = st.text_input("Activation token")
        desired_username = st.text_input("Choose a username", key="activation_desired_username")
        display_name = st.text_input("Display name (optional)")
        new_password = st.text_input("Create password", type="password")
        confirm_password = st.text_input("Confirm password", type="password")
        complete_activation = st.form_submit_button("Activate account")
    if complete_activation:
        if not desired_username:
            st.error("Choose a username to complete activation.")
        elif new_password != confirm_password:
            st.error("Passwords do not match.")
        elif len(new_password) < 8:
            st.error("Please choose a password with at least 8 characters.")
        else:
            success, message = auth_manager.complete_activation_with_username(
                token_value,
                desired_username,
                new_password,
                display_name=display_name or None,
            )
            if success:
                st.success(message)
                st.session_state.pop("activation_last_token", None)
            else:
                st.error(message)

    with reset_tab.expander("Request password reset", expanded=True):
        with st.form("password_reset_request", clear_on_submit=True):
            reset_username = st.text_input("Username", key="reset_username")
            reset_email = st.text_input("Email", key="reset_email")
            send_reset = st.form_submit_button("Send reset email")
        if send_reset:
            if not reset_username or not reset_email:
                st.error("Provide both username and email to request a reset token.")
            else:
                result = auth_manager.request_password_reset(reset_username, reset_email, notifier)
                if result.sent:
                    st.success(result.message)
                else:
                    st.warning(result.message)

    with reset_tab.form("password_reset_complete", clear_on_submit=True):
        reset_token = st.text_input("Reset token")
        new_password = st.text_input("New password", type="password")
        confirm_password = st.text_input("Confirm password", type="password")
        complete_reset = st.form_submit_button("Update password")
    if complete_reset:
        if new_password != confirm_password:
            st.error("Passwords do not match.")
        elif len(new_password) < 8:
            st.error("Please choose a password with at least 8 characters.")
        else:
            success, message = auth_manager.complete_password_reset(reset_token, new_password)
            if success:
                st.success(message)
            else:
                st.error(message)

    st.stop()

def _latest_release_windows_asset(repo_full: str) -> tuple[str,str] | None:
    try:
        owner, repo = repo_full.split('/', 1)
        r = requests.get(f'https://api.github.com/repos/{owner}/{repo}/releases/latest', timeout=8)
        if r.status_code != 200:
            return None
        data = r.json()
        for asset in data.get('assets', []):
            name = (asset.get('name') or '').lower()
            if name.endswith('.exe') and ('dcf' in name or 'deposits' in name):
                return asset.get('name') or 'Windows Installer', asset.get('browser_download_url')
    except Exception:
        return None
    return None


def _render_desktop_download() -> None:
    gh_repo = os.environ.get('GH_REPO')
    if not gh_repo and hasattr(st, 'secrets'):
        try:
            gh_repo = st.secrets.get('github', {}).get('repo')
        except Exception:
            gh_repo = None
    if not gh_repo:
        return
    with st.expander('Download Desktop App (Windows)', expanded=False):
        asset = _latest_release_windows_asset(gh_repo)
        if asset is None:
            st.caption("Installer not available yet. Ask the maintainer to run the 'Build Windows Desktop App' workflow.")
        else:
            name, url = asset
            st.write('Run locally with no Python install. Built with PyInstaller.')
            try:
                st.link_button(f'Download {name}', url)
            except Exception:
                st.markdown(f"[Download {name}]({url})")

from src.core.validator import ValidationError
from src.engine import ALMEngine
from src.core.fred_loader import FREDYieldCurveLoader
from src.core.yield_curve import YieldCurve
from src.core.monte_carlo import MonteCarloConfig, MonteCarloLevel, VasicekParams
from src.core.decay import resolve_decay_parameters
from src.reporting import InMemoryReportBuilder
from src.utils.numbers import decimalize
from src.visualization import (
    create_monte_carlo_dashboard,
    create_rate_path_animation,
    extract_monte_carlo_data,
    extract_shock_data,
    plot_percentile_ladder,
    plot_portfolio_pv_distribution,
    plot_rate_confidence_fan,
    plot_rate_path_spaghetti,
    plot_shock_magnitude,
    plot_shock_pv_delta,
    plot_shock_rate_paths,
    plot_shock_tenor_comparison,
)
try:
    from src.visualization import plot_shock_group_summary
except ImportError:  # pragma: no cover - compatibility for older package builds
    from src.visualization.shock_plots import plot_shock_group_summary  # type: ignore

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
        "decay_rate": 0.1828,
        "wal_years": 5.0,
        "deposit_beta_up": 0.40,
        "deposit_beta_down": 0.25,
        "repricing_beta_up": 1.00,
        "repricing_beta_down": 1.00,
    },
    "savings": {
        "decay_rate": 0.2511,
        "wal_years": 3.5,
        "deposit_beta_up": 0.55,
        "deposit_beta_down": 0.35,
        "repricing_beta_up": 1.00,
        "repricing_beta_down": 1.00,
    },
    "money market": {
        "decay_rate": 0.4961,
        "wal_years": 1.5,
        "deposit_beta_up": 0.75,
        "deposit_beta_down": 0.60,
        "repricing_beta_up": 1.00,
        "repricing_beta_down": 1.00,
    },
}

DEFAULT_SINGLE_RATE = 0.035
DEFAULT_MANUAL_TENOR_RATES: Dict[int, float] = {
    3: 0.0300,
    6: 0.0310,
    12: 0.0325,
    24: 0.0335,
    36: 0.0350,
    60: 0.0365,
    84: 0.0380,
    120: 0.0400,
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

def _select_sample_months(actual_months: Optional[int]) -> List[int]:
    """Return representative months for sample cash flow output."""
    if not actual_months:
        return [1]
    canonical = [1, 12, 24, 60, 120, 180, 240]
    months = [month for month in canonical if month <= actual_months]
    if actual_months not in months:
        months.append(actual_months)
    return sorted(set(months))


def _build_analysis_payload(
    *,
    field_map: Dict[str, str],
    optional_fields: Iterable[str],
    segmentation: str,
    assumptions: Dict[str, Dict[str, Any]],
    projection_months: int,
    max_projection_months: int,
    materiality_threshold: float,
    scenario_flags: Dict[str, bool],
    discount_config: Dict[str, Any],
    base_market_path: Iterable[float],
    monte_carlo_config: Optional[MonteCarloConfig],
    package_options: Dict[str, Any],
    selected_curve: Optional[YieldCurve],
    report_title: str = "Deposit Analysis",
) -> Dict[str, Any]:
    """Serialize the current analysis configuration for the background worker."""

    assumptions_payload: Dict[str, Dict[str, Any]] = {}
    for segment_key, values in assumptions.items():
        assumptions_payload[segment_key] = {
            "decay_rate": float(values["decay_rate"]) if values.get("decay_rate") is not None else None,
            "wal_years": float(values["wal_years"]) if values.get("wal_years") is not None else None,
            "deposit_beta_up": float(values.get("deposit_beta_up", 0.0)),
            "deposit_beta_down": float(values.get("deposit_beta_down", 0.0)),
            "repricing_beta_up": float(values.get("repricing_beta_up", 0.0)),
            "repricing_beta_down": float(values.get("repricing_beta_down", 0.0)),
            "decay_priority": values.get("decay_priority", "auto"),
        }

    discount_payload: Dict[str, Any] = {"mode": discount_config.get("mode")}
    if discount_config.get("mode") == "single":
        discount_payload["rate"] = float(discount_config.get("rate", 0.0))
    elif discount_config.get("mode") == "manual":
        discount_payload["tenor_rates"] = {
            int(k): float(v) for k, v in discount_config.get("tenor_rates", {}).items()
        }
        discount_payload["interpolation"] = discount_config.get("interpolation", "linear")
    elif discount_config.get("mode") == "fred":
        discount_payload["api_key"] = discount_config.get("api_key")
        discount_payload["interpolation"] = discount_config.get("interpolation", "linear")
        discount_payload["target_date"] = discount_config.get("target_date")

    curve_snapshot: Optional[Dict[str, Any]] = None
    if selected_curve is not None:
        curve_snapshot = {
            "tenors": [int(t) for t in selected_curve.tenors],
            "rates": [float(r) for r in selected_curve.rates],
            "interpolation": selected_curve.interpolation_method,
            "metadata": getattr(selected_curve, "metadata", {}),
        }
    discount_payload["curve_snapshot"] = curve_snapshot

    monte_carlo_metadata = (
        monte_carlo_config.to_metadata() if isinstance(monte_carlo_config, MonteCarloConfig) else None
    )

    payload: Dict[str, Any] = {
        "field_map": field_map,
        "optional_fields": list(optional_fields or []),
        "segmentation": segmentation,
        "assumptions": assumptions_payload,
        "projection_settings": {
            "projection_months": int(projection_months),
            "max_projection_months": int(max_projection_months),
            "materiality_threshold": float(materiality_threshold),
        },
        "scenario_flags": scenario_flags,
        "discount_config": discount_payload,
        "base_market_path": [float(value) for value in base_market_path],
        "monte_carlo_config": monte_carlo_metadata,
        "package_options": package_options,
        "report_title": report_title,
    }
    return payload


def _build_cashflow_sample_table(
    cashflows: pd.DataFrame, actual_months: Optional[int]
) -> pd.DataFrame:
    """Aggregate portfolio cash flows for selected months."""
    if cashflows.empty:
        return pd.DataFrame(
            columns=[
                "Month",
                "Begin Balance",
                "Decay %",
                "Runoff",
                "End Balance",
                "Int Rate",
                "Interest",
                "Total Outflow",
            ]
        )
    sample_months = _select_sample_months(actual_months)
    aggregated = (
        cashflows.groupby("month")
        .agg(
            beginning_balance=("beginning_balance", "sum"),
            ending_balance=("ending_balance", "sum"),
            principal=("principal", "sum"),
            interest=("interest", "sum"),
            total_cash_flow=("total_cash_flow", "sum"),
            deposit_rate=("deposit_rate", "mean"),
            monthly_decay_rate=("monthly_decay_rate", "mean"),
        )
        .reset_index()
    )
    subset = aggregated[aggregated["month"].isin(sample_months)].copy()
    if subset.empty:
        return pd.DataFrame()
    subset["Decay %"] = subset["monthly_decay_rate"] * 100.0
    subset["Int Rate"] = subset["deposit_rate"] * 100.0
    subset = subset.rename(
        columns={
            "month": "Month",
            "beginning_balance": "Begin Balance",
            "principal": "Runoff",
            "ending_balance": "End Balance",
            "interest": "Interest",
            "total_cash_flow": "Total Outflow",
        }
    )
    ordered_columns = [
        "Month",
        "Begin Balance",
        "Decay %",
        "Runoff",
        "End Balance",
        "Int Rate",
        "Interest",
        "Total Outflow",
    ]
    subset = subset[ordered_columns]
    for column in ordered_columns:
        if column != "Month":
            subset[column] = subset[column].astype(float)
    return subset.sort_values("Month").reset_index(drop=True)


def _render_parameter_summary(results: "EngineResults") -> None:
    """Display parameter summary information in the UI."""
    summary = results.parameter_summary or {}
    if not summary:
        st.info("Parameter summary unavailable. Run the analysis to refresh results.")
        return

    portfolio = summary.get("portfolio", {})
    projection = summary.get("projection", {})
    segments = summary.get("segments", [])

    st.markdown("### NMD Parameter Summary")
    st.markdown(
        f"- Starting Balance: ${portfolio.get('starting_balance', 0.0):,.2f}\n"
        f"- Account Count: {portfolio.get('account_count', 0)}\n"
        f"- Portfolio Weighted WAL: {portfolio.get('weighted_average_wal_years') or 0:.2f} years"
    )
    st.markdown(
        f"- Base Horizon: {projection.get('projection_months')} months\n"
        f"- Max Horizon: {projection.get('max_projection_months')} months\n"
        f"- Materiality Threshold: ${projection.get('materiality_threshold', 0):,.2f}\n"
        f"- Actual Projection Length: {projection.get('actual_projection_months') or 0} months\n"
        f"- Residual Balance: ${projection.get('residual_balance') or 0:,.2f}"
    )

    for segment in segments:
        st.markdown(f"#### Segment: {segment.get('segment')}")
        wal_input = segment.get("input_wal_years")
        decay_input = segment.get("input_decay_rate")
        resolved_wal = segment.get("resolved_wal_years")
        resolved_decay = segment.get("resolved_decay_rate")
        monthly_decay = segment.get("monthly_decay_rate")
        lines = []
        if wal_input is not None:
            lines.append(f"Input WAL: {wal_input:.2f} years")
        if decay_input is not None:
            lines.append(f"Input Decay Rate: {decay_input * 100:.2f}%")
        lines.append(f"Calculated WAL: {resolved_wal:.2f} years")
        lines.append(f"Calculated Annual Decay: {resolved_decay * 100:.2f}%")
        lines.append(f"Calculated Monthly Decay: {monthly_decay * 100:.2f}%")
        lines.append(
            f"Deposit Betas: up {segment.get('deposit_beta_up'):.2f} | down {segment.get('deposit_beta_down'):.2f}"
        )
        st.markdown("\n".join(f"- {item}" for item in lines))
        warning = segment.get("decay_warning")
        if warning:
            st.warning(warning)

    base_id = results.base_scenario_id
    if base_id and base_id in results.scenario_results:
        base_result = results.scenario_results[base_id]
        sample_table = _build_cashflow_sample_table(
            base_result.cashflows,
            projection.get("actual_projection_months"),
        )
        if not sample_table.empty:
            st.markdown("#### Deposit Cash Flow Schedule (Sample Months)")
            st.dataframe(sample_table, width='stretch')


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
            min_value=0.0,
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
        decay_input = decay if decay > 0 else None
        wal_input = wal if wal > 0 else None

        resolution = None
        try:
            resolution = resolve_decay_parameters(wal_input, decay_input, priority="auto")
        except ValueError as exc:
            st.error(f"{segment}: {exc}")
            continue

        decay_priority = resolution.priority_used
        if resolution.inconsistent and resolution.warning:
            st.warning(
                f"{segment}: {resolution.warning} "
                f"Difference: {resolution.difference_years:.2f} years "
                f"({(resolution.difference_ratio or 0.0) * 100:.1f}%)."
            )
            priority_label = st.radio(
                "Which parameter should take precedence?",
                options=[
                    "Use WAL (override decay rate)",
                    "Use decay rate (override WAL)",
                ],
                index=0 if resolution.priority_used == "wal" else 1,
                key=f"decay_priority_{segment}",
                horizontal=True,
            )
            decay_priority = "wal" if priority_label.startswith("Use WAL") else "decay"
        else:
            if wal_input is not None and decay_input is not None:
                st.info(
                    f"{segment}: WAL and decay inputs align. "
                    f"Difference {resolution.difference_years or 0.0:.2f} years."
                )
            elif wal_input is not None:
                st.info(
                    f"{segment}: WAL provided G-> derived annual decay "
                    f"{resolution.annual_decay_rate * 100:.2f}% "
                    f"(monthly {resolution.monthly_decay_rate * 100:.2f}%)."
                )
            elif decay_input is not None:
                st.info(
                    f"{segment}: Decay rate provided G-> implied WAL "
                    f"{resolution.wal_years:.2f} years "
                    f"(monthly {resolution.monthly_decay_rate * 100:.2f}%)."
                )

        summary_lines = []
        if wal_input is not None:
            summary_lines.append(f"Input: Weighted Average Life = {wal_input:.2f} years")
        if decay_input is not None:
            summary_lines.append(f"Input: Annual Decay Rate = {decay_input * 100:.2f}%")
        summary_lines.append(
            f"Calculated: Annual Decay Rate = {resolution.annual_decay_rate * 100:.2f}%"
        )
        summary_lines.append(
            f"Calculated: Weighted Average Life = {resolution.wal_years:.2f} years"
        )
        summary_lines.append(
            f"Calculated: Monthly Decay Rate = {resolution.monthly_decay_rate * 100:.2f}%"
        )
        st.markdown("\n".join(f"- {line}" for line in summary_lines))

        assumption_values[segment] = {
            "decay_rate": decay_input,
            "wal_years": wal_input,
            "deposit_beta_up": deposit_beta_up,
            "deposit_beta_down": deposit_beta_down,
            "repricing_beta_up": repricing_beta_up,
            "repricing_beta_down": repricing_beta_down,
            "decay_priority": decay_priority,
            "resolved_decay_rate": resolution.annual_decay_rate,
            "resolved_wal_years": resolution.wal_years,
            "resolved_monthly_decay_rate": resolution.monthly_decay_rate,
            "decay_warning": resolution.warning,
            "implied_wal_from_decay": resolution.implied_wal_from_decay,
            "implied_decay_from_wal": resolution.implied_decay_from_wal,
            "decay_difference_years": resolution.difference_years,
            "decay_difference_ratio": resolution.difference_ratio,
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
                short_speed = st.number_input(
                    "Reversion speed to 3M rate",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.15,
                    step=0.01,
                    format="%.2f",
                    help="How quickly the simulated 3M rate gravitates back toward its typical level (0 = never, 1 = instantly).",
                    key="mc_short_speed",
                )
            with short_col[1]:
                short_avg_pct = st.number_input(
                    "Short-term average rate (%)",
                    min_value=0.0,
                    max_value=20.0,
                    value=float(short_anchor) * 100.0,
                    step=0.05,
                    format="%.2f",
                    help="Long-run average level for the 3M rate.",
                    key="mc_short_avg_pct",
                )
            with short_col[2]:
                short_vol_pct = st.number_input(
                    "Annual volatility of 3M rate (%)",
                    min_value=0.0,
                    max_value=10.0,
                    value=1.0,
                    step=0.05,
                    format="%.2f",
                    help="Annualised volatility applied to the 3M rate path.",
                    key="mc_short_vol_pct",
                )
            short_params = VasicekParams(
                mean_reversion=float(short_speed),
                long_term_mean=float(short_avg_pct) / 100.0,
                volatility=float(short_vol_pct) / 100.0,
            )

            long_params = None
            correlation = 0.70
            if level == MonteCarloLevel.TWO_FACTOR:
                long_col = st.columns(3)
                with long_col[0]:
                    long_speed = st.number_input(
                        "Reversion speed to 10Y rate",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.07,
                        step=0.01,
                        format="%.2f",
                        help="How quickly the simulated 10Y anchor returns toward its typical level.",
                        key="mc_long_speed",
                    )
                with long_col[1]:
                    long_avg_pct = st.number_input(
                        "Long-term average 10Y rate (%)",
                        min_value=0.0,
                        max_value=20.0,
                        value=float(long_anchor) * 100.0,
                        step=0.05,
                        format="%.2f",
                        help="Long-run average level for the 10Y anchor.",
                        key="mc_long_avg_pct",
                    )
                with long_col[2]:
                    long_vol_pct = st.number_input(
                        "Annual volatility of 10Y rate (%)",
                        min_value=0.0,
                        max_value=10.0,
                        value=0.8,
                        step=0.05,
                        format="%.2f",
                        help="Annualised volatility applied to the 10Y anchor.",
                        key="mc_long_vol_pct",
                    )
                long_params = VasicekParams(
                    mean_reversion=float(long_speed),
                    long_term_mean=float(long_avg_pct) / 100.0,
                    volatility=float(long_vol_pct) / 100.0,
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

def _load_uploaded_file(uploaded_file: "st.runtime.uploaded_file_manager.UploadedFile") -> pd.DataFrame:
    """Read uploaded CSV or Excel file into a dataframe."""
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)


def _find_rate_outliers(curve: Optional[YieldCurve]) -> Tuple[List[int], List[int]]:
    """Return lists of tenor months with negative or >20% annual rates."""
    if curve is None:
        return [], []
    tenors = [int(float(t)) for t in curve.tenors]
    rates = list(curve.rates)
    negative = []
    excessive = []
    for tenor, rate in zip(tenors, rates):
        if rate < -1e-9:
            negative.append(tenor)
        elif rate > 0.20 + 1e-9:
            excessive.append(tenor)
    return negative, excessive


def _reset_discount_defaults(discount_method: str) -> None:
    """Restore discount inputs to a safe default configuration and rerun."""
    if discount_method == "Single rate":
        st.session_state["discount_method_choice"] = "Single rate"
        st.session_state["single_rate_input"] = DEFAULT_SINGLE_RATE
    elif discount_method == "Manual yield curve":
        st.session_state["discount_method_choice"] = "Manual yield curve"
        for tenor, rate in DEFAULT_MANUAL_TENOR_RATES.items():
            st.session_state[f"tenor_{tenor}"] = f"{rate:.4f}"
    else:
        st.session_state["discount_method_choice"] = "Single rate"
        st.session_state["single_rate_input"] = DEFAULT_SINGLE_RATE
    st.session_state.pop("run_results", None)
    st.rerun()


def _render_rate_adjustment_controls(discount_method: str) -> None:
    """Offer controls to reset discount curve inputs after invalid rates."""
    if discount_method == "Single rate":
        if st.button("Reset to a 3.5% flat discount curve", key="reset_single_curve"):
            _reset_discount_defaults("Single rate")
    elif discount_method == "Manual yield curve":
        if st.button(
            "Reset manual curve to default tenor rates",
            key="reset_manual_curve",
        ):
            _reset_discount_defaults("Manual yield curve")
    else:
        if st.button(
            "Switch to default 3.5% flat curve",
            key="reset_other_curve",
        ):
            _reset_discount_defaults("Single rate")


def _render_invalid_rate_message(discount_method: str, details: str) -> None:
    detail_text = details.rstrip(".")
    st.error(
        "The selected parameters produced discount rates outside the supported "
        "range (0%G->20%). "
        f"{detail_text}. "
        "Please adjust the curve inputs or reset them to the defaults below."
    )
    _render_rate_adjustment_controls(discount_method)


def _handle_rate_outliers(curve: Optional[YieldCurve], discount_method: str) -> bool:
    """Display a friendly message if the chosen curve falls outside bounds."""
    negative, excessive = _find_rate_outliers(curve)
    if not negative and not excessive:
        return False
    fragments: List[str] = []
    if negative:
        fragments.append(
            f"Negative rates detected at months {', '.join(str(t) for t in negative)}"
        )
    if excessive:
        fragments.append(
            f"Rates above 20% detected at months {', '.join(str(t) for t in excessive)}"
        )
    details = "; ".join(fragments)
    _render_invalid_rate_message(discount_method, details)
    return True


def _scenario_display_name(results, scenario_id: str) -> str:
    scenario = results.scenario_results.get(scenario_id)
    if scenario is None:
        return scenario_id
    metadata = scenario.metadata or {}
    label = metadata.get("description")
    if label:
        return label
    return scenario_id.replace("_", " ").title()


def _render_shock_detail(results, scenario_id: str) -> None:
    shock_viz_data = extract_shock_data(results, scenario_id)
    if not shock_viz_data:
        st.info("Visualisations are unavailable for this scenario.")
        return

    metrics = st.columns(3)
    metrics[0].metric("Scenario PV", f"${shock_viz_data['scenario_pv']:,.0f}")

    delta = shock_viz_data.get("delta")
    delta_pct = shock_viz_data.get("delta_pct")
    if delta is not None:
        pct_display = f"{delta_pct * 100:+.2f}%" if delta_pct is not None else ""
        metrics[1].metric("+Change vs Base", f"${delta:,.0f}", pct_display)
    else:
        metrics[1].metric("+Change vs Base", "N/A")

    metrics[2].metric(
        "Max |Shock|",
        f"{shock_viz_data['abs_max_bps']:,.0f} bps",
        f"Avg {shock_viz_data['mean_bps']:.1f} bps",
    )

    col_left, col_right = st.columns(2)
    with col_left:
        fig = plot_shock_rate_paths(
            shock_viz_data["curve_comparison"],
            shock_viz_data["scenario_label"],
        )
        st.pyplot(fig)
    with col_right:
        fig = plot_shock_magnitude(
            shock_viz_data["curve_comparison"],
            shock_viz_data["scenario_label"],
        )
        st.pyplot(fig)

    tenor_fig = plot_shock_tenor_comparison(
        shock_viz_data["tenor_comparison"],
        shock_viz_data["scenario_label"],
    )
    if tenor_fig:
        st.pyplot(tenor_fig)

    pv_fig = plot_shock_pv_delta(
        base_pv=shock_viz_data["base_pv"],
        scenario_pv=shock_viz_data["scenario_pv"],
        scenario_label=shock_viz_data["scenario_label"],
        delta=delta,
        delta_pct=delta_pct,
    )
    if pv_fig:
        st.pyplot(pv_fig)


def _render_shock_group(
    results,
    scenario_ids: List[str],
    *,
    title: str,
    select_label: str,
    select_key: str,
) -> None:
    if not scenario_ids:
        st.info("No scenarios are available in this category.")
        return

    ordered_ids = sorted(
        scenario_ids,
        key=lambda sid: _scenario_display_name(results, sid),
    )

    summary_fig = plot_shock_group_summary(
        results,
        ordered_ids,
        title=title,
    )
    if summary_fig:
        st.pyplot(summary_fig)

    selected_id = st.selectbox(
        select_label,
        options=ordered_ids,
        format_func=lambda sid: _scenario_display_name(results, sid),
        key=select_key,
    )
    _render_shock_detail(results, selected_id)


def _render_monte_carlo_detail(results, scenario_id: str) -> None:
    scenario_result = results.scenario_results.get(scenario_id)
    if scenario_result is None:
        st.info("Scenario not found.")
        return

    viz_data = extract_monte_carlo_data(results, scenario_id=scenario_id)
    if not viz_data:
        st.info("Monte Carlo visualisations are unavailable.")
        return

    base_result = (
        results.scenario_results.get(results.base_scenario_id)
        if results.base_scenario_id
        else None
    )
    base_pv = float(base_result.present_value) if base_result else None
    scenario_pv = float(scenario_result.present_value)

    metrics = st.columns(3)
    metrics[0].metric("Scenario PV", f"${scenario_pv:,.0f}")

    if base_pv is not None:
        delta = scenario_pv - base_pv
        pct = delta / base_pv if base_pv else None
        pct_display = f"{pct * 100:+.2f}%" if pct is not None else ""
        metrics[1].metric("+Change vs Base", f"${delta:,.0f}", pct_display)
    else:
        metrics[1].metric("+Change vs Base", "N/A")

    percentiles = viz_data.get("percentiles", {})
    if percentiles and percentiles.get("p5") is not None and percentiles.get("p95") is not None:
        metrics[2].metric(
            "Central 90% PV",
            f"${percentiles['p5']:,.0f} G-> ${percentiles['p95']:,.0f}",
        )
    else:
        metrics[2].metric("Central 90% PV", "N/A")

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
        percentiles=percentiles,
    )
    st.pyplot(fig)

    fig = plot_percentile_ladder(
        percentiles,
        book_value=viz_data.get("book_value"),
        base_case_pv=viz_data.get("base_case_pv"),
    )
    st.pyplot(fig)

    dashboard = create_monte_carlo_dashboard(viz_data)
    st.pyplot(dashboard)

    try:
        animation_fig = create_rate_path_animation(
            viz_data["rate_sample"],
            viz_data["rate_summary"],
        )
    except Exception as exc:
        st.warning(
            f"Monte Carlo animation unavailable: {exc}",
            icon="GRun Analysis",
        )
    else:
        st.plotly_chart(
            animation_fig,
            width='stretch',
            key=f"mc_animation_{scenario_id}",
        )


def _render_monte_carlo_group(results, scenario_ids: List[str]) -> None:
    if not scenario_ids:
        st.info("Monte Carlo simulations have not been configured.")
        return

    ordered_ids = sorted(
        scenario_ids,
        key=lambda sid: _scenario_display_name(results, sid),
    )

    summary_fig = plot_shock_group_summary(
        results,
        ordered_ids,
        title="Monte Carlo Present Value Comparison",
    )
    if summary_fig:
        st.pyplot(summary_fig)

    selected_id = st.selectbox(
        "Select Monte Carlo scenario",
        options=ordered_ids,
        format_func=lambda sid: _scenario_display_name(results, sid),
        key="monte_carlo_select",
    )
    _render_monte_carlo_detail(results, selected_id)


def main() -> None:
    """Streamlit application entry point."""
    st.set_page_config(
        page_title="ALM Validation - Deposit Accounts: DCF Calculation Engine",
        page_icon="",
        layout="wide",
    )

    if "active_job" not in st.session_state:
        st.session_state["active_job"] = None
    if "analysis_metadata" not in st.session_state:
        st.session_state["analysis_metadata"] = None
    if "analysis_status_message" not in st.session_state:
        st.session_state["analysis_status_message"] = None

    if not _ensure_authenticated():
        return

    auth_user = st.session_state.get("auth_user")
    with st.sidebar:
        if auth_user:
            st.markdown(f"**Signed in as {auth_user['name']}**")
        if st.button("Sign out"):
            st.session_state["auth_user"] = None
            st.rerun()

    if not IS_DESKTOP_MODE:
        render_desktop_build_expander()

    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #0f2d63 0%, #133f7f 45%, #0f2d63 100%);
            color: #ffffff;
        }
        .brand-badge {
            position: absolute;
            top: 30px;
            left: 32px;
            z-index: 1100;
            padding: 6px 10px;
        }
        .brand-badge img {
            max-width: 150px;
            height: auto;
            filter: drop-shadow(0 4px 10px rgba(0,0,0,0.35));
        }
        .brand-badge svg {
            width: 160px;
            height: auto;
            display: block;
            filter: drop-shadow(0 6px 12px rgba(0,0,0,0.35));
        }
        .main .block-container {
            padding-top: 140px;
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
            color: #f3f7ff;
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
            color: #eef4ff;
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
            color: #eef3ff !important;
        }
        .stApp .stNumberInput input,
        .stApp .stTextInput input,
        .stApp textarea,
        .stApp div[data-baseweb="input"] input {
            background: rgba(255, 255, 255, 0.12) !important;
            border: 1px solid rgba(244, 247, 255, 0.45) !important;
            color: #f7faff !important;
        }
        .stApp .stNumberInput input::placeholder,
        .stApp .stTextInput input::placeholder,
        .stApp textarea::placeholder,
        .stApp div[data-baseweb="input"] input::placeholder {
            color: rgba(247, 250, 255, 0.72) !important;
        }
        .stApp div[data-baseweb="select"] span,
        .stApp .stMultiSelect div[data-baseweb="tag"] span,
        .stApp div[data-baseweb="select"] input {
            color: #f7faff !important;
        }
        .stApp .stMultiSelect div[data-baseweb="tag"] {
            background: rgba(255, 255, 255, 0.18) !important;
            border: 1px solid rgba(244, 247, 255, 0.35) !important;
        }
        .stApp div[data-baseweb="select"] div[role="button"] {
            background: rgba(255, 255, 255, 0.12) !important;
            border: 1px solid rgba(244, 247, 255, 0.45) !important;
            color: #f7faff !important;
        }
        .stApp div[data-baseweb="select"] div[role="button"]:hover,
        .stApp div[data-baseweb="select"] div[role="button"]:focus,
        .stApp .stNumberInput input:focus,
        .stApp .stTextInput input:focus,
        .stApp textarea:focus {
            border-color: #ffc94b !important;
            box-shadow: 0 0 0 2px rgba(255, 201, 75, 0.35) !important;
        }
        .stApp div[data-baseweb="popover"] {
            background: rgba(7, 20, 52, 0.95) !important;
            color: #f7faff !important;
            border: 1px solid rgba(244, 247, 255, 0.35) !important;
        }
        .stApp div[data-baseweb="popover"] li,
        .stApp div[data-baseweb="popover"] span {
            color: #f7faff !important;
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
        .stFileUploader label,
        .stFileUploader label span,
        .stFileUploader div[data-testid="stMarkdownContainer"] p,
        .stFileUploader div[data-testid=\"stFileUploaderDropzone\"] div {
            color: #fdfdfd !important;
            font-weight: 600;
        }
        .stFileUploader div[data-testid=\"stFileUploaderDropzone\"] {
            background: rgba(255,255,255,0.15) !important;
            border: 1px dashed rgba(255,255,255,0.55) !important;
        }
        .stFileUploader [data-testid="stUploadedFileName"],
        .stFileUploader [data-testid="stUploadedFileSize"] {
            color: #fdfdfd !important;
            font-weight: 600;
        }
        .stDataFrame, .stTable {
            background: rgba(255, 255, 255, 0.95) !important;
            border-radius: 12px;
            padding: 10px;
        }
        div[data-testid="stMetricLabel"] {
            color: #f8fbff !important;
            font-weight: 600 !important;
        }
        div[data-testid="stMetricValue"] {
            color: #fdfdfd !important;
            font-weight: 700 !important;
            text-shadow: 0 0 6px rgba(15, 45, 99, 0.45);
        }
        div[data-testid="stMetricDelta"] {
            color: #9ef7c9 !important;
            font-weight: 600 !important;
            text-shadow: 0 0 4px rgba(0, 0, 0, 0.25);
        }
        div[data-testid="stMetricDelta"] svg {
            fill: currentColor !important;
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
        unsafe_allow_html=True)
    _render_desktop_download()


    brand_html = _brand_logo_html()
    if brand_html:
        st.markdown(brand_html, unsafe_allow_html=True)

    if IS_DESKTOP_MODE:
        st.info("Desktop mode: calculations run on this machine and outputs are saved under the `output` folder next to the application.")

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
    _render_desktop_download()
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

    wal_lookup = {segment: values["resolved_wal_years"] for segment, values in assumptions.items()}
    wal_summary = segment_summary.copy()
    wal_summary["wal_years"] = wal_summary["segment"].map(wal_lookup)
    wal_summary = wal_summary.dropna(subset=["wal_years"])
    total_balance = float(wal_summary["balance"].abs().sum())
    if total_balance > 0:
        weighted_wal_years = float(
            (wal_summary["balance"].abs() * wal_summary["wal_years"]).sum() / total_balance
        )
    elif wal_lookup:
        weighted_wal_years = float(sum(value for value in wal_lookup.values() if value))
    else:
        weighted_wal_years = 0.0

    st.caption(
        f"Portfolio weighted average life (reference only): {weighted_wal_years:.2f} years "
        f"({weighted_wal_years * 12:.0f} months)."
    )
    st.markdown("### Step 3 - Projection Settings")
    default_projection_months = int(st.session_state.get("projection_months_default", 240))
    projection_months = st.number_input(
        "Base projection horizon (months)",
        min_value=12,
        max_value=600,
        step=12,
        value=default_projection_months,
        key="projection_months_input",
    )
    default_max_months = int(
        st.session_state.get("max_projection_months_default", max(360, projection_months))
    )
    max_projection_months = st.number_input(
        "Maximum projection horizon (months)",
        min_value=int(projection_months),
        max_value=720,
        step=12,
        value=max(default_max_months, int(projection_months)),
        key="max_projection_months_input",
    )
    default_materiality = float(st.session_state.get("materiality_threshold_default", 1_000.0))
    materiality_threshold = st.number_input(
        "Materiality threshold for remaining balance",
        min_value=0.0,
        max_value=10_000_000.0,
        step=500.0,
        value=default_materiality,
        format="%.2f",
    )
    st.session_state["projection_months_default"] = int(projection_months)
    st.session_state["max_projection_months_default"] = int(max_projection_months)
    st.session_state["materiality_threshold_default"] = float(materiality_threshold)

    discount_method = st.radio(
        "Discount rate configuration",
        options=["Single rate", "Fetch from FRED", "Manual yield curve"],
        index=1,
        horizontal=True,
        key="discount_method_choice",
    )

    discount_config: Dict[str, object]
    selected_curve: Optional[YieldCurve] = st.session_state.get("selected_discount_curve")
    selected_interpolation = st.session_state.get("discount_interpolation_method", "linear")
    if discount_method == "Single rate":
        default_single_rate = st.session_state.get("single_rate_input", DEFAULT_SINGLE_RATE)
        discount_rate_input = st.number_input(
            "Discount rate (annual decimal, e.g., 0.035 for 3.5%)",
            min_value=0.0,
            max_value=0.20,
            value=default_single_rate,
            step=0.005,
            key="single_rate_input",
        )
        discount_config = {"mode": "single", "rate": discount_rate_input, "interpolation": "linear"}
        flat_tenors = [3, 6, 12, 24, 36, 60, 84, 120]
        flat_rates = [float(discount_rate_input)] * len(flat_tenors)
        selected_curve = YieldCurve(flat_tenors, flat_rates, metadata={"source": "single_rate"})
        selected_interpolation = "linear"
    elif discount_method == "Fetch from FRED":
        fred_api_key_default = os.environ.get("FRED_API_KEY", FRED_API_KEY)
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
        if st.button("Fetch current curve", width='stretch'):
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
        for tenor, default_rate in DEFAULT_MANUAL_TENOR_RATES.items():
            st.session_state.setdefault(f"tenor_{tenor}", f"{default_rate:.4f}")
        tenor_values: Dict[int, float] = {}
        columns = st.columns(2)
        for idx, (tenor, label) in enumerate(TENOR_LABELS):
            with columns[idx % 2]:
                value = st.text_input(label, key=f"tenor_{tenor}")
            if value.strip():
                try:
                    tenor_values[tenor] = decimalize(float(value.strip()))
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
        base_market_path = [0.03] * int(max_projection_months)
        st.caption(
            "No yield curve selected yet  using a flat 3% base market path until a curve is configured."
        )
    st.session_state["base_market_path"] = base_market_path

    if _handle_rate_outliers(selected_curve, discount_method):
        st.session_state.pop("run_results", None)
        return

    scenario_flags, monte_carlo_config = _collect_scenarios(
        int(projection_months),
        selected_curve,
    )


    st.markdown("### Step 4 - Download Package Options")
    st.caption(
        "Execution mode: GitHub Actions (remote worker)"
        if jobs.USING_GITHUB_DRIVER
        else "Execution mode: Local worker (runs inside the Streamlit container)"
    )
    gh_repo_env = os.environ.get("GH_REPO", "")
    gh_token_present = bool(os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"))
    st.caption(
        f"GitHub configuration detected → repo: {'set' if gh_repo_env else 'missing'}, token: {'present' if gh_token_present else 'missing'}"
    )
    if jobs.GITHUB_DRIVER_ERROR:
        st.warning(
            "GitHub Actions driver could not be initialised. Running locally instead. "
            "Details: " + jobs.GITHUB_DRIVER_ERROR
        )
    default_package_opts = st.session_state.get(
        "download_package_options",
        {
            "enabled": True,
            "export_cashflows": False,
            "cashflow_mode": "sample",
            "cashflow_sample_size": 20,
        },
    )
    package_enabled = st.checkbox(
        "Build a downloadable zip (Excel + Word) after each run",
        value=default_package_opts.get("enabled", True),
        key="download_package_enabled",
    )
    export_cashflows_toggle = st.checkbox(
        "Embed cash flow detail in the Excel workbook",
        value=default_package_opts.get("export_cashflows", False),
        key="download_package_cashflows",
        disabled=not package_enabled,
    )
    cashflow_mode_options = ["Sampled (top balances)", "Full detail"]
    default_mode_index = (
        0 if default_package_opts.get("cashflow_mode", "sample") == "sample" else 1
    )
    cashflow_mode_label = st.selectbox(
        "Cash flow detail level",
        cashflow_mode_options,
        index=default_mode_index,
        key="download_package_cashflow_mode",
        disabled=not (package_enabled and export_cashflows_toggle),
    )
    cashflow_mode = "sample" if cashflow_mode_label.startswith("Sampled") else "full"
    cashflow_sample_size = st.number_input(
        "Cash flow sample size (accounts)",
        min_value=5,
        max_value=5000,
        step=5,
        value=int(default_package_opts.get("cashflow_sample_size", 20)),
        key="download_package_cashflow_sample_size",
        disabled=not (
            package_enabled and export_cashflows_toggle and cashflow_mode == "sample"
        ),
    )
    if package_enabled:
        st.caption(
            "A zip download (Excel workbook + Word report + charts) will launch automatically "
            "after each successful run. Save it locally before the session idles."
        )
    else:
        st.caption("Disable the automatic download if you only want to explore results in-app.")
    st.session_state["download_package_options"] = {
        "enabled": package_enabled,
        "export_cashflows": export_cashflows_toggle,
        "cashflow_mode": cashflow_mode,
        "cashflow_sample_size": int(cashflow_sample_size),
    }
    active_job_info = st.session_state.get("active_job")
    run_clicked = st.button("Run Analysis", type="primary", disabled=bool(active_job_info))

    results = st.session_state.get("run_results")
    progress_text_placeholder = st.empty()
    progress_bar_placeholder = st.empty()

    if run_clicked:
        if active_job_info:
            st.warning("An analysis job is already running. Please wait for it to finish before starting another.")
        else:
            package_opts = st.session_state.get("download_package_options", {})
            try:
                payload = _build_analysis_payload(
                    field_map=st.session_state["field_map"],
                    optional_fields=st.session_state.get("optional_fields", []),
                    segmentation=segmentation,
                    assumptions=assumptions,
                    projection_months=int(projection_months),
                    max_projection_months=int(max_projection_months),
                    materiality_threshold=float(materiality_threshold),
                    scenario_flags=scenario_flags,
                    discount_config=discount_config,
                    base_market_path=base_market_path,
                    monte_carlo_config=monte_carlo_config,
                    package_options=package_opts,
                    selected_curve=selected_curve,
                    report_title="Deposit Analysis",
                )
            except Exception as exc:
                st.error(f"Unable to prepare analysis payload: {exc}")
                return

            try:
                handle = jobs.launch_analysis_job(payload, df_raw)
            except Exception as exc:
                LOGGER.exception("Failed to launch analysis job")
                st.session_state["analysis_status_message"] = f"Dispatch failed: {exc}"
                st.error(
                    "Unable to launch the background analysis job. Check GitHub credentials "
                    "and workflow configuration, then try again."
                )
                return
            st.session_state["active_job"] = {
                "job_id": handle.job_id,
                "job_dir": str(handle.job_dir),
                "mode": "github" if jobs.USING_GITHUB_DRIVER else "local",
            }
            st.session_state["run_results"] = None
            st.session_state["latest_bundle"] = None
            st.session_state["analysis_metadata"] = None
            st.session_state["analysis_status_message"] = (
                "Remote job submitted to GitHub Actions."
                if jobs.USING_GITHUB_DRIVER
                else "Job dispatched to local worker process."
            )
            st.success("Analysis job started. Progress updates will appear below.")
            active_job_info = st.session_state["active_job"]

    if active_job_info:
        handle = jobs.AnalysisJobHandle(
            job_id=active_job_info["job_id"],
            job_dir=Path(active_job_info["job_dir"]),
        )
        cancel_requested = st.button(
            "Cancel Running Job",
            type="secondary",
            key="cancel_job_button",
        )
        if cancel_requested:
            try:
                jobs.cancel_job(handle)
                st.session_state["analysis_status_message"] = "Cancellation requested. Waiting for remote worker to halt."
                st.warning("Cancellation requested. Remote workflows may take a few seconds to stop.")
            except Exception as cancel_exc:
                st.error(f"Unable to cancel job: {cancel_exc}")
        status = jobs.read_job_status(handle)
        total_steps = max(status.total, 1)
        progress_pct = min(100, int((status.step / total_steps) * 100))
        progress_bar_placeholder.progress(progress_pct)
        progress_text_placeholder.markdown(f"**{status.message or 'Running analysis...'}**")
        if status.state == "completed":
            try:
                results_obj = jobs.load_job_results(handle)
                st.session_state["run_results"] = results_obj
                extras = status.extras or {}
                st.session_state["analysis_metadata"] = extras.get("analysis_metadata")
                bundle_info = jobs.load_job_bundle(handle)
                if bundle_info:
                    st.session_state["latest_bundle"] = bundle_info
                else:
                    # Build the package locally if remote worker did not provide it
                    package_opts = st.session_state.get("download_package_options", {}) or {}
                    if package_opts.get("enabled"):
                        try:
                            builder = InMemoryReportBuilder(base_title="Deposit Analysis")
                            local_bundle = builder.build(
                                results_obj,
                                discount_config=None,
                                analysis_metadata=st.session_state.get("analysis_metadata") or {},
                                export_cashflows=bool(package_opts.get("export_cashflows", False)),
                                cashflow_mode=str(package_opts.get("cashflow_mode", "sample")),
                                cashflow_sample_size=int(package_opts.get("cashflow_sample_size", 20)),
                                cashflow_random_state=42,
                            )
                            st.session_state["latest_bundle"] = {
                                "zip_bytes": local_bundle.zip_bytes,
                                "zip_name": local_bundle.zip_name,
                                "excel_name": local_bundle.excel_name,
                                "word_name": local_bundle.word_name,
                                "manifest": local_bundle.manifest,
                                "created_at": local_bundle.created_at.isoformat(),
                                "token": local_bundle.created_at.isoformat(),
                            }
                        except Exception as bundle_exc:
                            st.session_state["latest_bundle_error"] = str(bundle_exc)
                jobs.cleanup_job_artifacts(handle)
                st.session_state["analysis_status_message"] = "Analysis complete! Preparing visualisations..."
            except Exception as exc:
                st.error(f"Analysis completed but results could not be loaded: {exc}")
                st.session_state["analysis_status_message"] = f"Error loading results: {exc}"
            finally:
                st.session_state["active_job"] = None
                progress_bar_placeholder.progress(100)
                if st.session_state.get("analysis_status_message"):
                    progress_text_placeholder.markdown(
                        f"**{st.session_state['analysis_status_message']}**"
                    )
        elif status.state == "cancelled":
            st.session_state["active_job"] = None
            st.session_state["analysis_status_message"] = "Analysis cancelled."
            progress_text_placeholder.markdown("**Analysis cancelled**")
            progress_bar_placeholder.progress(0)
            st.info("Analysis run was cancelled. You can start a new run when ready.")
        elif status.state == "failed":
            st.session_state["active_job"] = None
            st.session_state["analysis_status_message"] = "Analysis failed."
            progress_text_placeholder.markdown("**Analysis failed**")
            progress_bar_placeholder.progress(0)
            st.error("Background analysis failed. Review the error details below.")
            if status.error:
                with st.expander("Job error details", expanded=False):
                    st.code(status.error)
        else:
            if st_autorefresh is not None:
                st_autorefresh(interval=1000, key="analysis_auto_refresh")
            else:  # pragma: no cover - legacy fallback
                rerun = getattr(st, "rerun", None)
                if callable(rerun):
                    rerun()
                else:
                    st.markdown(
                        "<script>setTimeout(function(){window.location.reload();}, 1000);</script>",
                        unsafe_allow_html=True,
                    )
    else:
        if st.session_state.get("analysis_status_message"):
            progress_text_placeholder.markdown(f"**{st.session_state['analysis_status_message']}**")
            progress_bar_placeholder.progress(100 if results else 0)
        else:
            progress_text_placeholder.empty()
            progress_bar_placeholder.empty()

    results = st.session_state.get("run_results")
    bundle_info = st.session_state.get("latest_bundle")
    bundle_error = st.session_state.pop("latest_bundle_error", None)
    if bundle_info:
        st.markdown("#### Latest Download Package")
        st.caption(
            "The package includes the Excel workbook, Word narrative, and supporting charts. "
            "Save the downloaded zip locally to retain the run."
        )
        token = bundle_info.get("token")
        zip_bytes = bundle_info.get("zip_bytes")
        zip_name = bundle_info.get("zip_name")
        if token and zip_bytes and zip_name:
            if st.session_state.get("auto_download_last_token") != token:
                _trigger_auto_download(zip_bytes, zip_name, token)
                st.session_state["auto_download_last_token"] = token
            st.download_button(
                "Download again",
                data=zip_bytes,
                file_name=zip_name,
                mime="application/zip",
                key=f"download_bundle_{token}",
            )
        manifest = bundle_info.get("manifest", {})
        if manifest:
            with st.expander("Package contents", expanded=False):
                for path_str, description in manifest.items():
                    st.write(f"**{path_str}** - {description}")
    elif bundle_error:
        st.warning(f"Automatic download encountered an issue: {bundle_error}")

    if results is None:
        st.info("Configure inputs and click **Run Analysis** to generate results.")
        return
    if not run_clicked:
        st.caption(
            "Displaying previously generated visualisations. Run the analysis again to refresh."
        )

    _render_parameter_summary(results)

    scenario_results = results.scenario_results
    if not scenario_results:
        st.info("No scenarios were produced for the selected configuration.")
        return

    parallel_ids: List[str] = []
    curve_ids: List[str] = []
    monte_carlo_ids: List[str] = []
    for scenario_id, scenario_result in scenario_results.items():
        method = (scenario_result.metadata or {}).get("method")
        if method == "parallel":
            parallel_ids.append(scenario_id)
        elif method == "monte_carlo":
            monte_carlo_ids.append(scenario_id)
        elif method not in {"base", None}:
            curve_ids.append(scenario_id)

    st.markdown("### Visualisation Explorer")

    group_options: List[str] = []
    if parallel_ids:
        group_options.append("Parallel Shocks")
    if curve_ids:
        group_options.append("Curve Shape Shocks")
    if monte_carlo_ids:
        group_options.append("Monte Carlo")

    if not group_options:
        st.info("No visualisations are available for the selected configuration.")
        return

    group_choice = st.radio(
        "Select visualisation set",
        group_options,
        index=0,
        key="visualisation_set",
    )

    if group_choice == "Parallel Shocks":
        _render_shock_group(
            results,
            parallel_ids,
            title="Parallel Shock Present Value Comparison",
            select_label="Select parallel scenario",
            select_key="parallel_shock_select",
        )
    elif group_choice == "Curve Shape Shocks":
        _render_shock_group(
            results,
            curve_ids,
            title="Curve-Shape Shock Present Value Comparison",
            select_label="Select curve scenario",
            select_key="curve_shock_select",
        )
    else:
        _render_monte_carlo_group(results, monte_carlo_ids)

    if progress_bar:
        progress_bar.progress(100)
    if progress_text:
        progress_text.empty()

if __name__ == "__main__":
    main()


















"""Typer-based command line interface for manual input workflows."""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import ALMEngine
from src.core.discount import DiscountCurve
from src.core.monte_carlo import MonteCarloConfig, MonteCarloLevel, VasicekParams
from src.core.decay import resolve_decay_parameters
from src.config import FRED_API_KEY
from src.reporting import ReportGenerator
from src.models.results import EngineResults

app = typer.Typer(help="Non-maturity deposit ALM cash flow projection engine")
console = Console()

REQUIRED_FIELDS = {
    "account_id": "Unique account identifier",
    "balance": "Current balance column",
    "interest_rate": "Current interest rate column (decimal)",
}
OPTIONAL_FIELDS = {
    "account_type": "Account type (for segmentation)",
    "customer_segment": "Customer segment (for segmentation)",
    "rate_type": "Fixed/Variable/Tiered (optional metadata)",
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


def _load_preview(file_path: Path, nrows: int = 5) -> pd.DataFrame:
    """Load a small preview sample from the CSV."""
    return pd.read_csv(file_path, nrows=nrows)


def _display_preview(df: pd.DataFrame) -> None:
    table = Table(title="Data Preview", show_lines=False)
    for column in df.columns:
        table.add_column(str(column))
    for _, row in df.iterrows():
        table.add_row(*[str(value) for value in row])
    console.print(table)


def _resolve_column(defaults: Dict[str, str], field: str, columns: Iterable[str]) -> str:
    """Prompt the user to select a column for a required field."""
    columns = list(columns)
    default = defaults.get(field)
    while True:
        prompt = f"Select column for '{field}'"
        if default and default in columns:
            choice = typer.prompt(prompt, default=default)
        else:
            choice = typer.prompt(prompt)
        if choice in columns:
            return choice
        console.print(f"[red]Invalid column. Please choose from: {', '.join(columns)}[/red]")


def _infer_defaults(columns: Iterable[str]) -> Dict[str, str]:
    """Suggest default mappings based on column names."""
    columns_lower = {col.lower(): col for col in columns}
    defaults: Dict[str, str] = {}
    for canonical, _ in REQUIRED_FIELDS.items():
        for candidate in columns_lower:
            if candidate == canonical or candidate.replace("_", "") == canonical.replace("_", ""):
                defaults[canonical] = columns_lower[candidate]
                break
    for optional in OPTIONAL_FIELDS:
        for candidate in columns_lower:
            if candidate == optional or candidate.replace("_", "") == optional.replace("_", ""):
                defaults[optional] = columns_lower[candidate]
                break
    return defaults


def _as_decimal(value: str, allow_empty: bool = False) -> Optional[float]:
    """Convert user input to decimal, accepting either decimal or percent inputs."""
    text = value.strip()
    if allow_empty and not text:
        return None
    numeric = float(text)
    if numeric > 1.5:  # assume percentage input
        numeric /= 100.0
    return numeric


def _format_currency(value: Optional[float]) -> str:
    """Format numbers as currency for console output."""
    return f"${value:,.2f}" if value is not None else "N/A"


def _select_sample_months(actual_months: Optional[int]) -> List[int]:
    """Return a set of representative months for sample output."""
    if not actual_months:
        return [1]
    canonical = [1, 12, 24, 60, 120, 180, 240]
    months = [month for month in canonical if month <= actual_months]
    if actual_months not in months:
        months.append(actual_months)
    return sorted(set(months))


def _build_sample_table(
    cashflows: pd.DataFrame, actual_months: Optional[int]
) -> Optional[Table]:
    """Create a Rich table summarising portfolio cash flows for selected months."""
    if cashflows.empty:
        return None
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
        return None
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
    table = Table(title="Sample Cash Flows", show_lines=False)
    for column in [
        "Month",
        "Begin Balance",
        "Decay %",
        "Runoff",
        "End Balance",
        "Int Rate",
        "Interest",
        "Total Outflow",
    ]:
        justify = "right" if column != "Decay %" else "center"
        table.add_column(column, justify=justify)
    for _, row in subset.sort_values("Month").iterrows():
        table.add_row(
            f"{int(row['Month'])}",
            f"{row['Begin Balance']:,.2f}",
            f"{row['Decay %']:.3f}%",
            f"{row['Runoff']:,.2f}",
            f"{row['End Balance']:,.2f}",
            f"{row['Int Rate']:.3f}%",
            f"{row['Interest']:,.2f}",
            f"{row['Total Outflow']:,.2f}",
        )
    return table


def _print_parameter_summary(results: EngineResults) -> None:
    """Print high-level parameter summary to the console."""
    summary = results.parameter_summary or {}
    if not summary:
        console.print("[yellow]Parameter summary unavailable.[/yellow]")
        return

    portfolio = summary.get("portfolio", {})
    projection = summary.get("projection", {})
    segments = summary.get("segments", [])

    console.print("\n[bold]NMD Parameter Summary[/bold]")
    console.print(
        f"Starting Balance: {_format_currency(portfolio.get('starting_balance'))}\n"
        f"Account Count: {portfolio.get('account_count', 0)}\n"
        f"Portfolio Weighted WAL: {portfolio.get('weighted_average_wal_years') or 0:.2f} years"
    )
    console.print(
        f"Base Horizon: {projection.get('projection_months')} months\n"
        f"Max Horizon: {projection.get('max_projection_months')} months\n"
        f"Materiality Threshold: {_format_currency(projection.get('materiality_threshold'))}\n"
        f"Actual Projection Length: {projection.get('actual_projection_months') or 0} months\n"
        f"Residual Balance: {_format_currency(projection.get('residual_balance'))}"
    )

    for segment in segments:
        console.print(f"\n[bold]{segment.get('segment')}[/bold]")
        wal_input = segment.get("input_wal_years")
        decay_input = segment.get("input_decay_rate")
        resolved_wal = segment.get("resolved_wal_years")
        resolved_decay = segment.get("resolved_decay_rate")
        monthly_decay = segment.get("monthly_decay_rate")
        detail_lines = []
        if wal_input is not None:
            detail_lines.append(f"Input WAL: {wal_input:.2f} years")
        if decay_input is not None:
            detail_lines.append(f"Input Decay Rate: {decay_input * 100:.2f}%")
        detail_lines.append(f"Calculated WAL: {resolved_wal:.2f} years")
        detail_lines.append(f"Calculated Annual Decay: {resolved_decay * 100:.2f}%")
        detail_lines.append(f"Calculated Monthly Decay: {monthly_decay * 100:.2f}%")
        detail_lines.append(
            f"Deposit Betas ↑ {segment.get('deposit_beta_up'):.2f} | ↓ {segment.get('deposit_beta_down'):.2f}"
        )
        for line in detail_lines:
            console.print(f"  - {line}")
        warning = segment.get("decay_warning")
        if warning:
            console.print(f"[yellow]{warning}[/yellow]")

    base_id = results.base_scenario_id
    if base_id and base_id in results.scenario_results:
        table = _build_sample_table(
            results.scenario_results[base_id].cashflows,
            projection.get("actual_projection_months"),
        )
        if table is not None:
            console.print()
            console.print(table)


def _prompt_assumptions(segment: str) -> Dict[str, float]:
    """Prompt user for the core assumption inputs."""
    key = segment.lower()
    suggestions = next(
        (values for pattern, values in DEFAULT_ASSUMPTIONS.items() if pattern in key),
        DEFAULT_ASSUMPTIONS.get("checking"),
    )
    console.print(f"\n[bold]Assumptions for segment: {segment}[/bold]")
    decay_input = _as_decimal(
        typer.prompt(
            "Decay / Runoff Rate (annual decimal) (leave blank to derive from WAL)",
            default=f"{suggestions['decay_rate']:.2f}",
        ),
        allow_empty=True,
    )
    wal_text = typer.prompt(
        "Weighted Average Life (years) (leave blank to derive from decay)",
        default=f"{suggestions['wal_years']:.2f}",
    ).strip()
    wal_input = float(wal_text) if wal_text else None
    while wal_input is None and decay_input is None:
        console.print(
            "[red]At least one of WAL or decay rate must be provided. Please enter a value.[/red]"
        )
        if decay_input is None:
            decay_input = _as_decimal(
                typer.prompt(
                    "Decay / Runoff Rate (annual decimal)",
                    default=f"{suggestions['decay_rate']:.2f}",
                ),
                allow_empty=True,
            )
        if wal_input is None:
            wal_text = typer.prompt(
                "Weighted Average Life (years)",
                default=f"{suggestions['wal_years']:.2f}",
            ).strip()
            wal_input = float(wal_text) if wal_text else None

    deposit_beta_up = _as_decimal(
        typer.prompt(
            "Deposit Beta (rising rates)",
            default=f"{suggestions['deposit_beta_up']:.2f}",
        )
    )
    deposit_beta_down = _as_decimal(
        typer.prompt(
            "Deposit Beta (falling rates)",
            default=f"{suggestions['deposit_beta_down']:.2f}",
        )
    )
    repricing_beta_up = _as_decimal(
        typer.prompt(
            "Repricing Beta (rising rates)",
            default=f"{suggestions['repricing_beta_up']:.2f}",
        )
    )
    repricing_beta_down = _as_decimal(
        typer.prompt(
            "Repricing Beta (falling rates)",
            default=f"{suggestions['repricing_beta_down']:.2f}",
        )
    )
    resolution = resolve_decay_parameters(wal_input, decay_input, priority="auto")
    decay_priority = resolution.priority_used
    if resolution.inconsistent and resolution.warning:
        console.print(
            f"[yellow]{resolution.warning} Difference "
            f"{resolution.difference_years:.2f} years "
            f"({(resolution.difference_ratio or 0.0) * 100:.1f}%).[/yellow]"
        )
        while True:
            choice = (
                typer.prompt(
                    "Choose precedence ('wal' to override decay rate, 'decay' to override WAL)",
                    default=resolution.priority_used,
                )
                .strip()
                .lower()
            )
            if choice in {"wal", "decay"}:
                decay_priority = choice
                break
            console.print("[red]Please enter 'wal' or 'decay'.[/red]")
    summary_lines = []
    if wal_input is not None:
        summary_lines.append(f"Input WAL: {wal_input:.2f} years")
    if decay_input is not None:
        summary_lines.append(f"Input decay: {decay_input * 100:.2f}%")
    summary_lines.extend(
        [
            f"Calculated annual decay: {resolution.annual_decay_rate * 100:.2f}%",
            f"Calculated WAL: {resolution.wal_years:.2f} years",
            f"Monthly decay rate: {resolution.monthly_decay_rate * 100:.2f}%",
        ]
    )
    console.print("[cyan]" + " | ".join(summary_lines) + "[/cyan]")
    return {
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


def _collect_segments(df: pd.DataFrame, segmentation: str) -> List[str]:
    """Determine which segments require assumption entry."""
    if segmentation == "all":
        return ["ALL"]
    column = "account_type" if segmentation == "by_account_type" else "customer_segment"
    if column not in df.columns:
        raise typer.BadParameter(
            f"Segmentation '{segmentation}' requires the column '{column}' to be mapped."
        )
    segments = (
        df[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    segments.sort()
    return segments



def _prompt_monte_carlo_config(
    projection_months: int,
    base_curve: Optional["DiscountCurve"],
) -> Optional[MonteCarloConfig]:
    """Collect Monte Carlo configuration settings from the user."""
    if not typer.confirm("Run Monte Carlo simulation?", default=False):
        return None

    console.print(
        "\n[bold]Monte Carlo Overview[/bold]\n"
        "• [green]Level 1 – Static curve[/green]: keeps today’s discount curve fixed and only simulates the short-rate path used for deposit repricing.\n"
        "• [green]Level 2 – Two-factor curve[/green]: simulates correlated 3M and 10Y anchors, rebuilds the curve each month, and discounts using that evolved curve."
    )

    short_anchor = float(base_curve.get_rate(3)) if base_curve else 0.03
    long_anchor = float(base_curve.get_rate(120)) if base_curve else short_anchor
    console.print(
        f"\nBase curve anchors from Treasury data:\n"
        f"  • 3M anchor: {short_anchor * 100:.2f}%\n"
        f"  • 10Y anchor: {long_anchor * 100:.2f}%"
    )
    console.print(
        "Mean reversion (a) controls how quickly simulated rates revert toward the anchor.\n"
        "Long-term mean (b) sets that anchor level. Volatility (sigma) drives monthly randomness."
    )

    level_input = typer.prompt(
        "Implementation level (1 = Static curve, 2 = Two-factor)",
        default="1",
    ).strip()
    try:
        level = MonteCarloLevel(int(level_input))
    except (ValueError, KeyError):
        raise typer.BadParameter("Implementation level must be 1 or 2.")

    num_sim = int(typer.prompt("Number of simulations", default="1000"))
    if not (100 <= num_sim <= 10000):
        raise typer.BadParameter("Number of simulations must be between 100 and 10,000.")

    seed_text = typer.prompt("Random seed (leave blank for random)", default="").strip()
    random_seed = int(seed_text) if seed_text else None

    def _prompt_vasicek(prefix: str, defaults: VasicekParams) -> VasicekParams:
        mean_reversion = float(
            typer.prompt(
                f"{prefix} mean reversion speed (a)",
                default=f"{defaults.mean_reversion:.2f}",
            )
        )
        long_term_mean = _as_decimal(
            typer.prompt(
                f"{prefix} long-term mean (b)",
                default=f"{defaults.long_term_mean:.3f}",
            )
        )
        volatility = float(
            typer.prompt(
                f"{prefix} volatility (sigma)",
                default=f"{defaults.volatility:.3f}",
            )
        )
        return VasicekParams(
            mean_reversion=mean_reversion,
            long_term_mean=long_term_mean,
            volatility=volatility,
        )

    short_defaults = VasicekParams(mean_reversion=0.15, long_term_mean=short_anchor, volatility=0.01)
    short_params = _prompt_vasicek("Short rate", short_defaults)

    long_params = None
    correlation = 0.70
    if level == MonteCarloLevel.TWO_FACTOR:
        long_defaults = VasicekParams(mean_reversion=0.07, long_term_mean=long_anchor, volatility=0.008)
        long_params = _prompt_vasicek("Long rate", long_defaults)
        correlation = float(typer.prompt("Correlation between short and long rates (rho)", default="0.70"))
        correlation = max(min(correlation, 0.99), -0.99)

    save_paths = typer.confirm("Save rate path samples?", default=True)
    sample_size = 0
    if save_paths:
        sample_size = int(typer.prompt("Rate path sample size", default="100"))

    generate_reports = typer.confirm("Generate detailed Monte Carlo reports?", default=True)
    interpolation_method = (
        typer.prompt("Yield curve interpolation method", default="linear").strip().lower()
        or "linear"
    )

    return MonteCarloConfig(
        level=level,
        num_simulations=num_sim,
        projection_months=projection_months,
        random_seed=random_seed,
        short_rate=short_params,
        long_rate=long_params,
        correlation=correlation,
        save_rate_paths=save_paths,
        generate_reports=generate_reports,
        sample_size=max(sample_size, 0),
        interpolation_method=interpolation_method,
    )


def _prompt_scenario_selection(
    projection_months: int,
    base_curve: Optional["DiscountCurve"],
) -> tuple[Dict[str, bool], Optional[MonteCarloConfig]]:
    """Ask user which standard scenarios to run."""
    console.print("\n[bold]Scenario Selection[/bold]")
    scenarios = {
        "base": True,
        "parallel_100": typer.confirm("+100 bps parallel shock?", default=True),
        "parallel_200": typer.confirm("+200 bps parallel shock?", default=True),
        "parallel_300": typer.confirm("+300 bps parallel shock?", default=False),
        "parallel_400": typer.confirm("+400 bps parallel shock?", default=False),
        "parallel_minus_100": typer.confirm("-100 bps parallel shock?", default=True),
        "parallel_minus_200": typer.confirm("-200 bps parallel shock?", default=False),
        "parallel_minus_300": typer.confirm("-300 bps parallel shock?", default=False),
        "parallel_minus_400": typer.confirm("-400 bps parallel shock?", default=False),
        "steepener": typer.confirm("Curve steepener (short down, long up)?", default=False),
        "flattener": typer.confirm("Curve flattener (short up, long down)?", default=False),
        "short_shock_up": typer.confirm("Short-rate shock up (+200 bps front-end)?", default=False),
    }


    mc_config = _prompt_monte_carlo_config(projection_months, base_curve)
    if mc_config is not None:
        scenarios["monte_carlo"] = True

    return scenarios, mc_config




@app.command()
def run(
    file_path: Path = typer.Argument(..., help="CSV file containing account level data"),
    segmentation: str = typer.Option(
        "all",
        case_sensitive=False,
        help="Segmentation option: all | by_account_type | by_customer_segment",
    ),
    projection_months: int = typer.Option(240, help="Base projection horizon in months"),
    max_projection_months: int = typer.Option(600, help="Maximum projection horizon in months"),
    materiality_threshold: float = typer.Option(
        1000.0, help="Materiality threshold for residual balance ($)"
    ),
    output_dir: Path = typer.Option(Path("output"), help="Directory for report exports"),
    cashflow_sample_size: int = typer.Option(
        20, help="Number of accounts to include when exporting detailed cashflows (0 to export all)."
    ),
    generate_plots: bool = typer.Option(
        False,
        help="Generate Monte Carlo visualisation PNGs when a Monte Carlo scenario is present.",
    ),
) -> None:
    """Run the ALM engine with manual input prompts."""
    if not file_path.exists():
        raise typer.BadParameter(f"File not found: {file_path}")
    if max_projection_months < projection_months:
        raise typer.BadParameter("max_projection_months must be >= projection_months")



    preview = _load_preview(file_path)
    console.print(f"\nLoaded file with {preview.shape[1]} columns. Preview below:")
    _display_preview(preview)

    defaults = _infer_defaults(preview.columns)
    console.print("\n[bold]Field Mapping[/bold]")
    mapped_fields: Dict[str, str] = {}
    for field in REQUIRED_FIELDS:
        mapped_fields[field] = _resolve_column(defaults, field, preview.columns)

    optional_fields: List[str] = []
    for field, description in OPTIONAL_FIELDS.items():
        if typer.confirm(f"Map optional field '{field}'? ({description})", default=False):
            column = _resolve_column(defaults, field, preview.columns)
            mapped_fields[field] = column
            optional_fields.append(field)

    engine = ALMEngine()
    load_result = engine.load_data(
        file_path=str(file_path),
        field_map=mapped_fields,
        optional_fields=optional_fields,
    )

    segmentation = segmentation.lower()
    engine.set_segmentation(segmentation)
    segments = _collect_segments(load_result.dataframe, segmentation)

    console.print("\n[bold]Assumption Entry[/bold]")
    if segments == ["ALL"]:
        values = _prompt_assumptions("Portfolio")
        engine.set_assumptions(
            "ALL",
            decay_rate=values["decay_rate"],
            wal_years=values["wal_years"],
            deposit_beta_up=values["deposit_beta_up"],
            deposit_beta_down=values["deposit_beta_down"],
            repricing_beta_up=values["repricing_beta_up"],
            repricing_beta_down=values["repricing_beta_down"],
            decay_priority=values.get("decay_priority", "auto"),
        )
    else:
        for segment in segments:
            values = _prompt_assumptions(segment)
            engine.set_assumptions(
                segment,
                decay_rate=values["decay_rate"],
                wal_years=values["wal_years"],
                deposit_beta_up=values["deposit_beta_up"],
                deposit_beta_down=values["deposit_beta_down"],
                repricing_beta_up=values["repricing_beta_up"],
                repricing_beta_down=values["repricing_beta_down"],
                decay_priority=values.get("decay_priority", "auto"),
            )

    console.print("\n[bold]Discount Rate Configuration[/bold]")
    console.print(
        "1) Single discount rate (flat)\n"
        "2) Fetch current Treasury curve from FRED (recommended)\n"
        "3) Manually enter yield curve tenors"
    )
    discount_choice = (
        typer.prompt("Select option [1/2/3]", default="2").strip().lower()
    )

    tenor_labels = [
        (3, "3 Month"),
        (6, "6 Month"),
        (12, "1 Year"),
        (24, "2 Year"),
        (36, "3 Year"),
        (60, "5 Year"),
        (84, "7 Year"),
        (120, "10 Year"),
    ]

    target_date: Optional[str] = None
    if discount_choice == "1":
        rate = _as_decimal(typer.prompt("Discount rate (annual decimal)", default="0.035"))
        engine.set_discount_single_rate(rate)
        console.print(f"Using flat discount rate of {rate * 100:.2f}%")
    elif discount_choice == "2":
        api_key = os.environ.get("FRED_API_KEY") or FRED_API_KEY
        if not api_key:
            raise typer.BadParameter(
                "FRED API key is required to fetch the curve (set FRED_API_KEY env var or update src/config.py)."
            )
        console.print("[green]Using stored FRED API key.[/green]")
        interpolation_method = (
            typer.prompt("Interpolation method (linear/log-linear/cubic)", default="linear")
            .strip()
            .lower()
            or "linear"
        )
        if typer.confirm("Fetch curve for a specific valuation date?", default=False):
            target_date = typer.prompt("Enter valuation date (YYYY-MM-DD)", default="").strip() or None
        curve = engine.set_discount_curve_from_fred(
            api_key,
            interpolation_method=interpolation_method,
            target_date=target_date,
        )
        as_of = curve.metadata.get("as_of", "latest")
        console.print(f"[green]FRED curve loaded (as of {as_of}).[/green]")
        display_rows = [
            f"{int(tenor):>3}M: {rate * 100:.2f}%"
            for tenor, rate in zip(curve.tenors, curve.rates)
        ]
        console.print("  " + "  |  ".join(display_rows))
    elif discount_choice == "3":
        console.print("Enter annualised rates for each tenor (leave blank to skip).")
        tenor_map: Dict[int, float] = {}
        for tenor, label in tenor_labels:
            value = typer.prompt(f"{label}", default="")
            if value:
                tenor_map[tenor] = _as_decimal(value)
        if not tenor_map:
            raise typer.BadParameter("At least one tenor rate is required for manual yield curve mode.")
        interpolation_method = (
            typer.prompt("Interpolation method (linear/log-linear/cubic)", default="linear")
            .strip()
            .lower()
            or "linear"
        )
        engine.set_discount_yield_curve(
            tenor_map,
            interpolation_method=interpolation_method,
            source="manual",
        )
        console.print(
            "Manual yield curve registered: "
            + "  |  ".join(f"{tenor:>3}M {rate * 100:.2f}%" for tenor, rate in sorted(tenor_map.items()))
        )
    else:
        raise typer.BadParameter("Invalid discount configuration selection. Choose 1, 2, or 3.")

    discount_curve = engine.discount_curve()
    base_path = [
        float(discount_curve.get_rate(month))
        for month in range(1, max_projection_months + 1)
    ]
    engine.set_base_market_rate_path(base_path)
    console.print(
        f"\nBase market rate path derived from the yield curve (month 1: {base_path[0] * 100:.2f}%)."
    )

    scenario_flags, monte_carlo_config = _prompt_scenario_selection(
        projection_months,
        discount_curve,
    )
    if monte_carlo_config:
        engine.set_monte_carlo_config(monte_carlo_config)
    engine.configure_standard_scenarios(scenario_flags)

    console.print("\n[bold]Running projections...[/bold]")
    results = engine.run_analysis(
        projection_months=projection_months,
        max_projection_months=max_projection_months,
        materiality_threshold=materiality_threshold,
    )


    reporter = ReportGenerator(output_dir)
    summary_path = reporter.export_summary(results)
    base_cashflow_path = reporter.export_cashflows(
        results, "base", sample_size=max(0, cashflow_sample_size)
    )
    discount_path = reporter.export_discount_configuration(engine.discount_configuration())
    monte_carlo_exports: Dict[str, Path] = {}
    if "monte_carlo" in results.scenario_results:
        monte_carlo_exports = reporter.export_monte_carlo_tables(results)
    plot_paths = {}
    if generate_plots:
        plot_paths = reporter.export_monte_carlo_visuals(results)

    console.print("\n[bold green]Analysis complete![/bold green]")
    console.print(f"Summary exported to: {summary_path}")
    console.print(f"Base cash flows exported to: {base_cashflow_path}")
    console.print(f"Discount configuration exported to: {discount_path}")
    if monte_carlo_exports:
        console.print("Monte Carlo artefacts:")
        for name, path in monte_carlo_exports.items():
            console.print(f"  - {name}: {path}")
    if plot_paths:
        console.print("Generated Monte Carlo charts:")
        for name, path in plot_paths.items():
            console.print(f"  - {name}: {path}")
    console.print("Additional scenarios can be exported using the reporting module.")


def main() -> None:
    """Entry point for CLI execution."""
    app()


if __name__ == "__main__":
    main()

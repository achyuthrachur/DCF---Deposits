"""Typer-based command line interface for manual input workflows."""

from __future__ import annotations

import sys
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
from src.reporting import ReportGenerator

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
    "checking": {"decay_rate": 0.05, "wal_years": 5.0, "beta_up": 0.40, "beta_down": 0.25},
    "savings": {"decay_rate": 0.08, "wal_years": 3.5, "beta_up": 0.55, "beta_down": 0.35},
    "money market": {"decay_rate": 0.20, "wal_years": 1.5, "beta_up": 0.75, "beta_down": 0.60},
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


def _as_decimal(value: str) -> float:
    """Convert user input to decimal, accepting either decimal or percent inputs."""
    numeric = float(value)
    if numeric > 1.5:  # assume percentage input
        return numeric / 100
    return numeric


def _prompt_assumptions(segment: str) -> Dict[str, float]:
    """Prompt user for the four core assumption inputs."""
    key = segment.lower()
    suggestions = next(
        (values for pattern, values in DEFAULT_ASSUMPTIONS.items() if pattern in key),
        DEFAULT_ASSUMPTIONS.get("checking"),
    )
    console.print(f"\n[bold]Assumptions for segment: {segment}[/bold]")
    decay = _as_decimal(
        typer.prompt(
            "Decay / Runoff Rate (annual decimal)",
            default=f"{suggestions['decay_rate']:.2f}",
        )
    )
    wal = float(
        typer.prompt(
            "Weighted Average Life (years)",
            default=f"{suggestions['wal_years']:.2f}",
        )
    )
    beta_up = _as_decimal(
        typer.prompt(
            "Deposit Beta (rising rates)",
            default=f"{suggestions['beta_up']:.2f}",
        )
    )
    beta_down = _as_decimal(
        typer.prompt(
            "Repricing Beta (falling rates)",
            default=f"{suggestions['beta_down']:.2f}",
        )
    )
    return {
        "decay_rate": decay,
        "wal_years": wal,
        "beta_up": beta_up,
        "beta_down": beta_down,
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


def _prompt_scenario_selection() -> tuple[Dict[str, bool], Optional[Dict[str, float]]]:
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
    }
    monte_carlo_config: Optional[Dict[str, float]] = None
    if typer.confirm("Run Monte Carlo simulation?", default=False):
        scenarios["monte_carlo"] = True
        num_sim = int(
            typer.prompt("Number of simulations", default="1000")
        )
        vol_bps = float(
            typer.prompt("Quarterly volatility (bps)", default="25")
        )
        drift_bps = float(
            typer.prompt("Quarterly drift (bps)", default="0")
        )
        symmetric = typer.confirm("Use symmetric shocks (pair +Z/-Z)?", default=False)
        opposite_drift = typer.confirm("Pair opposite drift signs?", default=False)
        fixed_mag = typer.confirm("Use fixed ±vol shocks (Rademacher)?", default=False)
        seed_value = typer.prompt("Random seed (leave blank for random)", default="")
        monte_carlo_config = {
            "num_simulations": num_sim,
            "quarterly_volatility": vol_bps / 10000,
            "quarterly_drift": drift_bps / 10000,
            "symmetric_shocks": symmetric,
            "pair_opposite_drift": opposite_drift,
            "use_fixed_magnitude": fixed_mag,
        }
        if seed_value.strip():
            monte_carlo_config["random_seed"] = int(seed_value.strip())
    return scenarios, monte_carlo_config


@app.command()
def run(
    file_path: Path = typer.Argument(..., help="CSV file containing account level data"),
    segmentation: str = typer.Option(
        "all",
        case_sensitive=False,
        help="Segmentation option: all | by_account_type | by_customer_segment",
    ),
    projection_months: int = typer.Option(120, help="Projection horizon in months"),
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
        engine.set_assumptions("ALL", **values)
    else:
        for segment in segments:
            values = _prompt_assumptions(segment)
            engine.set_assumptions(segment, **values)

    console.print("\n[bold]Discount Rate Configuration[/bold]")
    if typer.confirm("Use a single discount rate for all periods?", default=True):
        rate = _as_decimal(typer.prompt("Discount rate (annual decimal)", default="0.035"))
        engine.set_discount_single_rate(rate)
    else:
        console.print("Enter rates for tenors in months (leave blank to skip):")
        tenor_map: Dict[int, float] = {}
        for tenor, label in [
            (3, "3 Months"),
            (6, "6 Months"),
            (12, "1 Year"),
            (24, "2 Years"),
            (36, "3 Years"),
            (60, "5 Years"),
            (84, "7 Years"),
            (120, "10 Years"),
        ]:
            value = typer.prompt(label, default="")
            if value:
                tenor_map[tenor] = _as_decimal(value)
        if not tenor_map:
            raise typer.BadParameter("At least one tenor rate is required for yield curve mode.")
        engine.set_discount_yield_curve(tenor_map)

    console.print("\n[bold]Base Market Rate[/bold]")
    base_rate = _as_decimal(
        typer.prompt("Enter base market rate (annual decimal)", default="0.03")
    )
    engine.set_base_market_rate_path(base_rate)

    scenario_flags, monte_carlo_config = _prompt_scenario_selection()
    if monte_carlo_config:
        engine.set_monte_carlo_config(
            num_simulations=int(monte_carlo_config["num_simulations"]),
            quarterly_volatility=float(monte_carlo_config["quarterly_volatility"]),
            quarterly_drift=float(monte_carlo_config["quarterly_drift"]),
            random_seed=int(monte_carlo_config["random_seed"])
            if "random_seed" in monte_carlo_config
            else None,
            symmetric_shocks=bool(monte_carlo_config.get("symmetric_shocks", False)),
            pair_opposite_drift=bool(monte_carlo_config.get("pair_opposite_drift", False)),
            use_fixed_magnitude=bool(monte_carlo_config.get("use_fixed_magnitude", False)),
        )
    engine.configure_standard_scenarios(scenario_flags)

    console.print("\n[bold]Running projections...[/bold]")
    results = engine.run_analysis(projection_months=projection_months)

    reporter = ReportGenerator(output_dir)
    summary_path = reporter.export_summary(results)
    base_cashflow_path = reporter.export_cashflows(
        results, "base", sample_size=max(0, cashflow_sample_size)
    )
    plot_paths = {}
    if generate_plots:
        plot_paths = reporter.export_monte_carlo_visuals(results)

    console.print("\n[bold green]Analysis complete![/bold green]")
    console.print(f"Summary exported to: {summary_path}")
    console.print(f"Base cash flows exported to: {base_cashflow_path}")
    if plot_paths:
        console.print("Generated Monte Carlo charts:")
        for name, path in plot_paths.items():
            console.print(f"  • {name}: {path}")
    console.print("Additional scenarios can be exported using the reporting module.")


def main() -> None:
    """Entry point for CLI execution."""
    app()


if __name__ == "__main__":
    main()

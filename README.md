# Non-Maturity Deposit (NMD) ALM Engine

This project provides a Python-based asset-liability management engine for modelling
non-maturity deposit portfolios. The engine focuses on manual assumption entry and
flexible field mapping so it can accommodate institution-specific data layouts.

## Key Capabilities
- CSV/Excel ingestion with interactive field mapping prompts.
- Support for multiple segmentation approaches (all accounts, by account type, by customer segment).
- Manual capture of the core assumptions (decay, WAL, deposit betas up/down, repricing betas up/down).
- Yield curve discounting with linear, log-linear, or cubic interpolation.
- One-click Treasury curve downloads from the FRED API or manual tenor entry.
- Scenario generation for parallel/non-parallel curve shocks plus Monte Carlo simulations with configurable volatility and drift.
- Account-level cash flow projection, terminal value capture, and present value calculation.
- CSV/JSON report exports (scenario summary, discount curve snapshot, cash flow detail, account-level PV) with optional Monte Carlo visualisations (rate spaghetti/fan charts, PV distributions, dashboards).

## Getting Started
1. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
2. Run the interactive CLI:
   ```bash
   python -m src.ui.cli run path/to/data.csv
   ```
3. Launch the Streamlit web app:
   ```bash
   streamlit run src/ui/web_app.py
   ```

The CLI will guide you through:
1. Reviewing a preview of the uploaded file.
2. Mapping source columns to required fields.
3. Selecting a segmentation method and entering assumptions.
4. Configuring discount rates and market scenarios.

After execution the CLI exports reports to the `output/` directory.  
The web app renders results in-browser and provides download buttons for summary,
cashflow, and account-level PV CSVs.

### Yield Curve Sources
- **Single rate** – retains backwards compatibility for quick what-if analysis.
- **FRED API** – supply a FRED API key (`FRED_API_KEY` environment variable or UI input) to pull the latest Treasury curve. Snapshots are saved to `output/discount_curve.json`.
- **Manual curve** – enter tenor points (3M–10Y) directly in the CLI or web app.

### Supported Scenarios
- Deterministic parallel shocks: rising and falling rate moves from +/-100 bps through +/-400 bps.
- Non-parallel shocks: steepener, flattener, and short-rate focused profiles derived from the base curve.
- Monte Carlo simulations: configurable number of simulations, monthly volatility/drift inputs, and downloadable PV distributions.

## Project Structure
```
src/
├── core/          # Calculation engines (cash flows, PV, scenarios)
├── models/        # Pydantic models for accounts, assumptions, scenarios, results
├── reporting/     # CSV export helpers
└── ui/            # Typer CLI for manual data entry
```

## Sharing with Collaborators
To publish a shareable web link, deploy the Streamlit app:

1. Push this project to a GitHub repository (public or private).
2. Sign in to [Streamlit Community Cloud](https://streamlit.io/cloud) (free tier available).
3. Create a new app, selecting your repository and the `src/ui/web_app.py` entry point.
4. Streamlit Cloud automatically installs dependencies from `requirements.txt` and launches
   the app. Copy the generated URL and share it with collaborators—they can access the tool
   directly from their browser without installing anything locally.

Alternative hosting options include Streamlit on AWS/GCP/Azure, containerising with Docker
and serving via services like Azure Container Apps, or embedding the engine behind a FastAPI
service with a custom frontend. Streamlit Cloud offers the fastest path to a working link.

## Next Steps
- Extend segmentation to support cross-segmentation (account × customer) with assumption presets.
- Add advanced scenarios (steepener/flattener, ramps, Monte Carlo).
- Harden the Streamlit UX (assumption templates, saved profiles, richer visuals).
- Build comprehensive unit test coverage using `pytest`.

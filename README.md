# Non-Maturity Deposit (NMD) ALM Engine

This project provides a Python-based asset-liability management engine for modelling
non-maturity deposit portfolios. The engine focuses on manual assumption entry and
flexible field mapping so it can accommodate institution-specific data layouts.

## Key Capabilities
- CSV/Excel ingestion with interactive field mapping prompts.
- Support for multiple segmentation approaches (all accounts, by account type, by customer segment).
- Manual capture of the core assumptions (decay, WAL, deposit betas up/down, repricing betas up/down).
- Scenario generation for parallel, curve-shape, and Monte Carlo shocks with configurable volatility and drift.
- Account-level cash flow projection, terminal value capture, and present value calculation.
- Flexible discounting: choose a single flat rate or supply a full yield curve (manual entry or fetched automatically from FRED).
- Interactive Plotly/Dash dashboards for Monte Carlo analytics, including live simulation monitoring and linked visual exploration.
- CSV report exports (scenario summary, cash flow detail, account-level PV) with optional Monte Carlo visualisations (rate spaghetti/fan charts, PV distributions, dashboards).

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

### Supported Scenarios
- Deterministic parallel shocks: rising and falling rate moves from +/-100 bps through +/-400 bps.
- Curve shape shocks: steepener, flattener, and front-end shock patterns derived from the base yield curve.
- Monte Carlo simulations: configurable number of simulations, quarterly volatility/drift inputs, and downloadable PV distributions.

## Project Structure
```
src/
|-- analysis/       # PV comparison and yield curve validation helpers
|-- core/           # Calculation engines (cash flows, PV, scenarios)
|-- integration/    # External data loaders (e.g., FRED Treasury curve fetcher)
|-- models/         # Pydantic models for accounts, assumptions, scenarios, results
|-- reporting/      # CSV export helpers
|-- ui/             # CLI and Streamlit interfaces
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

## Interactive Dashboards
- Run `python -m src.ui.cli` to configure Monte Carlo scenarios, then consume results via Plotly/Dash dashboards.
- Use the Python API: `from src.visualization.monte_carlo_dashboard import create_dashboard_app`; pass your `EngineResults` to spin up the full Monte Carlo dashboard.
- For live simulation monitoring, create a `LiveSimulationFeed` and pass it to `create_live_dashboard_app` to observe convergence and distributions in real time.
- Export static artefacts with `src.reporting.ReportGenerator.export_monte_carlo_visuals` (uses Plotly+Kaleido).

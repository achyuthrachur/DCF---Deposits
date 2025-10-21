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
- Download packages combining Excel workbooks, Word narrative reports, and charts (served as a zip in Streamlit; legacy CSV exports remain available via the CLI) plus optional Monte Carlo visualisations (rate spaghetti/fan charts, PV distributions, dashboards).
- Background execution pipeline keeps the Streamlit UI responsive during long multi-scenario runs and streams progress updates even when processing thousands of accounts.
- Built-in authentication with configurable username/email allowlists, optional email-delivered activation/reset tokens, and per-user password management.

## Getting Started
1. Install dependencies:
   ```
src/
|-- core/          # Calculation engines (cash flows, PV, scenarios)
|-- models/        # Pydantic models for accounts, assumptions, scenarios, results
|-- reporting/     # Disk exports and in-memory download bundler
|-- ui/            # Typer CLI for manual data entry
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
The web app renders results in-browser and automatically launches a zip download
containing the Excel workbook, Word narrative, and packaged charts; you can also trigger
the download again from within the app if needed.

### Yield Curve Sources
- **Single rate** - retains backwards compatibility for quick what-if analysis.
- **FRED API** - supply a FRED API key (`FRED_API_KEY` environment variable or UI input) to pull the latest Treasury curve. Snapshots are saved to `output/discount_curve.json`.
- **Manual curve** - enter tenor points (3M-10Y) directly in the CLI or web app.

### Supported Scenarios
- Deterministic parallel shocks: rising and falling rate moves from +/-100 bps through +/-400 bps.
- Non-parallel shocks: steepener, flattener, and short-rate focused profiles derived from the base curve.
- Monte Carlo simulations: configurable number of simulations, monthly volatility/drift inputs, and downloadable PV distributions.

## Project Structure
```
src/
|-- core/          # Calculation engines (cash flows, PV, scenarios)
|-- models/        # Pydantic models for accounts, assumptions, scenarios, results
|-- reporting/     # CSV export helpers
|-- ui/            # Typer CLI for manual data entry
```


## Sharing with Collaborators
To publish a shareable web link, deploy the Streamlit app:

1. Push this project to a GitHub repository (public or private).
2. Sign in to [Streamlit Community Cloud](https://streamlit.io/cloud) (free tier available).
3. Create a new app, selecting your repository and the `src/ui/web_app.py` entry point.
4. Streamlit Cloud automatically installs dependencies from `requirements.txt` and launches
   the app. Copy the generated URL and share it with collaborators-they can access the tool
   directly from their browser without installing anything locally.

Alternative hosting options include Streamlit on AWS/GCP/Azure, containerising with Docker
and serving via services like Azure Container Apps, or embedding the engine behind a FastAPI
service with a custom frontend. Streamlit Cloud offers the fastest path to a working link.

## Next Steps
- Extend segmentation to support cross-segmentation (account -- customer) with assumption presets.
- Add advanced scenarios (steepener/flattener, ramps, Monte Carlo).
- Harden the Streamlit UX (assumption templates, saved profiles, richer visuals).
- Build comprehensive unit test coverage using `pytest`.


## Authentication & Access Control
- Update `config/auth.yaml` with the list of permitted usernames/emails. Accounts with empty `password_hash`/`salt` will prompt for activation.
- First-time setup: open the app and initialise the administrator account by supplying an email and strong password when prompted.
- Users can request activation or password reset tokens. If SMTP details are configured under `smtp` in `config/auth.yaml` (or via the environment variable referenced in `password_env`), the app sends an email automatically; otherwise the token is displayed for manual distribution.
- Optionally set `APP_BASE_URL` to include in activation emails so recipients can navigate directly back to the app.
- Tokens expire after 60 minutes and passwords are stored as PBKDF2-SHA256 hashes.


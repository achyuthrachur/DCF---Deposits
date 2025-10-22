"""Windows desktop entry point for local, no‑Python installation runs.

This script reuses the existing interactive CLI workflow (``src.ui.cli.run``).
When packaged with PyInstaller (one‑file), users can double‑click the EXE,
choose their CSV, answer the prompts in the console, and the output files are
written to an ``output`` folder next to the EXE.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _pick_file() -> Path | None:
    """Open a native file‑picker; fall back to console input if Tk is missing."""
    try:
        import tkinter as tk  # type: ignore
        from tkinter import filedialog  # type: ignore

        try:
            root = tk.Tk()
            root.withdraw()
            filename = filedialog.askopenfilename(
                title="Select accounts CSV",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )
            return Path(filename) if filename else None
        except Exception:
            pass
    except Exception:
        pass

    # Fallback to plain input
    try:
        path_str = input("Enter path to accounts CSV: ").strip()
        return Path(path_str) if path_str else None
    except EOFError:
        return None


def main() -> None:
    # Ensure project root (source or PyInstaller bundle) is on sys.path
    if hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).resolve().parents[2]

    if str(base) not in sys.path:
        sys.path.insert(0, str(base))
    src_path = base / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from src.ui.cli import run as cli_run  # import lazily after sys.path tweak

    print("\nDCF – Deposits Desktop Runner")
    print("--------------------------------")

    # Pick file
    csv_path = _pick_file()
    if not csv_path or not csv_path.exists():
        print("No file selected. Exiting.")
        input("\nPress Enter to close…")
        return

    # Default options — keep identical to Streamlit defaults
    try:
        cli_run(
            file_path=csv_path,
            segmentation="all",
            projection_months=240,
            max_projection_months=600,
            materiality_threshold=1000.0,
            output_dir=Path("output"),
            cashflow_sample_size=20,
            generate_plots=False,
        )
    except Exception as exc:
        # Write full traceback so support can diagnose issues on user machines.
        import traceback

        log_path = Path.cwd() / "desktop_error.log"
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write("\n=== DCF Deposits Desktop Error ===\n")
            traceback.print_exc(file=fh)
        print("\nAn error occurred while running the analysis. Details were written to:")
        print(str(log_path))
        print("\nError:\n" + str(exc))
    finally:
        print("\nDone. Outputs are in the 'output' folder next to this application.")
        input("\nPress Enter to close…")


if __name__ == "__main__":
    main()

"""Launch the Streamlit UI locally inside the packaged desktop app."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_paths() -> Path:
    if hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).resolve().parents[2]

    if str(base) not in sys.path:
        sys.path.insert(0, str(base))
    src_path = base / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    return src_path


def main() -> None:
    src_path = _ensure_paths()

    os.environ.setdefault("APP_DESKTOP_MODE", "1")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "localhost")
    os.environ.setdefault("STREAMLIT_SERVER_PORT", "8501")

    script = str(src_path / "ui" / "web_app.py")

    import streamlit.web.cli as stcli

    sys.argv = ["streamlit", "run", script]

    print("Launching DCF Deposits Desktop UI at http://localhost:8501 …")
    try:
        import threading
        import time
        import webbrowser

        def _open_browser() -> None:
            time.sleep(2)
            webbrowser.open("http://localhost:8501", new=2)

        threading.Thread(target=_open_browser, daemon=True).start()
    except Exception:
        pass

    stcli.main()


if __name__ == "__main__":
    main()

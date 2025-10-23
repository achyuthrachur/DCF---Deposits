"""Launch the Streamlit UI locally inside the packaged desktop app."""

from __future__ import annotations

import atexit
import importlib
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

try:  # pragma: no cover - informs PyInstaller of runtime deps
    import src.desktop._pyinstaller_hints  # type: ignore  # noqa: F401
except Exception:
    try:
        import _pyinstaller_hints  # type: ignore  # noqa: F401
    except Exception:
        pass

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

    exe_dir = Path(getattr(sys, 'frozen', False) and Path(sys.executable).resolve().parent or Path.cwd())
    os.environ.setdefault("APP_OUTPUT_ROOT", str(exe_dir / "output"))

    os.environ.setdefault("APP_DESKTOP_MODE", "1")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "localhost")
    os.environ.setdefault("STREAMLIT_SERVER_PORT", "8501")
    os.environ.setdefault("STREAMLIT_GLOBAL_DEVELOPMENT_MODE", "false")

    spec = importlib.util.find_spec("src.ui.web_app")
    if spec is None or not spec.origin:
        raise RuntimeError("Unable to locate src.ui.web_app module inside bundle.")
    script_path = Path(spec.origin)

    if not script_path.exists():
        temp_root = Path(tempfile.mkdtemp(prefix="dcf_desktop_streamlit_"))
        stub_path = temp_root / "launch_web_app.py"
        stub_path.write_text(
            "import runpy\n"
            f"runpy.run_module('{spec.name}', run_name='__main__')\n",
            encoding="utf-8",
        )

        def _cleanup_tmp(path: Path = temp_root) -> None:
            try:
                shutil.rmtree(path, ignore_errors=True)
            except Exception:
                pass

        atexit.register(_cleanup_tmp)
        script_path = stub_path

    import streamlit.web.cli as stcli

    script = str(script_path)
    sys.argv = ["streamlit", "run", script]

    print("Launching DCF Deposits Desktop UI at http://localhost:8501 ...")
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

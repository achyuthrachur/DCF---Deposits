"""Launch the Streamlit UI locally inside the packaged desktop app."""

from __future__ import annotations

import atexit
import importlib
import importlib.util
import os
import shutil
import socket
import sys
import tempfile
from pathlib import Path
from typing import Final

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


def _pick_server_port(preferred: int = 8501) -> int:
    def _can_bind(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return True
            except OSError:
                return False

    if _can_bind(preferred):
        return preferred

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> None:
    src_path = _ensure_paths()

    exe_dir = Path(getattr(sys, 'frozen', False) and Path(sys.executable).resolve().parent or Path.cwd())
    os.environ["APP_OUTPUT_ROOT"] = str(exe_dir / "output")

    os.environ["APP_DESKTOP_MODE"] = "1"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "localhost"
    port = _pick_server_port()
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"

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
    sys.argv = [
        "streamlit",
        "run",
        script,
        "--server.headless=true",
        f"--server.port={port}",
        "--server.address=localhost",
    ]

    print(f"Launching DCF Deposits Desktop UI at http://localhost:{port} ...")
    try:
        import threading
        import time
        import webbrowser

        def _open_browser() -> None:
            time.sleep(2)
            webbrowser.open(f"http://localhost:{port}", new=2)

        threading.Thread(target=_open_browser, daemon=True).start()
    except Exception:
        pass

    stcli.main()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main_pid_env: Final[str] = "DCF_DESKTOP_MAIN_PID"
    existing_pid = os.environ.get(main_pid_env)
    current_pid = str(os.getpid())
    if existing_pid and existing_pid != current_pid:
        sys.exit(0)
    os.environ[main_pid_env] = current_pid
    main()

# -*- mode: python ; coding: utf-8 -*-

import os
import pathlib

from PyInstaller.building.build_main import Analysis, PYZ, EXE
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

project_root = pathlib.Path(os.getcwd()).resolve()

# Core entry point and paths
entry_script = project_root / "src" / "desktop" / "ui_launcher.py"
pathex = [str(project_root), str(project_root / "src")]

# Packages that rely on dynamic imports or data files.
PACKAGES_TO_COLLECT = [
    "tkinter",
    "pydantic",
    "pydantic_core",
    "fredapi",
    "typer",
    "rich",
    "streamlit",
    "plotly",
    "seaborn",
    "openpyxl",
    "matplotlib",
    "numpy",
    "pandas",
    "scipy",
    "pyarrow",
    "docx",
    "python_docx",
    "imageio",
    "yaml",
    "altair",
    "pydeck",
    "tenacity",
]

hiddenimports = []
datas = [
    (str(project_root / "src"), "src"),
    (str(project_root / "config"), "config"),
]

for package in PACKAGES_TO_COLLECT:
    try:
        hiddenimports.extend(collect_submodules(package))
    except Exception:
        pass
    try:
        datas.extend(collect_data_files(package, include_py_files=False))
    except Exception:
        pass

for package in ["streamlit", "typer", "rich", "plotly", "pydantic", "fredapi"]:
    try:
        datas.extend(copy_metadata(package))
    except Exception:
        pass

# Remove potential duplicates while preserving order.
def _dedupe(seq):
    seen = set()
    output = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output

hiddenimports = _dedupe(hiddenimports)
datas = _dedupe(datas)

block_cipher = None

a = Analysis(
    [str(entry_script)],
    pathex=pathex,
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="DCF-Deposits-UI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

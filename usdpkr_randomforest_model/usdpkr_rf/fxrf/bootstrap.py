"""Start-up helpers: auto-install packages and set the matplotlib backend.
Uses ONLY the standard library, so it can run before numpy/pandas exist."""
import sys
import importlib
import subprocess
import warnings

REQUIRED = [("numpy", "numpy"), ("pandas", "pandas"), ("scikit-learn", "sklearn"),
            ("matplotlib", "matplotlib"), ("openpyxl", "openpyxl")]


def ensure(pip_name, import_name=None):
    try:
        importlib.import_module(import_name or pip_name)
    except ImportError:
        print(f"[Install] Installing {pip_name} ...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
        except subprocess.CalledProcessError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                                   "--break-system-packages", pip_name])
        importlib.invalidate_caches()
        importlib.import_module(import_name or pip_name)


def setup():
    """Call ONCE at the very top of main.py, before importing the other modules."""
    warnings.filterwarnings("ignore")
    for pip_name, import_name in REQUIRED:
        ensure(pip_name, import_name)
    import matplotlib
    matplotlib.use("Agg")      # charts are only SAVED as PNG (never shown) -> no Tk window / Tk errors

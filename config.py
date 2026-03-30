"""Local configuration defaults for this project.

This is intentionally small and editable; keep simulation logic in calibration/.
"""

from __future__ import annotations

from pathlib import Path

from calibration.builders import build_example_metadata, build_example_raw_params


# ---- Model input ----
MODEL_INP = str(Path("models") / "PATTERN.inp")


# ---- Output folders ----
OUTPUT_DIR = Path("outputs")
RUNS_DIR = OUTPUT_DIR / "runs"
DEBUG_DIR = OUTPUT_DIR / "debug"
REPORTS_DIR = OUTPUT_DIR / "reports"


# ---- Output toggles ----
SAVE_CSV = True
SAVE_DEBUG_JSON = True
VERBOSE = True


def build_default_metadata():
    return build_example_metadata()


def build_default_raw_params() -> dict:
    return build_example_raw_params()

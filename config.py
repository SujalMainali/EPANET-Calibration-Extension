"""Local configuration defaults for this project.

This is intentionally small and editable; keep simulation logic in calibration/.
"""

from __future__ import annotations

from pathlib import Path

from calibration.builders import (
    build_example_raw_params,
    build_service_nodes_from_inp,
    build_zone_mapping_from_inp,
    load_zone_mapping_csv,
)


# ---- Model input ----
MODEL_INP = str(Path("models") / "PATTERN.inp")


# Provide your sensor node IDs here (must match INP junction IDs and observed CSV columns).
SENSOR_NODES: list[str] = [
    "HOUSEEND_16032",
    "HOUSEEND_16239",
    "HOUSEEND_16317",
    "HOUSEEND_16426",
    "HOUSEEND_16547",
    "HOUSEEND_16598",
    "HOUSEEND_16702",
]


# ---- Observations (pressures at sensor nodes) ----
# Provide the observed pressure CSV here.
# Expected: one time column + sensor columns (matching SENSOR_NODES).
OBSERVED_PRESSURE_CSV: str | None = None

# Optional: provide multiple observed CSVs (e.g., one per day). If set, this list is used
# and OBSERVED_PRESSURE_CSV is ignored. Each CSV can have its own time column, but the
# column name must be the same across files if you set OBSERVED_TIME_COLUMN.
OBSERVED_PRESSURE_CSVS: list[str] | None = ["Data/HourlyData_2025-12-18.csv", "Data/HourlyData_2025-12-19.csv", "Data/HourlyData_2025-12-20.csv", "Data/HourlyData_2025-12-21.csv"]

# Optional: if your observed CSV's time column is NOT the first column, set it here.
# If None, the loader assumes the first column is the time column.
OBSERVED_TIME_COLUMN: str | None = None


# Optional: node->zone mapping CSV created by scripts/assign_zones_from_coords.py
# Set to None to keep all zones as "UNKNOWN".
ZONE_MAP_CSV: str | None = None

# If True and ZONE_MAP_CSV is None, derive zones directly from INP [COORDINATES]
# (median split into 4 zones).
AUTO_ZONES_FROM_INP: bool = True

# Zone assignment mode for AUTO_ZONES_FROM_INP. Currently supported: "grid".
ZONE_ASSIGN_MODE: str = "grid"


# ---- Output folders ----
OUTPUT_DIR = Path("outputs")
RUNS_DIR = OUTPUT_DIR / "runs"
DEBUG_DIR = OUTPUT_DIR / "debug"
REPORTS_DIR = OUTPUT_DIR / "reports"


# ---- Output toggles ----
SAVE_CSV = True
SAVE_DEBUG_JSON = True
VERBOSE = True


# ---- Optimizer (gradient descent) ----

# Which raw parameter paths to optimize.
# Examples:
#   - "demand.demand_multiplier"
#   - "pda.required_pressure"
#   - "pattern_family.morning_center"
#   - "leakage.zone_multipliers.Z_NW"   (if you enable zone leakage)
OPT_PARAM_PATHS: list[str] = [
    "demand.demand_multiplier",
    "pda.required_pressure",
]

# Optional bounds per parameter path.
# Any param not listed here is left unbounded.
OPT_BOUNDS: dict[str, tuple[float, float]] = {
    "demand.demand_multiplier": (0.5, 1.8),
    "pda.required_pressure": (8.0, 35.0),
}

# Gradient descent settings
OPT_MAX_ITERS: int = 20
OPT_LEARNING_RATE: float = 0.5
OPT_LEARNING_RATE_DECAY: float = 0.95

# Finite-difference step sizing
OPT_FD_EPS_REL: float = 1e-2  # eps = eps_rel * max(1, |x|)
OPT_FD_EPS_ABS: float = 1e-3  # absolute floor

# Stop when relative improvement is small
OPT_TOL_REL: float = 1e-4

# Output
OPT_HISTORY_CSV = REPORTS_DIR / "opt_history.csv"
OPT_BEST_PARAMS_JSON = REPORTS_DIR / "best_params.json"


def build_default_metadata():
    zone_by_node = None
    if ZONE_MAP_CSV:
        zone_by_node = load_zone_mapping_csv(ZONE_MAP_CSV)
    elif AUTO_ZONES_FROM_INP:
        zone_by_node = build_zone_mapping_from_inp(
            MODEL_INP,
            mode=ZONE_ASSIGN_MODE,
            node_prefix="HOUSEEND_",
        )

    metadata, info = build_service_nodes_from_inp(
        MODEL_INP,
        sensor_nodes=SENSOR_NODES,
        zone_by_node=zone_by_node,
    )
    if VERBOSE:
        print("[metadata]", info)
    return metadata


def build_default_raw_params() -> dict:
    return build_example_raw_params()

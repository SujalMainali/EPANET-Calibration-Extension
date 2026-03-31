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
OBSERVED_PRESSURE_CSV: str | None = "Calibrated_hourly_2025-12-18_Final_t.csv"

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

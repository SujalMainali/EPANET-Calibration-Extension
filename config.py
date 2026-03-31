"""Local configuration defaults for this project.

This is intentionally small and editable; keep simulation logic in calibration/.
"""

from __future__ import annotations

from pathlib import Path

from calibration.builders import (
    build_example_raw_params,
    build_leak_nodes_from_ids,
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
# Supported: "grid", "circular"
ZONE_ASSIGN_MODE: str = "circular"

# Which INP nodes are eligible to be assigned zones.
# - None: consider all nodes that have coordinates
# - "HOUSEEND_": only assign zones to HOUSEEND nodes (legacy behavior)
# For circular zoning + leak calibration, you typically want None.
ZONE_NODE_PREFIX: str | None = None

# Circular zone definitions (used when ZONE_ASSIGN_MODE == "circular").
# - Coordinates and radius use the same units as INP [COORDINATES].
# - Nodes outside all circles remain as service nodes with zone "UNKNOWN".
# - Overlaps are resolved by nearest center.
ZONE_CIRCULAR_SPECS: list[dict] = [
    
    {"zone": "Z_A", "center_node": "HOUSEEND_16598", "radius": 193.0},
    {"zone": "Z_B", "center_node": "HOUSEEND_16426", "radius": 132.0},
    {"zone": "Z_C", "center_node": "HOUSEEND_16702", "radius": 327.0},
    {"zone": "Z_D", "center_node": "HOUSEEND_16032", "radius": 118.0},
]

# When circular zoning is enabled, optionally drop HOUSEEND nodes that aren't assigned to any zone.
# Default False: nodes outside all circles still exist and simply have zone "UNKNOWN".
ZONE_IGNORE_UNASSIGNED_NODES: bool = False


# ---- Leak/emitter nodes (optional but needed if you want leaks to affect optimization) ----

# If True, all service nodes that are assigned to a zone (zone != "UNKNOWN")
# are added as leak/emitter nodes automatically.
LEAK_NODES_FROM_ZONES: bool = True

# Nodes where emitters will be applied (must exist in the INP node list).
# Leave empty to disable emitters/leaks entirely.
LEAK_NODE_IDS: list[str] = []

# Optional: instead of enumerating IDs, select leak nodes by prefix (e.g., "LEAK_" or "EMITTER_").
# If set and LEAK_NODE_IDS is empty, all INP nodes with any of these prefixes are used.
LEAK_NODE_PREFIXES: list[str] = []

LEAKS_ENABLED: bool = bool(LEAK_NODES_FROM_ZONES or LEAK_NODE_IDS or LEAK_NODE_PREFIXES)

# Default weight applied to each leak node (emitter coefficient = zone_multiplier * weight).
LEAK_NODE_DEFAULT_WEIGHT: float = 1.0


# ---- Output folders ----
OUTPUT_DIR = Path("outputs")
RUNS_DIR = OUTPUT_DIR / "runs"
DEBUG_DIR = OUTPUT_DIR / "debug"
REPORTS_DIR = OUTPUT_DIR / "reports"


# ---- Output toggles ----
SAVE_CSV = True
SAVE_DEBUG_JSON = True
VERBOSE = True


# ---- Objective function weights (editable) ----
# Controls the combined objective:
#   J = w_ts*J_timeseries + w_feat*J_features + w_sp*J_spatial + w_vol*J_volume + w_reg*J_regularization
# Used by optimize.py (and compare.py).
OBJECTIVE_WEIGHTS: dict[str, float] = {
    "w_ts": 0.40,
    "w_feat": 0.30,
    "w_sp": 0.15,
    "w_vol": 0.10,
    "w_reg": 0.05,
}


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
    "pda.minimum_pressure",
    "pda.pressure_exponent",
    # Global 24-hour demand pattern shape parameters (applies to all service nodes).
    # These control *when* demand happens (centers), how spread it is (widths), and
    # relative peak strengths (weights). Total daily demand is still controlled by
    # demand.demand_multiplier because the pattern is normalized to sum to 24.
    "pattern_family.morning_center",
    "pattern_family.morning_width",
    "pattern_family.morning_weight",
    "pattern_family.noon_center",
    "pattern_family.noon_width",
    "pattern_family.noon_weight",
    "pattern_family.evening_center",
    "pattern_family.evening_width",
    "pattern_family.evening_weight",
    "pattern_family.background_weight",
    "pattern_family.floor",
    "leakage.global_scale",
    "leakage.emitter_exponent",
]

# Add zone leakage multipliers to the optimization set (one parameter per zone).
# Note: These only matter if you also configure leak nodes (LEAK_NODE_IDS or LEAK_NODE_PREFIXES).
if AUTO_ZONES_FROM_INP and LEAKS_ENABLED:
    if ZONE_ASSIGN_MODE == "circular" and ZONE_CIRCULAR_SPECS:
        OPT_PARAM_PATHS.extend([f"leakage.zone_multipliers.{z['zone']}" for z in ZONE_CIRCULAR_SPECS])
    elif ZONE_ASSIGN_MODE == "grid":
        OPT_PARAM_PATHS.extend(
            [
                "leakage.zone_multipliers.Z_NW",
                "leakage.zone_multipliers.Z_NE",
                "leakage.zone_multipliers.Z_SW",
                "leakage.zone_multipliers.Z_SE",
            ]
        )

# Optional bounds per parameter path.
# Any param not listed here is left unbounded.
OPT_BOUNDS: dict[str, tuple[float, float]] = {
    # Guard rails for parameters that must stay non-negative / positive to satisfy
    # validation in calibration/parameterization_layer.py.
    "demand.demand_multiplier": (1e-6, 10.0),

    # Demand pattern family bounds (see calibration/datamodels.py: PatternFamilyParams).
    # Centers are hours in [0, 23]. Widths are in hours and must be > 0.
    # Weights and floor are kept non-negative to avoid negative hourly demands.
    "pattern_family.morning_center": (0.0, 23.0),
    "pattern_family.noon_center": (0.0, 23.0),
    "pattern_family.evening_center": (0.0, 23.0),
    "pattern_family.morning_width": (0.05, 8.0),
    "pattern_family.noon_width": (0.05, 8.0),
    "pattern_family.evening_width": (0.05, 8.0),
    "pattern_family.morning_weight": (0.0, 5.0),
    "pattern_family.noon_weight": (0.0, 5.0),
    "pattern_family.evening_weight": (0.0, 5.0),
    "pattern_family.background_weight": (0.0, 5.0),
    # Keep a tiny floor so the pattern sum stays positive even if all weights drift to 0.
    "pattern_family.floor": (1e-6, 2.0),

    # Exponents close to zero can lead to numerical issues.
    "pda.pressure_exponent": (0.05, 5.0),
    "leakage.global_scale": (0.0, 50.0),
    # EPANET emitter law is Q = C * P^n. Extremely small n can produce NaNs
    # (e.g., when pressures dip negative). Keep n in a reasonable range.
    "leakage.emitter_exponent": (0.1, 2.0),

    # Optional tighter bounds (uncomment if you want a smaller search space):
    # "pda.required_pressure": (8.0, 35.0),
    # "pda.minimum_pressure": (0.0, 10.0),
}

if LEAKS_ENABLED:
    if ZONE_ASSIGN_MODE == "circular" and ZONE_CIRCULAR_SPECS:
        for z in ZONE_CIRCULAR_SPECS:
            OPT_BOUNDS[f"leakage.zone_multipliers.{z['zone']}"] = (0.0, 5.0)
    elif ZONE_ASSIGN_MODE == "grid":
        for zname in ("Z_NW", "Z_NE", "Z_SW", "Z_SE"):
            OPT_BOUNDS[f"leakage.zone_multipliers.{zname}"] = (0.0, 5.0)

# Gradient descent settings
OPT_MAX_ITERS: int = 100
OPT_LEARNING_RATE: float = 0.05
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
            node_prefix=ZONE_NODE_PREFIX,
            circular_specs=ZONE_CIRCULAR_SPECS if ZONE_ASSIGN_MODE == "circular" else None,
        )

    metadata, info = build_service_nodes_from_inp(
        MODEL_INP,
        sensor_nodes=SENSOR_NODES,
        zone_by_node=zone_by_node,
        ignore_unmapped_zones=bool(ZONE_IGNORE_UNASSIGNED_NODES and ZONE_ASSIGN_MODE == "circular"),
    )

    # Optional leak nodes where emitters are applied.
    # 1) from zones: all JUNCTION nodes that are assigned a zone (inside circles / grid zones)
    leak_ids: list[str] = []
    if LEAK_NODES_FROM_ZONES and zone_by_node:
        import wntr

        wn = wntr.network.WaterNetworkModel(MODEL_INP)
        zoned = set(str(k) for k in zone_by_node.keys())
        for jname in wn.junction_name_list:
            sj = str(jname)
            if sj in zoned:
                leak_ids.append(sj)

    # 2) from explicit IDs
    leak_ids.extend(list(LEAK_NODE_IDS))

    # 3) from prefixes
    if LEAK_NODE_PREFIXES:
        import wntr

        wn = wntr.network.WaterNetworkModel(MODEL_INP)
        for nid in wn.node_name_list:
            sid = str(nid)
            if any(sid.startswith(pref) for pref in LEAK_NODE_PREFIXES):
                leak_ids.append(sid)

    # De-duplicate while preserving order
    seen = set()
    leak_ids = [x for x in leak_ids if not (x in seen or seen.add(x))]

    if leak_ids:
        metadata.leak_nodes = build_leak_nodes_from_ids(
            MODEL_INP,
            leak_ids,
            zone_by_node=zone_by_node,
            default_weight=float(LEAK_NODE_DEFAULT_WEIGHT),
        )
        metadata.leak_check_node = next(iter(metadata.leak_nodes.keys()), None)

    if VERBOSE:
        print("[metadata]", info)
    return metadata


def build_default_raw_params() -> dict:
    raw = build_example_raw_params()

    # Seed zone multipliers for whichever zones exist, so optimization has a sensible start.
    lk = raw.setdefault("leakage", {})
    zm = lk.setdefault("zone_multipliers", {})
    if ZONE_ASSIGN_MODE == "circular" and ZONE_CIRCULAR_SPECS:
        for z in ZONE_CIRCULAR_SPECS:
            zm.setdefault(str(z["zone"]), 1.0)
    elif AUTO_ZONES_FROM_INP and ZONE_ASSIGN_MODE == "grid":
        for zname in ("Z_NW", "Z_NE", "Z_SW", "Z_SE"):
            zm.setdefault(zname, 1.0)

    return raw

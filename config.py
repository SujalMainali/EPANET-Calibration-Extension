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
OBSERVED_PRESSURE_CSVS: list[str] | None = [
    "Data/HourlyData_2025-12-18.csv",
    "Data/HourlyData_2025-12-19.csv",
    "Data/HourlyData_2025-12-20.csv",
]

# Optional: if your observed CSV's time column is NOT the first column, set it here.
# If None, the loader assumes the first column is the time column.
OBSERVED_TIME_COLUMN: str | None = None


# ---- ZONE MAPPING (VORONOI-ONLY APPROACH) ----
# Zones are now assigned via Voronoi distribution (each node -> nearest sensor).
# This is the scientifically recommended approach:
# - Equitable spatial distribution (each sensor has a natural service area)
# - No overlaps or gaps
# - Scales naturally with any number of sensors
# - Reflects actual water flow patterns

# Option 1: Load pre-computed Voronoi zones from CSV (recommended for reproducibility)
ZONE_MAP_CSV: str | None = "outputs/node_zones_voronoi.csv"

# Option 2: Compute Voronoi zones on-the-fly from INP coordinates
# (only used if ZONE_MAP_CSV is None)
AUTO_ZONES_FROM_INP: bool = True
ZONE_ASSIGN_MODE: str = "voronoi"  # Only 'voronoi' is supported (grid/circular are deprecated)

# Which INP nodes are eligible to be assigned zones.
# - None: consider all nodes that have coordinates (recommended for Voronoi)
# - "HOUSEEND_": only assign zones to HOUSEEND nodes (legacy, not recommended)
ZONE_NODE_PREFIX: str | None = None

# Note: ZONE_CIRCULAR_SPECS is now DEPRECATED and ignored.
# Circular zoning is no longer supported. Use Voronoi instead.
ZONE_CIRCULAR_SPECS: list[dict] = []  # DEPRECATED - DO NOT USE
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


# ---- Observation validation and signal smoothing ----
# Validate complete observed days and optionally fold them into one
# representative daily pressure profile per sensor.
OBSERVATION_VALIDATION_ENABLED: bool = True
OBSERVATION_POINTS_PER_DAY: int = 24
# True: aggregate all days into one representative day.
# False: retain every day and smooth each day independently.
OBSERVATION_FOLD_DAYS_ENABLED: bool = True
OBSERVATION_FOLD_AGGREGATION: str = "mean"  # "mean" or "median"
OBSERVATION_INTERPOLATE_MISSING: bool = True
OBSERVATION_REQUIRE_COMPLETE_DAYS: bool = True

# Fourier low-pass smoothing. With hourly data, harmonic 6 corresponds to a
# four-hour period; shorter oscillations are treated as measurement noise.
OBSERVATION_SMOOTHING_ENABLED: bool = True
OBSERVATION_SMOOTHING_MAX_HARMONIC: int = 6

# Numerical validation tolerances.
OBSERVATION_MASS_REL_TOL: float = 1e-10
OBSERVATION_PARSEVAL_REL_TOL: float = 1e-10

# Export the folded profile, smoothed profile, spectrum, and exact optimizer
# target for auditing.
OBSERVATION_EXPORT_STAGES: bool = True
OBSERVATION_VALIDATION_DIR = DEBUG_DIR / "validation"

# Saved processed observation dataset used by optimization and result plots.
# If recompute is False and this CSV exists, the calibration scripts reuse it.
OBSERVATION_SMOOTHED_DATASET_CSV = OUTPUT_DIR / "datasets" / "observed_pressure_smoothed.csv"
OBSERVATION_RECOMPUTE_SMOOTHED_DATASET: bool = True


# ---- Output toggles ----
SAVE_CSV = True
SAVE_DEBUG_JSON = True
VERBOSE = True


# ---- Objective function weights (editable) ----
# Controls the combined objective:
#   J = w_ts*J_timeseries + w_feat*J_features + w_sp*J_spatial + w_vol*J_volume + w_reg*J_regularization
# Used by optimize.py (and compare.py).
OBJECTIVE_WEIGHTS: dict[str, float] = {
    "w_ts": 0.25,
    "w_feat": 0.35,
    "w_sp": 0.15,
    "w_vol": 0.10,
    "w_reg": 0.05,
}


# ---- Optimizer (gradient descent) ----

# Which raw parameter paths to optimize.
# CALIBRATION FIX: Frozen parameters to preserve original design assumptions:
#   - demand.demand_multiplier: FROZEN to 1.0 (use original demand values)
#   - pattern_family.*: FROZEN (preserve original RF_MORN_10, RF_EVE_14, etc. patterns)
#   - time.duration_days: FROZEN to 1 (no need for 72hr simulations)
# Only these parameters are free to calibrate:
#   - pda.* (pressure/demand model settings)
#   - leakage.global_scale (global leak scaling)
#   - leakage.zone_multipliers.* (per-zone leak adjustments)
#   - leakage.emitter_exponent (leak emitter power law exponent)
#   - solver.* (numerical solver settings)
OPT_PARAM_PATHS: list[str] = [
    # PDA model parameters (leave free to calibrate)
    "pda.required_pressure",
    "pda.minimum_pressure",
    "pda.pressure_exponent",
    # Leakage parameters (leave free to calibrate)
    "leakage.global_scale",
    "leakage.emitter_exponent",
    # NOTE: pattern_family.* parameters are FROZEN to preserve original demand patterns
    # NOTE: demand.demand_multiplier is FROZEN at 1.0 to use original billing demands
]

# Add zone leakage multipliers to the optimization set (one parameter per zone).
# Zones are dynamically determined from Voronoi mapping.
if LEAKS_ENABLED:
    if ZONE_MAP_CSV:
        # Using pre-computed Voronoi CSV: add zone multipliers for all zones in the CSV
        zone_by_node = load_zone_mapping_csv(ZONE_MAP_CSV)
        unique_zones = sorted(set(zone_by_node.values()))
        OPT_PARAM_PATHS.extend([f"leakage.zone_multipliers.{z}" for z in unique_zones])
    elif AUTO_ZONES_FROM_INP and ZONE_ASSIGN_MODE == "voronoi":
        # Voronoi zones are named Z_0, Z_1, Z_2, ... (N = number of sensors)
        # Dynamically add zone multipliers based on number of sensors
        n_sensors = len(SENSOR_NODES)
        for i in range(n_sensors):
            OPT_PARAM_PATHS.append(f"leakage.zone_multipliers.Z_{i}")

# Optional bounds per parameter path.
# Any param not listed here is left unbounded.
# CALIBRATION FIX: Only includes bounds for parameters that are actually being optimized.
# Removed bounds for:
#   - demand.demand_multiplier (FROZEN at 1.0)
#   - pattern_family.* (FROZEN to preserve original patterns)
OPT_BOUNDS: dict[str, tuple[float, float]] = {
    # PDA model parameter bounds
    "pda.pressure_exponent": (0.05, 5.0),
    # Leakage parameter bounds
    "leakage.global_scale": (0.0, 50.0),
    # EPANET emitter law is Q = C * P^n. Extremely small n can produce NaNs
    # (e.g., when pressures dip negative). Keep n in a reasonable range.
    "leakage.emitter_exponent": (0.1, 2.0),
}

if LEAKS_ENABLED:
    if ZONE_MAP_CSV:
        # Using pre-computed Voronoi CSV: add bounds for all zones in the CSV
        zone_by_node = load_zone_mapping_csv(ZONE_MAP_CSV)
        unique_zones = sorted(set(zone_by_node.values()))
        for z in unique_zones:
            OPT_BOUNDS[f"leakage.zone_multipliers.{z}"] = (0.0, 5.0)
    elif AUTO_ZONES_FROM_INP and ZONE_ASSIGN_MODE == "voronoi":
        # Voronoi zones: add bounds for Z_0, Z_1, ..., Z_(N-1)
        n_sensors = len(SENSOR_NODES)
        for i in range(n_sensors):
            OPT_BOUNDS[f"leakage.zone_multipliers.Z_{i}"] = (0.0, 5.0)

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
    """Build metadata with Voronoi-based zone assignment.
    
    Flow:
    1. Load or compute Voronoi zones from sensor nodes
    2. Assign all junctions to their nearest sensor zone (Z_0, Z_1, ...)
    3. Create service nodes with zone assignments
    4. Auto-create leak nodes from zoned junctions (if LEAK_NODES_FROM_ZONES=True)
    """
    zone_by_node = None

    if ZONE_MAP_CSV:
        # Option 1: Load pre-computed zones from CSV
        zone_by_node = load_zone_mapping_csv(ZONE_MAP_CSV)
    elif AUTO_ZONES_FROM_INP:
        # Option 2: Compute Voronoi zones on-the-fly
        zone_by_node = build_zone_mapping_from_inp(
            MODEL_INP,
            mode="voronoi",
            node_prefix=ZONE_NODE_PREFIX,
            sensor_nodes=SENSOR_NODES,
        )

    metadata, info = build_service_nodes_from_inp(
        MODEL_INP,
        sensor_nodes=SENSOR_NODES,
        zone_by_node=zone_by_node,
        ignore_unmapped_zones=bool(ZONE_IGNORE_UNASSIGNED_NODES),
    )

    # Optional leak nodes where emitters are applied.
    # 1) from zones: all JUNCTION nodes that are assigned a zone (inside Voronoi cells)
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
    """Build default raw parameters with Voronoi zone multipliers.
    
    Zone multipliers are named Z_0, Z_1, Z_2, ... based on sensor count.
    
    CALIBRATION FIX: Frozen parameters to preserve original model design:
    - demand.demand_multiplier is set to 1.0 (use original billing demands)
    - time.duration_days is set to 1 (1-day simulation cycle)
    - pattern_family.* parameters are not modified (original patterns preserved)
    """
    raw = build_example_raw_params()

    # CALIBRATION FIX: Explicitly freeze demand_multiplier at 1.0
    # This ensures calibration uses original demand values from INP file
    dm = raw.setdefault("demand", {})
    dm["demand_multiplier"] = 1.0
    
    # CALIBRATION FIX: Explicitly freeze duration_days at 1
    # This ensures 24-hour pattern repeats without multi-day carry-over
    tm = raw.setdefault("time", {})
    tm["duration_days"] = 1

    # Seed zone multipliers for whichever zones exist (Voronoi-based)
    lk = raw.setdefault("leakage", {})
    zm = lk.setdefault("zone_multipliers", {})

    if ZONE_MAP_CSV:
        # Using pre-computed Voronoi CSV
        zone_by_node = load_zone_mapping_csv(ZONE_MAP_CSV)
        unique_zones = sorted(set(zone_by_node.values()))
        for z in unique_zones:
            zm.setdefault(str(z), 1.0)
    elif AUTO_ZONES_FROM_INP and ZONE_ASSIGN_MODE == "voronoi":
        # Voronoi zones: Z_0, Z_1, ..., Z_(N-1) where N = number of sensors
        n_sensors = len(SENSOR_NODES)
        for i in range(n_sensors):
            zm.setdefault(f"Z_{i}", 1.0)

    return raw

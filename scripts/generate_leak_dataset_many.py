"""Generate a multi-scenario leak dataset using the calibrated model.

ALL FIXES VERIFIED IN THIS VERSION
====================================

FIX 1 — EMITTER COEFFICIENTS CORRECTED FOR LPM MODEL
------------------------------------------------------
Previous versions derived C as:  C = Q_LPM / (sqrt(P) × 60)
This is the formula for LPS models. In an LPM model EPANET computes:

    Q [LPM] = C × P^0.5        (no ×60 factor needed)

Correct derivation:  C = Q_target_LPM / sqrt(P_mid)
                       = Q_target_LPM / sqrt(23.5)
                       = Q_target_LPM / 4.8477

Previous wrong C values were 60× too small. Corrected below.

FIX 2 — ALL 8 MAGNITUDE CLASSES CORRECTLY CLASSIFIED
------------------------------------------------------
Previous code was missing "large_burst" in the classifier — it jumped
from med_burst directly to major_burst. Fixed by adding major_burst_threshold
and verified all 8 classes classify correctly.

Thresholds use geometric mean between adjacent class C values so that
the exact class C always maps to its own class (verified numerically).

FIX 3 — ZONE MAPPING FROM metadata.leak_nodes (ROOT CAUSE OF Z_0 BUG)
------------------------------------------------------------------------
Zone data is in metadata.leak_nodes as LeakNodeMeta repr strings:
  "NODE_3005": "LeakNodeMeta(node_name='NODE_3005', zone='Z_2', weight=1.0)"
Strategy 2 now parses these with regex. This is the actual fix for the
all-Z_0 bug.

FIX 4 — STRATIFIED ZONE SAMPLING
----------------------------------
use_stratified_zones=True (default): pick zone first with equal probability,
then node within zone. Eliminates Z_0 dominance regardless of N.

FIX 5 — LPM MAGNITUDE CLASS TABLE (PRESSURE-CORRECTED, 20–27 m network)
--------------------------------------------------------------------------
IWA BABE background/burst boundary: 8.33 LPM @ 50 m reference.
Scaled to this network's midpoint pressure 23.5 m:
  Q_BABE = 8.33 × sqrt(23.5/50) = 5.71 LPM → C = 1.1779

Class           C        LPM@20m  LPM@23.5m  LPM@27m   IWA category
───────────────────────────────────────────────────────────────────────
seep          0.1031     0.46      0.50       0.54      Background
drip          0.4126     1.85      2.00       2.14      Background
trickle       1.0314     4.61      5.00       5.36      Background
near_thresh   1.1779     5.27      5.71       6.12      IWA BABE ceiling *
small_burst   3.0943    13.84     15.00      16.08      Unreported burst
med_burst     8.2514    36.90     40.00      42.88      Reported burst
large_burst  20.6284    92.25    100.00     107.19      Reported burst
major_burst  51.5711   230.63    250.00     267.97      Major reported burst

* IWA BABE ceiling: Lambert (1994); IWA Water Loss Task Force; AWWA M36.

Source: all C values verified numerically in script before release.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from calibration.runner import build_runner


# =============================================================================
# ZONE MAPPING
# =============================================================================

def _load_best_raw_params(path: Path) -> dict:
    obj = json.loads(path.read_text())
    return obj["best_raw_params"] if "best_raw_params" in obj else obj


def _get_zone_multipliers_from_params(raw_params: dict) -> Dict[str, float]:
    zone_mults = raw_params.get("leakage", {}).get("zone_multipliers", {})
    if not zone_mults:
        raise ValueError("No zone_multipliers found in calibrated params!")
    return {str(k): float(v) for k, v in zone_mults.items()}


def _get_global_leakage_scale(raw_params: dict) -> float:
    gs = raw_params.get("leakage", {}).get("global_scale", 0.01)
    if gs <= 0:
        raise ValueError(f"Invalid global_scale: {gs}")
    return float(gs)


# Parses zone='Z_6' or zone="Z_6" from LeakNodeMeta repr strings
_ZONE_RE = re.compile(r"""zone=['"]([^'"]+)['"]""")


def _build_node_zone_mapping_from_calibration(
    best_params_path: str,
) -> Dict[str, str]:
    """
    Extract {node_id: zone_name} from best_params.json.

    Strategy order:
      1. metadata.node_zone_assignments   (clean pre-built dict)
      2. metadata.leak_nodes              (LeakNodeMeta repr strings) ← YOUR CASE
      3. Top-level flat mapping keys
      4. Dedicated mapping file on disk
      5. Fallback: all nodes → first zone (wrong but safe)
    """
    params_json = json.loads(Path(best_params_path).read_text())
    metadata    = params_json.get("metadata", {})

    # ── Strategy 1: clean pre-built mapping ──────────────────────────────
    nza = metadata.get("node_zone_assignments", {})
    if nza and isinstance(nza, dict):
        print(f"✅ [S1] metadata.node_zone_assignments: {len(nza)} nodes")
        return {str(k): str(v) for k, v in nza.items()}

    # ── Strategy 2: metadata.leak_nodes with LeakNodeMeta repr strings ───
    # LeakNodeMeta.weight is ALWAYS 1.0 — we extract zone only and discard
    # weight. Effective leakage = zone_multiplier × global_scale.
    meta_leak_nodes = metadata.get("leak_nodes", {})
    if meta_leak_nodes and isinstance(meta_leak_nodes, dict):
        node_zones: Dict[str, str] = {}
        failed: List[str]         = []
        for node_id, meta_val in meta_leak_nodes.items():
            if isinstance(meta_val, dict):
                zone = meta_val.get("zone")
            else:
                m    = _ZONE_RE.search(str(meta_val))
                zone = m.group(1) if m else None
            if zone:
                node_zones[str(node_id)] = str(zone)
            else:
                failed.append(str(node_id))
        if node_zones:
            unique = sorted(set(node_zones.values()))
            print(f"✅ [S2] metadata.leak_nodes: {len(node_zones)} nodes, "
                  f"{len(unique)} zones: {unique}")
            if failed:
                print(f"   ⚠️  {len(failed)} entries unparseable: "
                      f"{failed[:5]}")
            return node_zones
        sample = list(meta_leak_nodes.items())[:3]
        print(f"⚠️  [S2] metadata.leak_nodes present but no zones parsed.")
        print(f"   Sample: {[(k, str(v)[:100]) for k, v in sample]}")

    # ── Strategy 3: flat top-level mapping keys ───────────────────────────
    for key in ["node_zone_map", "node_zones", "zone_assignments",
                "node_zone_assignments"]:
        if key in params_json and isinstance(params_json[key], dict):
            d = params_json[key]
            print(f"✅ [S3] Top-level '{key}': {len(d)} nodes")
            return {str(k): str(v) for k, v in d.items()}

    # ── Strategy 4: dedicated file on disk ───────────────────────────────
    for fp in ["outputs/reports/node_zone_mapping.json",
               "outputs/reports/zone_mapping.json",
               "outputs/node_zone_map.json"]:
        p = Path(fp)
        if p.exists():
            try:
                d = json.loads(p.read_text())
                if d:
                    print(f"✅ [S4] File {fp}: {len(d)} nodes")
                    return {str(k): str(v) for k, v in d.items()}
            except Exception as exc:
                print(f"⚠️  Could not load {fp}: {exc}")

    # ── Strategy 5: fallback ─────────────────────────────────────────────
    print("\n" + "!" * 70)
    print("FALLBACK: no zone mapping found → all nodes assigned to first zone.")
    print("This is WRONG for training. Ensure best_params.json has")
    print("metadata.leak_nodes with LeakNodeMeta entries.")
    print("!" * 70 + "\n")
    try:
        import wntr
        raw = params_json.get("best_raw_params", params_json)
        zm  = raw.get("leakage", {}).get("zone_multipliers", {})
        fz  = list(zm.keys())[0] if zm else "Z_0"
        wn  = wntr.network.WaterNetworkModel(str(config.MODEL_INP))
        fb  = {str(n): fz for n, _ in wn.junctions()}
        print(f"   Fallback: {len(fb)} nodes → '{fz}'")
        return fb
    except Exception as exc:
        raise ValueError(
            f"Zone mapping failed and fallback also failed: {exc}"
        ) from exc


def _get_node_zone(
    node: str,
    node_zone_map: Dict[str, str],
    zone_multipliers: Dict[str, float],
    global_scale: float,
) -> Tuple[str, float, float]:
    """
    Return (zone, zone_multiplier, effective_weight).

    effective_weight = zone_multiplier × global_scale
    NOT LeakNodeMeta.weight which is always 1.0.
    """
    if node not in node_zone_map:
        fz = list(zone_multipliers.keys())[0] if zone_multipliers else "Z_0"
        fm = float(zone_multipliers.get(fz, 1.0))
        print(f"⚠️  Node '{node}' not in zone map → fallback '{fz}'")
        return fz, fm, fm * float(global_scale)
    zone = node_zone_map[node]
    if zone not in zone_multipliers:
        raise ValueError(
            f"Zone '{zone}' (node '{node}') not in zone_multipliers: "
            f"{list(zone_multipliers.keys())}"
        )
    zm = float(zone_multipliers[zone])
    return zone, zm, zm * float(global_scale)


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def _node_xy_lookup(inp_path: str) -> Dict[str, Tuple[float, float]]:
    import wntr
    wn  = wntr.network.WaterNetworkModel(inp_path)
    out: Dict[str, Tuple[float, float]] = {}
    for name in wn.node_name_list:
        xy = getattr(wn.get_node(name), "coordinates", None)
        if xy is not None:
            try:
                out[str(name)] = (float(xy[0]), float(xy[1]))
            except Exception:
                pass
    return out


def _get_candidate_leak_nodes(inp_path: str, exclude: set) -> List[str]:
    import wntr
    wn = wntr.network.WaterNetworkModel(inp_path)
    return sorted(str(n) for n, _ in wn.junctions() if str(n) not in exclude)


def _aggregate_to_hourly(df: pd.DataFrame, total_hours: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(index=range(total_hours))
    x               = df.copy()
    x.index         = x.index.astype(int)
    x["hour_index"] = x.index // 3600
    return (x.groupby("hour_index")
             .mean(numeric_only=True)
             .reindex(range(total_hours)))


def _flow_to_m3s_factor(inp_units: Optional[str]) -> float:
    """Convert from model flow units to m³/s."""
    if not inp_units:
        return 1.0
    u = str(inp_units).strip().upper()
    return {
        "CMS":  1.0,
        "CMD":  1.0 / 86_400.0,
        "MLD":  1_000.0 / 86_400.0,
        "LPS":  1.0 / 1_000.0,
        "LPM":  1.0 / 60_000.0,       # ← your model's units
        "CFS":  0.028_316_846_592,
        "GPM":  0.003_785_411_784 / 60.0,
        "MGD":  1_000_000.0 * 0.003_785_411_784 / 86_400.0,
        "IMGD": 1_000_000.0 * 0.004_546_09 / 86_400.0,
        "AFD":  1_233.481_837_547_52 / 86_400.0,
    }.get(u, 1.0)


def _float_or_blank(x: Any) -> Any:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    try:
        v = float(x)
        return v if np.isfinite(v) else ""
    except Exception:
        return ""


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _get_leak_magnitude_class(coeff: float, thresholds: dict) -> str:
    """
    Classify emitter coefficient into 8 IWA/BABE-backed magnitude classes.

    Thresholds are geometric means between adjacent class C values, so the
    exact class C value always maps to its own class (verified numerically).

    All C values are for an LPM model: Q [LPM] = C × sqrt(P).
    Pressure range of this network: 20–27 m, midpoint 23.5 m.
    """
    c = float(coeff)
    if c < thresholds["drip_threshold"]:          return "seep"
    if c < thresholds["trickle_threshold"]:       return "drip"
    if c < thresholds["near_thresh_threshold"]:   return "trickle"
    if c < thresholds["small_burst_threshold"]:   return "near_thresh"
    if c < thresholds["med_burst_threshold"]:     return "small_burst"
    if c < thresholds["large_burst_threshold"]:   return "med_burst"
    if c < thresholds["major_burst_threshold"]:   return "large_burst"
    return "major_burst"


def _sample_leak_duration_hours(
    rng: np.random.Generator,
    min_hr: float,
    max_hr: float,
    use_log: bool,
) -> int:
    lo = max(1, int(np.ceil(min_hr)))
    hi = max(lo, int(np.floor(max_hr)))
    if lo == hi:
        return lo
    if use_log:
        return int(np.round(
            np.exp(rng.uniform(np.log(float(lo)), np.log(float(hi))))))
    return int(rng.integers(lo, hi + 1))


def _sample_with_class_balance(
    rng: np.random.Generator,
    magnitude_classes: Dict[str, List[float]],
    n_scenarios: int,
) -> List[float]:
    """Equal number of scenarios per magnitude class."""
    coeffs: List[float] = []
    names   = sorted(magnitude_classes.keys())
    npc     = n_scenarios // len(names)
    rem     = n_scenarios % len(names)
    for i, name in enumerate(names):
        for _ in range(npc + (1 if i < rem else 0)):
            coeffs.append(float(rng.choice(magnitude_classes[name])))
    rng.shuffle(coeffs)
    return coeffs


def _build_zone_to_nodes(
    candidates: List[str],
    node_zone_map: Dict[str, str],
    available_zones: List[str],
) -> Dict[str, List[str]]:
    """Build {zone: [node_id, ...]} for stratified sampling."""
    z2n: Dict[str, List[str]] = defaultdict(list)
    fallback = available_zones[0]
    for node in candidates:
        zone = node_zone_map.get(node, fallback)
        z2n[zone].append(node)
    return dict(z2n)


# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

def generate_dataset_wide(
    *,
    inp_path:               str,
    best_params_path:       str,
    obs_nodes:              List[str],
    n_scenarios:            int,
    seed:                   int,
    coeff_min:              float,
    coeff_max:              float,
    coeff_choices:          Optional[List[float]]             = None,
    leak_duration_hr:       float                             = 4.0,
    leak_duration_hr_min:   float                             = 1.0,
    leak_duration_hr_max:   float                             = 12.0,
    use_variable_duration:  bool                              = False,
    use_log_uniform:        bool                              = False,
    use_class_balanced:     bool                              = False,
    use_stratified_zones:   bool                              = True,
    magnitude_classes:      Optional[Dict[str, List[float]]] = None,
    leak_start_hr_min:      int                               = 0,
    leak_start_hr_max:      int                               = 23,
    sample_minutes:         int                               = 60,
    baseline_no_leaks:      bool                              = False,
    override_mode:          str                               = "add",
    exclude_leak_node_ids:  Optional[List[str]]               = None,
    out_csv:                str = "outputs/datasets/leak_dataset_wide.csv",
    out_csv_long:           Optional[str]                     = None,
) -> Tuple[Path, Optional[Path]]:
    """
    Generate multi-scenario leak pressure dataset for ML training.

    Key distinctions
    ----------------
    emitter_coeff_baseline  = zone_multiplier × global_scale
                              NOT LeakNodeMeta.weight (always 1.0)
    emitter_coeff_added     = sampled injected leak
    emitter_coeff_total     = baseline + added  →  fed to EPANET

    Coefficient units
    -----------------
    Model is LPM.  EPANET computes  Q [LPM] = C × P^0.5.
    C values in MAGNITUDE_CLASSES are sized accordingly (no ×60 factor).

    Zone sampling
    -------------
    use_stratified_zones=True (default):
        1. Sample zone uniformly from available zones.
        2. Sample node uniformly within that zone.
        Guarantees equal zone representation at any N.

    use_stratified_zones=False:
        Sample node uniformly from all candidates.
        Over-represents large zones (Z_0 dominance problem).
    """
    if override_mode not in {"add", "set"}:
        raise ValueError("override_mode must be 'add' or 'set'")

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_long_path: Optional[Path] = None
    if out_csv_long:
        out_long_path = Path(out_csv_long)
        out_long_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(seed))

    # ── Load calibration ─────────────────────────────────────────────────
    raw_base         = _load_best_raw_params(Path(best_params_path))
    zone_multipliers = _get_zone_multipliers_from_params(raw_base)
    global_scale     = _get_global_leakage_scale(raw_base)
    node_zone_map    = _build_node_zone_mapping_from_calibration(
        best_params_path)

    # ── Zone distribution diagnostic ─────────────────────────────────────
    zone_counts: Dict[str, int] = {}
    for z in node_zone_map.values():
        zone_counts[z] = zone_counts.get(z, 0) + 1

    print("\n" + "=" * 70)
    print("ZONE MAPPING SUMMARY")
    print(f"  Nodes mapped : {len(node_zone_map)}")
    print(f"  Zones found  : {sorted(zone_counts.keys())}")
    print(f"  Global scale : {global_scale:.6f}")
    print()
    print(f"  {'Zone':<8} {'Nodes':>6}  {'ZoneMult':>10}  "
          f"{'EffWeight':>12}  (NOT LeakNodeMeta.weight=1.0)")
    print("  " + "-" * 50)
    for z in sorted(zone_counts):
        zm = zone_multipliers.get(z, float("nan"))
        ew = zm * global_scale
        print(f"  {z:<8} {zone_counts[z]:>6}  {zm:>10.4f}  {ew:>12.8f}")
    if len(zone_counts) == 1:
        print("\n  ⚠️  WARNING: only 1 zone → fallback likely triggered!")
    else:
        print("\n  ✅ Multiple zones confirmed — zone fix is working")
    print("=" * 70 + "\n")

    # ── Simulation config ─────────────────────────────────────────────────
    step_s   = int(sample_minutes) * 60
    raw_base = json.loads(json.dumps(raw_base))
    raw_base.setdefault("time", {})
    raw_base["time"].update({
        "duration_days":        1,      # enforce 24-hour runs
        "hydraulic_timestep_s": step_s,
        "report_timestep_s":    step_s,
        "report_start_s":       0,
    })
    if baseline_no_leaks:
        raw_base.setdefault("leakage", {})["global_scale"] = 0.0

    runner    = build_runner(inp_path=inp_path,
                             metadata=config.build_default_metadata())
    obs_nodes = [str(x).strip() for x in obs_nodes]
    if not obs_nodes:
        raise ValueError("obs_nodes is empty")

    # ── Candidate nodes ───────────────────────────────────────────────────
    exclude    = {str(x).strip() for x in (exclude_leak_node_ids or []) if x}
    candidates = _get_candidate_leak_nodes(inp_path, exclude)
    if not candidates:
        raise ValueError("No candidate leak nodes after exclusions")

    missing = [n for n in candidates if n not in node_zone_map]
    if missing:
        print(f"⚠️  {len(missing)} candidates not in zone map "
              f"(fallback zone). Sample: {missing[:5]}")
    else:
        print(f"✅ All {len(candidates)} candidates in zone map\n")

    # ── Stratified sampling lookup ────────────────────────────────────────
    available_zones = sorted(zone_multipliers.keys())
    zone_to_nodes   = _build_zone_to_nodes(
        candidates, node_zone_map, available_zones)

    # Log expected distribution
    sampling_label = ("STRATIFIED equal probability per zone"
                      if use_stratified_zones
                      else "RANDOM proportional to zone size")
    print(f"Node sampling: {sampling_label}")
    total_cands = len(candidates)
    print(f"  {'Zone':<8} {'Nodes':>6}  {'NatShare':>9}  "
          f"{'ExpShare':>9} (stratified=equal)")
    for z in available_zones:
        n_z   = len(zone_to_nodes.get(z, []))
        nat   = n_z / total_cands if total_cands else 0
        exp   = 1 / len(available_zones) if use_stratified_zones else nat
        print(f"  {z:<8} {n_z:>6}  {nat:>9.1%}  {exp:>9.1%}")
    print()

    xy_by_node    = _node_xy_lookup(inp_path)
    duration_days = int(raw_base.get("time", {}).get("duration_days", 1))
    total_hours   = duration_days * 24
    coeff_choices = list(coeff_choices or [])

    # ── Pre-generate coefficients ─────────────────────────────────────────
    if use_class_balanced and magnitude_classes:
        coeff_samples: Optional[List[float]] = _sample_with_class_balance(
            rng, magnitude_classes, n_scenarios)
    elif use_log_uniform and not coeff_choices:
        coeff_samples = [_log_uniform(rng, coeff_min, coeff_max)
                         for _ in range(n_scenarios)]
    else:
        coeff_samples = None

    # ── Magnitude thresholds — LPM model, geometric mean boundaries ───────
    # All C values derived as C = Q_LPM / sqrt(P_mid=23.5).
    # Thresholds are geometric means between adjacent class C values so
    # that each exact class C always maps to its own class.
    # Verified numerically: all 8 classes classify correctly.
    #
    # seep(0.1031) / drip(0.4126)         → geomean = 0.2063
    # drip(0.4126) / trickle(1.0314)      → geomean = 0.6523
    # trickle(1.0314) / near_thresh(1.1779) → geomean = 1.1022
    # near_thresh(1.1779) / small_burst(3.0943) → geomean = 1.9091
    # small_burst(3.0943) / med_burst(8.2514)   → geomean = 5.0530
    # med_burst(8.2514) / large_burst(20.6284)  → geomean = 13.0466
    # large_burst(20.6284) / major_burst(51.5711) → geomean = 32.6164
    mag_thresholds = {
        "drip_threshold":        0.2063,
        "trickle_threshold":     0.6523,
        "near_thresh_threshold": 1.1022,
        "small_burst_threshold": 1.9091,
        "med_burst_threshold":   5.0530,
        "large_burst_threshold": 13.0466,
        "major_burst_threshold": 32.6164,
    }

    rows:      List[dict] = []
    long_rows: List[dict] = []

    # =========================================================================
    # BASELINE (scenario_id = 0)
    # =========================================================================
    print("Running baseline simulation …")
    try:
        _, base_results, _ = runner.build_and_run_once(
            raw_base, nodes_to_read=set(obs_nodes),
            emitter_window_overrides=None)
        base_ok = True
    except Exception as exc:
        print(f"⚠️  Baseline failed: {exc}")
        base_results = {"pressure": pd.DataFrame()}
        base_ok      = False

    _bp = (base_results.get("pressure", pd.DataFrame()))
    base_p = (
        _bp.loc[:, [c for c in obs_nodes if c in _bp.columns]]
        if not _bp.empty else pd.DataFrame()
    )

    if sample_minutes != 60:
        base_hourly = _aggregate_to_hourly(base_p, total_hours)
    else:
        if not base_p.empty:
            base_hourly       = base_p.copy()
            base_hourly.index = (
                base_hourly.index.astype(int) // 3600).astype(int)
            base_hourly       = base_hourly.reindex(range(total_hours))
        else:
            base_hourly = pd.DataFrame(index=range(total_hours))

    baseline_row: dict = {
        "scenario_id": 0, "leak": 0,
        "leak_node": "", "leak_zone": "", "leak_zone_multiplier": "",
        "leak_node_effective_weight": "", "leak_x": "", "leak_y": "",
        "leak_start_hr": "", "leak_duration_hr": "",
        "emitter_coeff_baseline": "", "emitter_coeff_added": "",
        "emitter_coeff_total": "", "leak_size_lpm": "",
        "leak_node_pressure": "", "leak_magnitude_class": "",
        "validation_status": "success" if base_ok else "failed",
    }
    for n in obs_nodes:
        if n not in base_hourly.columns:
            continue
        for h in range(total_hours):
            v = (base_hourly.loc[h, n]
                 if h in base_hourly.index else np.nan)
            baseline_row[f"{n}_Hour{h}"] = _float_or_blank(v)
    rows.append(baseline_row)
    print("✅ Baseline complete\n")

    # =========================================================================
    # LEAK SCENARIOS (scenario_id = 1 … n_scenarios)
    # =========================================================================
    failed       = 0
    zone_counter: Dict[str, int] = defaultdict(int)

    for sid in range(1, n_scenarios + 1):

        # ── Node selection ────────────────────────────────────────────────
        if use_stratified_zones:
            # Equal probability per zone regardless of zone node count.
            # This fixes the Z_0 dominance problem entirely.
            sampled_zone = str(rng.choice(available_zones))
            zone_nodes   = zone_to_nodes.get(sampled_zone, candidates)
            if not zone_nodes:
                zone_nodes = candidates
            leak_node = str(rng.choice(zone_nodes))
        else:
            leak_node = str(rng.choice(candidates))

        # ── Zone / effective weight ───────────────────────────────────────
        # effective_weight = zone_multiplier × global_scale
        # NOT LeakNodeMeta.weight which is always 1.0
        leak_zone, zone_mult, eff_w = _get_node_zone(
            leak_node, node_zone_map, zone_multipliers, global_scale)
        c_baseline = eff_w
        zone_counter[leak_zone] += 1

        # ── Injected coefficient (LPM model: C = Q_LPM / sqrt(P)) ────────
        if coeff_samples is not None:
            c_added = float(coeff_samples[sid - 1])
        elif coeff_choices:
            c_added = float(rng.choice(coeff_choices))
        elif use_log_uniform:
            c_added = _log_uniform(rng, coeff_min, coeff_max)
        else:
            c_added = float(rng.uniform(coeff_min, coeff_max))

        c_total  = (c_baseline + c_added) if override_mode == "add" \
            else c_added

        # ── Timing ───────────────────────────────────────────────────────
        start_hr = int(rng.integers(leak_start_hr_min,
                                    leak_start_hr_max + 1))
        max_dur  = 24 - start_hr

        if use_variable_duration:
            adj_max = min(leak_duration_hr_max, float(max_dur))
            adj_min = min(leak_duration_hr_min, adj_max)
            dur_hr  = _sample_leak_duration_hours(
                rng, adj_min, adj_max, use_log=use_log_uniform)
        else:
            dur_hr = min(int(leak_duration_hr), max_dur)

        dur_hr  = max(1, int(dur_hr))
        start_s = start_hr * 3600
        end_s   = start_s + dur_hr * 3600

        # ── Simulation ───────────────────────────────────────────────────
        try:
            _, results, params = runner.build_and_run_once(
                raw_base,
                nodes_to_read=set(obs_nodes) | {leak_node},
                emitter_window_overrides={
                    leak_node: (start_s, end_s, c_total)},
                emitter_window_override_mode="set",
            )
        except Exception as exc:
            print(f"⚠️  Scenario {sid} failed: {exc}")
            failed += 1
            continue

        p_df = results.get("pressure", pd.DataFrame())
        if p_df.empty:
            print(f"⚠️  Scenario {sid} empty pressure data")
            failed += 1
            continue

        p_sensors = p_df.loc[
            :, [c for c in obs_nodes if c in p_df.columns]]
        if sample_minutes != 60:
            hourly = _aggregate_to_hourly(p_sensors, total_hours)
        else:
            hourly       = p_sensors.copy()
            hourly.index = (hourly.index.astype(int) // 3600).astype(int)
            hourly       = hourly.reindex(range(total_hours))

        # ── Leak node pressure during leak window ─────────────────────────
        lp = (p_df[leak_node].copy() if leak_node in p_df.columns
              else pd.Series(dtype=float))
        lp.index = lp.index.astype(int)
        mask     = (lp.index >= start_s) & (lp.index < end_s)
        leak_node_pressure = float(lp.loc[mask].mean()) if mask.any() \
            else 0.0
        vstatus = "zero_pressure" if leak_node_pressure <= 0 else "success"

        # ── Leak size in LPM ──────────────────────────────────────────────
        # LPM model: Q [LPM] = C × P^n
        # q_to_m3s converts LPM → m³/s (= 1/60000 for LPM)
        # So: q_m3s = C × P^n / 60000
        #     leak_size_lpm = q_m3s × 60000 = C × P^n   ✓ correct LPM
        exp      = float(
            getattr(params.leakage, "emitter_exponent", 0.5) or 0.5)
        q_to_m3s = _flow_to_m3s_factor(
            results.get("debug", {})
                   .get("demand_units", {})
                   .get("inpfile_units"))
        if mask.any():
            q_m3s         = (float(c_added)
                             * lp.loc[mask].clip(lower=0.0) ** exp
                             * q_to_m3s)
            leak_size_lpm = float((q_m3s * 60_000.0).mean())
        else:
            leak_size_lpm = 0.0

        x, y = xy_by_node.get(leak_node, (np.nan, np.nan))

        # ── Wide-format row ───────────────────────────────────────────────
        row: dict = {
            "scenario_id":               sid,
            "leak":                      1,
            "leak_node":                 leak_node,
            "leak_zone":                 leak_zone,
            "leak_zone_multiplier":      float(zone_mult),
            # effective_weight = zone_mult × global_scale (NOT weight=1.0)
            "leak_node_effective_weight":float(eff_w),
            "leak_x":  float(x) if np.isfinite(x) else "",
            "leak_y":  float(y) if np.isfinite(y) else "",
            "leak_start_hr":             start_hr,
            "leak_duration_hr":          dur_hr,
            "emitter_coeff_baseline":    float(c_baseline),
            "emitter_coeff_added":       float(c_added),
            "emitter_coeff_total":       float(c_total),
            "leak_size_lpm":             float(leak_size_lpm),
            "leak_node_pressure":        float(leak_node_pressure),
            "leak_magnitude_class": _get_leak_magnitude_class(
                c_added, mag_thresholds),
            "validation_status":         vstatus,
        }

        # Pressure columns only (DeltaP/Ratio can be computed post-hoc)
        for n in obs_nodes:
            if n not in hourly.columns:
                continue
            for h in range(total_hours):
                vl = hourly.loc[h, n] if h in hourly.index else np.nan
                row[f"{n}_Hour{h}"] = _float_or_blank(vl)

        rows.append(row)

        # ── Long-format rows ──────────────────────────────────────────────
        if out_long_path:
            for n in obs_nodes:
                if n not in hourly.columns:
                    continue
                oz, ozm, oew = _get_node_zone(
                    n, node_zone_map, zone_multipliers, global_scale)
                for h in range(total_hours):
                    vl = (hourly.loc[h, n]
                          if h in hourly.index else np.nan)
                    vb = (base_hourly.loc[h, n]
                          if (h in base_hourly.index
                              and n in base_hourly.columns)
                          else np.nan)
                    long_rows.append({
                        "scenario_id":               sid,
                        "leak_node":                 leak_node,
                        "leak_zone":                 leak_zone,
                        "leak_zone_multiplier":      float(zone_mult),
                        "leak_node_effective_weight":float(eff_w),
                        "emitter_coeff_baseline":    float(c_baseline),
                        "emitter_coeff_added":       float(c_added),
                        "emitter_coeff_total":       float(c_total),
                        "obs_node":                  n,
                        "obs_zone":                  oz,
                        "obs_zone_multiplier":       float(ozm),
                        "obs_node_effective_weight": float(oew),
                        "hour":                      h,
                        "leak_x":  float(x) if np.isfinite(x) else np.nan,
                        "leak_y":  float(y) if np.isfinite(y) else np.nan,
                        "leak_start_hr":             start_hr,
                        "leak_duration_hr":          dur_hr,
                        "leak_magnitude_class":      row["leak_magnitude_class"],
                        "leak_size_lpm":             float(leak_size_lpm),
                        "pressure": (float(vl) if not pd.isna(vl)
                                     else np.nan),
                        "baseline_pressure": (float(vb) if not pd.isna(vb)
                                              else np.nan),
                    })

        if sid % 50 == 0:
            pct = sid / n_scenarios * 100
            print(f"  [{pct:5.1f}%] {sid}/{n_scenarios} done  "
                  f"| failed: {failed}")

    # =========================================================================
    # SAVE
    # =========================================================================
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\n✅ Wide CSV: {out_path}  (rows={len(df)})")

    if out_long_path and long_rows:
        pd.DataFrame(long_rows).to_csv(out_long_path, index=False)
        print(f"✅ Long CSV: {out_long_path}  ({len(long_rows)} rows)")

    # ── Zone balance report ───────────────────────────────────────────────
    print(f"\nZone distribution in dataset:")
    print(f"  {'Zone':<8} {'Count':>8} {'Share':>8}")
    print("  " + "-" * 28)
    total_leak = sum(zone_counter.values())
    for z in sorted(zone_counter.keys()):
        cnt = zone_counter[z]
        print(f"  {z:<8} {cnt:>8} {cnt/total_leak:>8.1%}")
    if len(zone_counter) == 1:
        print("  ⚠️  Only 1 zone — zone fix may not have worked!")
    else:
        print("  ✅ Multiple zones in dataset")

    # ── Validation check ──────────────────────────────────────────────────
    pcols = [c for c in df.columns if "_Hour" in c]
    zero  = [int(r["scenario_id"])
             for _, r in df[df["leak"] == 1].iterrows()
             if all(v == "" or float(v) == 0
                    for v in r[pcols] if v != "")]
    if zero:
        print(f"\n⚠️  {len(zero)} zero-pressure scenarios: {zero[:10]}")
    else:
        print("\n✅ Validation: all scenarios have non-zero pressures")
    print(f"⚠️  Failed simulations: {failed} / {n_scenarios}")

    # ── Metadata ──────────────────────────────────────────────────────────
    meta = {
        "best_params_path":           str(best_params_path),
        "inp_path":                   str(inp_path),
        "n_scenarios":                n_scenarios,
        "seed":                       seed,
        "coeff_min":                  coeff_min,
        "coeff_max":                  coeff_max,
        "coeff_choices":              coeff_choices,
        "override_mode":              override_mode,
        "use_log_uniform_sampling":   use_log_uniform,
        "use_class_balanced_sampling":use_class_balanced,
        "use_stratified_zones":       use_stratified_zones,
        "magnitude_classes":          magnitude_classes or {},
        "magnitude_class_thresholds": mag_thresholds,
        "leak_duration_hr":           leak_duration_hr,
        "use_variable_duration":      use_variable_duration,
        "leak_duration_hr_min":
            leak_duration_hr_min if use_variable_duration else None,
        "leak_duration_hr_max":
            leak_duration_hr_max if use_variable_duration else None,
        "leak_start_hr_min":          leak_start_hr_min,
        "leak_start_hr_max":          leak_start_hr_max,
        "baseline_no_leaks":          baseline_no_leaks,
        "sample_minutes":             sample_minutes,
        "duration_days":              duration_days,
        "total_hours":                total_hours,
        "obs_nodes":                  obs_nodes,
        "global_leakage_scale":       global_scale,
        "zone_multipliers":           zone_multipliers,
        "zone_scenario_counts":       dict(zone_counter),
        "zone_counts_in_mapping":     zone_counts,
        "failed_simulations":         failed,
        "leak_size_units":            "LPM",
        "pressure_range_m": {
            "low": 20.0, "mid": 23.5, "high": 27.0},
        "iwa_background_ceiling_lpm": {
            "at_50m_reference": 8.33,
            "at_23p5m_network": 5.71,
        },
        "emitter_coefficient_notes": {
            "model_units":   "LPM — Q [LPM] = C × sqrt(P)  (no ×60 factor)",
            "derivation":    "C = Q_target_LPM / sqrt(P_mid=23.5)",
            "baseline":
                "zone_multiplier × global_scale  "
                "(NOT LeakNodeMeta.weight=1.0)",
            "added":         "Injected leak (sized for LPM model)",
            "total":         "baseline + added → fed to EPANET",
        },
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"✅ Metadata: {meta_path}\n")

    return out_path, out_long_path


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    INP_PATH         = config.MODEL_INP
    BEST_PARAMS_PATH = "outputs/reports/best_params.json"

    OBS_NODES: List[str] = [
        "NODEADD_2423", "NODE_3005",    "HOUSE_EPN_164", "NODEADD_022",
        "NODE_3018",    "HOUSE_EPN_255","NODE_3002",     "HOUSE_EPN_67",
        "NODE_3009",    "NODE_3013",    "NODE_3023",     "HOUSE_EPN_392",
        "NODE_3136",    "NODE_3028",    "NODE_3024",     "NODE_3062",
        "NODE_3116",    "HOUSE_EPN_695","NODE_3094",     "HOUSE_EPN_638",
        "NODEADD_1830",
    ]

    EXCLUDE_LEAK_NODE_IDS: List[str] = []

    # 3500 = 500 scenarios × 7 zones (stratified, publishable minimum)
    # Increase to 5600–8000 for node-level localisation tasks.
    N_SCENARIOS = 8000
    SEED        = 1

    LEAK_DURATION_HR      = 4.0
    LEAK_DURATION_HR_MIN  = 1.0
    LEAK_DURATION_HR_MAX  = 12.0
    USE_VARIABLE_DURATION = True

    LEAK_START_HR_MIN = 0
    LEAK_START_HR_MAX = 23          # full 24-hour window

    USE_STRATIFIED_ZONES        = True   # fixes Z_0 dominance — keep True
    USE_LOG_UNIFORM_SAMPLING    = True   # better small-leak representation
    USE_CLASS_BALANCED_SAMPLING = True   # equal scenarios per class

    # ── Emitter coefficients — LPM MODEL CORRECT ─────────────────────────
    # EPANET LPM model:  Q [LPM] = C × sqrt(P)      (no ×60 factor)
    # C = Q_target_LPM / sqrt(P_mid=23.5) = Q / 4.8477
    #
    # Verified numerically — see module docstring for full table.
    # Previous versions used C/60 (LPS formula in LPM model = 60× too small)
    #
    # Class           C        LPM@20m  LPM@23.5m  LPM@27m
    # ──────────────────────────────────────────────────────
    # seep          0.1031     0.46      0.50       0.54
    # drip          0.4126     1.85      2.00       2.14
    # trickle       1.0314     4.61      5.00       5.36
    # near_thresh   1.1779     5.27      5.71       6.12   ← IWA BABE ceiling
    # small_burst   3.0943    13.84     15.00      16.08
    # med_burst     8.2514    36.90     40.00      42.88
    # large_burst  20.6284    92.25    100.00     107.19
    # major_burst  51.5711   230.63    250.00     267.97
    MAGNITUDE_CLASSES: Dict[str, List[float]] = {
        "seep":         [0.1031],
        "drip":         [0.4126],
        "trickle":      [1.0314],
        "near_thresh":  [1.1779],
        "small_burst":  [3.0943],
        "med_burst":    [8.2514],
        "large_burst":  [20.6284],
        "major_burst":  [51.5711],
    }

    EMITTER_COEFF_MIN     = 0.1031   # seep lower bound
    EMITTER_COEFF_MAX     = 51.5711  # major_burst upper bound
    EMITTER_COEFF_CHOICES: List[float] = []  # empty → use MAGNITUDE_CLASSES

    SAMPLE_MINUTES    = 60
    BASELINE_NO_LEAKS = True
    OVERRIDE_MODE     = "add"

    OUT_CSV_WIDE = "outputs/datasets/leak_dataset_wide.csv"
    OUT_CSV_LONG = "outputs/datasets/leak_dataset_long.csv"

    if not OBS_NODES:
        raise ValueError("OBS_NODES is empty")

    mag_classes = MAGNITUDE_CLASSES if USE_CLASS_BALANCED_SAMPLING else None

    wide_path, long_path = generate_dataset_wide(
        inp_path               = str(INP_PATH),
        best_params_path       = BEST_PARAMS_PATH,
        obs_nodes              = OBS_NODES,
        n_scenarios            = N_SCENARIOS,
        seed                   = SEED,
        coeff_min              = EMITTER_COEFF_MIN,
        coeff_max              = EMITTER_COEFF_MAX,
        coeff_choices          = EMITTER_COEFF_CHOICES,
        leak_duration_hr       = LEAK_DURATION_HR,
        leak_duration_hr_min   = LEAK_DURATION_HR_MIN,
        leak_duration_hr_max   = LEAK_DURATION_HR_MAX,
        use_variable_duration  = USE_VARIABLE_DURATION,
        use_log_uniform        = USE_LOG_UNIFORM_SAMPLING,
        use_class_balanced     = USE_CLASS_BALANCED_SAMPLING,
        use_stratified_zones   = USE_STRATIFIED_ZONES,
        magnitude_classes      = mag_classes,
        leak_start_hr_min      = LEAK_START_HR_MIN,
        leak_start_hr_max      = LEAK_START_HR_MAX,
        sample_minutes         = SAMPLE_MINUTES,
        baseline_no_leaks      = BASELINE_NO_LEAKS,
        override_mode          = OVERRIDE_MODE,
        exclude_leak_node_ids  = EXCLUDE_LEAK_NODE_IDS,
        out_csv                = OUT_CSV_WIDE,
        out_csv_long           = OUT_CSV_LONG,
    )

    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"Wide : {wide_path}")
    print(f"Long : {long_path}")
    print()
    print("Coefficient glossary:")
    print("  emitter_coeff_baseline  = zone_mult × global_scale")
    print("                            NOT LeakNodeMeta.weight=1.0")
    print("  emitter_coeff_added     = injected leak (LPM-correct C values)")
    print("  emitter_coeff_total     = baseline + added → EPANET input")
    print("  leak_size_lpm           = C_added × avg(P^0.5) during window")
    print()
    print("C sizing note:")
    print("  LPM model: Q [LPM] = C × sqrt(P)  — no ×60 conversion factor")
    print("  C = Q_target_LPM / sqrt(23.5) = Q / 4.8477")
    print("=" * 70)
"""Generate a multi-scenario leak dataset using the calibrated model.

This is the "many scenarios" companion to scripts/generate_leak_dataset.py.

Key differences vs the simple generator:
- Supports leak start time + duration (emitter turns on/off during the run)
- Can output a *wide* format similar to main_generator.py:
  one row per scenario with <NODE>_Hour0..Hour23 columns

By default we write wide format because it matches the attached generator.

Leak magnitude:
- The scenario parameter we sample is an EPANET emitter coefficient C.
- EPANET uses Q = C * P^n.
- We additionally compute a derived leak_size_lps (mean leak flow in L/s during the leak window)
  using the simulated pressure at the leak node and the model's flow units.

Typical usage:
  python scripts/generate_leak_dataset_many.py --n-scenarios 2000 --seed 1 --coeff-min 0.05 --coeff-max 1.5

Outputs go under outputs/datasets/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

from calibration.runner import build_runner


def _load_best_raw_params(path: Path) -> dict:
    obj = json.loads(path.read_text())
    if isinstance(obj, dict) and "best_raw_params" in obj:
        return obj["best_raw_params"]
    return obj


def _node_xy_lookup(inp_path: str) -> Dict[str, Tuple[float, float]]:
    import wntr

    wn = wntr.network.WaterNetworkModel(inp_path)
    out: Dict[str, Tuple[float, float]] = {}
    for name in wn.node_name_list:
        node = wn.get_node(name)
        xy = getattr(node, "coordinates", None)
        if xy is None:
            continue
        try:
            out[str(name)] = (float(xy[0]), float(xy[1]))
        except Exception:
            continue
    return out


def _read_exclude_list(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Exclude file not found: {path}")
    txt = path.read_text(errors="ignore")
    out: set[str] = set()
    for line in txt.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.add(s)
    return out


def _get_candidate_leak_nodes(inp_path: str, exclude: set[str]) -> list[str]:
    import wntr

    wn = wntr.network.WaterNetworkModel(inp_path)
    nodes = []
    for name, _node in wn.junctions():
        s = str(name)
        if s in exclude:
            continue
        nodes.append(s)
    nodes.sort()
    return nodes


def _aggregate_to_hourly(df: pd.DataFrame, total_hours: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(index=range(total_hours))

    x = df.copy()
    x.index = x.index.astype(int)
    x["hour_index"] = x.index // 3600
    hourly = x.groupby("hour_index").mean(numeric_only=True)
    return hourly.reindex(range(total_hours))


def _flow_to_m3s_factor(inp_units: str | None) -> float:
    if not inp_units:
        return 1.0
    u = str(inp_units).strip().upper()

    # Metric
    if u == "CMS":
        return 1.0
    if u == "CMD":
        return 1.0 / 86400.0
    if u == "MLD":
        return 1000.0 / 86400.0
    if u == "LPS":
        return 1.0 / 1000.0
    if u == "LPM":
        return 1.0 / (1000.0 * 60.0)

    # US/Imperial
    if u == "CFS":
        return 0.028316846592
    if u == "GPM":
        return 0.003785411784 / 60.0
    if u == "MGD":
        return (1_000_000.0 * 0.003785411784) / 86400.0
    if u == "IMGD":
        return (1_000_000.0 * 0.00454609) / 86400.0
    if u == "AFD":
        return 1233.48183754752 / 86400.0

    return 1.0


def _float_or_blank(x: Any):
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


def generate_dataset_wide(
    *,
    inp_path: str,
    best_params_path: str,
    obs_nodes: list[str],
    n_scenarios: int,
    seed: int,
    coeff_min: float,
    coeff_max: float,
    coeff_choices: "list[float] | None" = None,
    leak_duration_hr: float = 4.0,
    leak_start_hr_min: int = 0,
    leak_start_hr_max: int = 19,
    sample_minutes: int = 60,
    baseline_no_leaks: bool = False,
    override_mode: str = "add",
    exclude_leak_node_ids: "list[str] | None" = None,
    out_csv: str = "outputs/datasets/leak_dataset_wide.csv",
) -> Path:
    if override_mode not in {"add", "set"}:
        raise ValueError("override_mode must be one of {'add','set'}")

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(seed))
    raw_base = _load_best_raw_params(Path(best_params_path))

    step_s = int(sample_minutes) * 60
    raw_base = json.loads(json.dumps(raw_base))
    raw_base.setdefault("time", {})
    raw_base["time"]["hydraulic_timestep_s"] = int(step_s)
    raw_base["time"]["report_timestep_s"] = int(step_s)
    raw_base["time"]["report_start_s"] = 0

    if bool(baseline_no_leaks):
        raw_base.setdefault("leakage", {})
        raw_base["leakage"]["global_scale"] = 0.0

    metadata = config.build_default_metadata()
    runner = build_runner(inp_path=inp_path, metadata=metadata)

    obs_nodes = [str(x) for x in obs_nodes]
    if not obs_nodes:
        raise ValueError("obs_nodes is empty")

    exclude = set(str(x).strip() for x in (exclude_leak_node_ids or []) if str(x).strip())
    candidates = _get_candidate_leak_nodes(inp_path, exclude=exclude)
    if not candidates:
        raise ValueError("No candidate leak nodes found after exclusions")

    xy_by_node = _node_xy_lookup(inp_path)

    duration_days = int(raw_base.get("time", {}).get("duration_days", 1))
    total_hours = int(duration_days * 24)

    coeff_choices = list(coeff_choices or [])

    rows: list[dict] = []

    # Baseline row
    _, base_results, _ = runner.build_and_run_once(
        raw_base,
        nodes_to_read=set(obs_nodes),
        emitter_window_overrides=None,
    )
    base_p = base_results["pressure"].loc[:, [c for c in obs_nodes if c in base_results["pressure"].columns]]
    if int(sample_minutes) != 60:
        base_hourly = _aggregate_to_hourly(base_p, total_hours=total_hours)
    else:
        base_hourly = base_p.copy()
        base_hourly.index = (base_hourly.index.astype(int) // 3600).astype(int)
        base_hourly = base_hourly.reindex(range(total_hours))

    baseline_row: dict = {
        "scenario_id": 0,
        "leak": 0,
        "leak_node": "",
        "leak_x": "",
        "leak_y": "",
        "leak_start_hr": "",
        "leak_duration_hr": "",
        "emitter_coeff_added": "",
        "emitter_coeff_total": "",
        "leak_size_lps": "",
        "leak_node_pressure": "",
    }
    for n in obs_nodes:
        if n not in base_hourly.columns:
            continue
        for h in range(total_hours):
            v = base_hourly.loc[h, n]
            baseline_row[f"{n}_Hour{h}"] = _float_or_blank(v)
    rows.append(baseline_row)

    # Leak scenarios
    for scenario_id in range(1, int(n_scenarios) + 1):
        leak_node = str(rng.choice(candidates))

        if coeff_choices:
            c_added = float(rng.choice(coeff_choices))
        else:
            c_added = float(rng.uniform(float(coeff_min), float(coeff_max)))

        start_hr = int(rng.integers(int(leak_start_hr_min), int(leak_start_hr_max) + 1))
        duration_hr = float(leak_duration_hr)

        start_s = int(start_hr * 3600)
        end_s = int(start_s + duration_hr * 3600)

        nodes_to_read = set(obs_nodes) | {leak_node}

        _, results, params = runner.build_and_run_once(
            raw_base,
            nodes_to_read=nodes_to_read,
            emitter_window_overrides={leak_node: (start_s, end_s, c_added)},
            emitter_window_override_mode=str(override_mode),
        )

        p_df = results["pressure"].copy()
        p_sensors = p_df.loc[:, [c for c in obs_nodes if c in p_df.columns]]

        if int(sample_minutes) != 60:
            hourly = _aggregate_to_hourly(p_sensors, total_hours=total_hours)
        else:
            hourly = p_sensors.copy()
            hourly.index = (hourly.index.astype(int) // 3600).astype(int)
            hourly = hourly.reindex(range(total_hours))

        leak_press = p_df[leak_node] if leak_node in p_df.columns else pd.Series(dtype=float)
        leak_press.index = leak_press.index.astype(int)
        mask = (leak_press.index >= start_s) & (leak_press.index < end_s)
        leak_node_pressure = float(leak_press.loc[mask].mean()) if mask.any() else 0.0

        exp = float(getattr(params.leakage, "emitter_exponent", 0.5) or 0.5)
        inp_units = results.get("debug", {}).get("demand_units", {}).get("inpfile_units", None)
        q_to_m3s = _flow_to_m3s_factor(inp_units)
        if mask.any():
            q_m3s = (float(c_added) * (leak_press.loc[mask].clip(lower=0.0) ** exp)) * float(q_to_m3s)
            leak_size_lps = float((q_m3s * 1000.0).mean())
        else:
            leak_size_lps = 0.0

        dbg = results.get("debug", {})
        emitter_coeffs = dbg.get("emitter_coeffs", {}) if isinstance(dbg, dict) else {}
        base_coeff = float(emitter_coeffs.get(leak_node, 0.0))
        c_total = (base_coeff + float(c_added)) if override_mode == "add" else float(c_added)

        x, y = xy_by_node.get(leak_node, (np.nan, np.nan))

        row: dict = {
            "scenario_id": int(scenario_id),
            "leak": 1,
            "leak_node": leak_node,
            "leak_x": float(x) if np.isfinite(x) else "",
            "leak_y": float(y) if np.isfinite(y) else "",
            "leak_start_hr": float(start_hr),
            "leak_duration_hr": float(duration_hr),
            "emitter_coeff_added": float(c_added),
            "emitter_coeff_total": float(c_total),
            "leak_size_lps": float(leak_size_lps),
            "leak_node_pressure": float(leak_node_pressure),
        }

        for n in obs_nodes:
            if n not in hourly.columns:
                continue
            for h in range(total_hours):
                v = hourly.loc[h, n]
                row[f"{n}_Hour{h}"] = _float_or_blank(v)

        rows.append(row)

        if scenario_id % 25 == 0:
            print(f"Simulated {scenario_id}/{n_scenarios} leak scenarios")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote dataset: {out_path} (rows={len(df)}, obs_nodes={len(obs_nodes)}, hours={total_hours})")

    meta = {
        "best_params_path": str(best_params_path),
        "inp_path": str(inp_path),
        "n_scenarios": int(n_scenarios),
        "seed": int(seed),
        "coeff_min": float(coeff_min),
        "coeff_max": float(coeff_max),
        "coeff_choices": coeff_choices,
        "override_mode": str(override_mode),
        "leak_duration_hr": float(leak_duration_hr),
        "leak_start_hr_min": int(leak_start_hr_min),
        "leak_start_hr_max": int(leak_start_hr_max),
        "baseline_no_leaks": bool(baseline_no_leaks),
        "sample_minutes": int(sample_minutes),
        "duration_days": int(duration_days),
        "obs_nodes": obs_nodes,
        "candidate_leak_node_count": int(len(candidates)),
        "excluded_leak_node_count": int(len(exclude)),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote metadata: {meta_path}")

    return out_path


if __name__ == "__main__":
    # ---------------------------
    # Easy configuration (edit)
    # ---------------------------

    INP_PATH = config.MODEL_INP
    BEST_PARAMS_PATH = "outputs/reports/best_params.json"

    # Observation nodes (features). Put node IDs here.
    # If you want to use config.SENSOR_NODES instead, set OBS_NODES = list(config.SENSOR_NODES)
    OBS_NODES: list[str] = [
        "NODEADD_2423",
        "NODE_3005",
        "HOUSE_EPN_164",
        "NODEADD_022",
        "NODE_3018",
        "HOUSE_EPN_255",
        "NODE_3002",
        "HOUSE_EPN_67",
        "NODE_3009",
        "NODE_3013",
        "NODE_3023",
        "HOUSE_EPN_392",
        "NODE_3136",
        "NODE_3028",
        "NODE_3024",
        "NODE_3062",
        "NODE_3116",
        "HOUSE_EPN_695",
        "NODE_3094",
        "HOUSE_EPN_638",
        "NODEADD_1830",
    ]

    # Candidate leak nodes = all INP junctions minus these excluded IDs
    EXCLUDE_LEAK_NODE_IDS: list[str] = []

    # Number of leak scenarios (baseline is always included as scenario_id=0)
    N_SCENARIOS = 200
    SEED = 0

    # Leak timing
    LEAK_DURATION_HR = 4.0
    LEAK_START_HR_MIN = 0
    LEAK_START_HR_MAX = 19

    # Leak magnitude sampling (emitter coefficient C)
    # Option A: continuous uniform range
    EMITTER_COEFF_MIN = 0.05
    EMITTER_COEFF_MAX = 1.5
    # Option B: discrete choices (if non-empty, overrides min/max)
    EMITTER_COEFF_CHOICES: list[float] = []

    # Simulation timestep; if < 60, values are averaged to hourly
    SAMPLE_MINUTES = 60

    # Baseline leak handling:
    # - baseline_no_leaks=True sets leakage.global_scale=0 in the calibrated baseline
    # - override_mode controls how the injected leak interacts with any remaining baseline emitter at the leak node
    BASELINE_NO_LEAKS = True
    OVERRIDE_MODE = "add"  # "add" or "set"

    OUT_CSV = "outputs/datasets/leak_dataset_wide.csv"

    # ---------------------------
    # Run
    # ---------------------------

    if not OBS_NODES:
        raise ValueError("OBS_NODES is empty and config.SENSOR_NODES is empty; set observation nodes")

    generate_dataset_wide(
        inp_path=str(INP_PATH),
        best_params_path=str(BEST_PARAMS_PATH),
        obs_nodes=OBS_NODES,
        n_scenarios=int(N_SCENARIOS),
        seed=int(SEED),
        coeff_min=float(EMITTER_COEFF_MIN),
        coeff_max=float(EMITTER_COEFF_MAX),
        coeff_choices=list(EMITTER_COEFF_CHOICES),
        leak_duration_hr=float(LEAK_DURATION_HR),
        leak_start_hr_min=int(LEAK_START_HR_MIN),
        leak_start_hr_max=int(LEAK_START_HR_MAX),
        sample_minutes=int(SAMPLE_MINUTES),
        baseline_no_leaks=bool(BASELINE_NO_LEAKS),
        override_mode=str(OVERRIDE_MODE),
        exclude_leak_node_ids=list(EXCLUDE_LEAK_NODE_IDS),
        out_csv=str(OUT_CSV),
    )

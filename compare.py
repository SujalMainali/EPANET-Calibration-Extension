"""Compare simulated vs observed sensor pressures and compute objective.

This is a "main-like" entry script:
- reads settings only from config.py
- runs the simulation once
- loads observed pressures at sensor junctions
- aligns observed and simulated time indices
- prints + saves sensor-level fit metrics
- computes the composite objective using calibration/objective.py

Configure:
- OBSERVED_PRESSURE_CSV (required)
- SENSOR_NODES (recommended)
- MODEL_INP (as usual)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

import config
from calibration.objective import load_observed_pressure_csv
from calibration.runner import RunResults, build_runner


def _ensure_dirs() -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    config.DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _coerce_index_to_seconds(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.dtype.kind in {"i", "u", "f"}:
        df.index = pd.Index(pd.to_numeric(df.index.to_numpy()))
        return df

    dt = pd.to_datetime(df.index)
    t0 = dt.min()
    seconds = (dt - t0).total_seconds()
    df.index = pd.Index(pd.to_numeric(np.asarray(seconds)))
    return df


def _align_sim_to_obs_index(obs: pd.DataFrame, sim: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = obs.sort_index()
    sim = sim.sort_index()

    sim_r = sim.reindex(obs.index)
    try:
        sim_r = sim_r.interpolate(method="index", limit_direction="both")
    except Exception:
        sim_r = sim_r.ffill().bfill()

    return obs, sim_r


def _rmse(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a**2))) if a.size else 0.0


def _mae(a: np.ndarray) -> float:
    return float(np.mean(np.abs(a))) if a.size else 0.0


def _compute_sensor_metrics(obs: pd.DataFrame, sim: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for col in obs.columns:
        o = obs[col].to_numpy(dtype=float)
        s = sim[col].to_numpy(dtype=float)
        mask = np.isfinite(o) & np.isfinite(s)
        e = (s - o)[mask]

        rows.append(
            {
                "sensor": str(col),
                "rmse": _rmse(e),
                "mae": _mae(e),
                "bias_mean": float(np.mean(e)) if e.size else 0.0,
                "max_abs_error": float(np.max(np.abs(e))) if e.size else 0.0,
                "n": float(e.size),
            }
        )

    return pd.DataFrame(rows).sort_values("rmse", ascending=False)


def _load_observed_from_config() -> pd.DataFrame:
    if not config.OBSERVED_PRESSURE_CSV:
        raise ValueError(
            "Set OBSERVED_PRESSURE_CSV in config.py to the observed pressure CSV path."
        )

    if config.OBSERVED_TIME_COLUMN is None:
        return load_observed_pressure_csv(config.OBSERVED_PRESSURE_CSV)

    df = pd.read_csv(config.OBSERVED_PRESSURE_CSV)
    if config.OBSERVED_TIME_COLUMN not in df.columns:
        raise ValueError(
            f"OBSERVED_TIME_COLUMN={config.OBSERVED_TIME_COLUMN!r} not found in observed CSV columns."
        )

    time_col = config.OBSERVED_TIME_COLUMN
    df = df.set_index(time_col)

    try:
        df.index = pd.Index(pd.to_numeric(df.index.to_numpy()))
    except Exception:
        df.index = pd.to_datetime(df.index)

    return df


def main() -> None:
    _ensure_dirs()

    obs_pressure = _load_observed_from_config().copy()

    inp_path = config.MODEL_INP
    metadata = config.build_default_metadata()
    raw_params = config.build_default_raw_params()

    runner = build_runner(inp_path=inp_path, metadata=metadata)
    _, results, _ = runner.build_and_run_once(raw_params)
    results = cast(RunResults, results)

    sim_pressure = results["pressure"].copy()

    sim_pressure = _coerce_index_to_seconds(sim_pressure)
    obs_pressure = _coerce_index_to_seconds(obs_pressure)

    sensors = [n for n in metadata.sensor_nodes if n in sim_pressure.columns and n in obs_pressure.columns]
    if not sensors:
        inter = [c for c in obs_pressure.columns if c in sim_pressure.columns]
        if not inter:
            raise ValueError("No overlapping sensor columns between observed CSV and simulated pressure")
        sensors = inter

    obs_s = obs_pressure[sensors]
    sim_s = sim_pressure[sensors]
    obs_aligned, sim_aligned = _align_sim_to_obs_index(obs_s, sim_s)

    metrics = _compute_sensor_metrics(obs_aligned, sim_aligned)

    metrics_csv = Path(config.REPORTS_DIR) / "sensor_pressure_fit_metrics.csv"
    obs_csv = Path(config.REPORTS_DIR) / "observed_sensor_pressure.csv"
    sim_csv = Path(config.REPORTS_DIR) / "simulated_sensor_pressure_aligned.csv"
    objective_json = Path(config.REPORTS_DIR) / "objective_breakdown.json"

    if config.SAVE_CSV:
        metrics.to_csv(metrics_csv, index=False)
        obs_aligned.to_csv(obs_csv, index=True)
        sim_aligned.to_csv(sim_csv, index=True)

    # Compute objective using the shared objective function.
    j_total, breakdown = runner.evaluate_objective(raw_params, observed_pressure=obs_pressure)
    breakdown = dict(breakdown)
    breakdown["J_total"] = float(j_total)

    objective_json.write_text(json.dumps(breakdown, indent=2, default=str))

    if config.VERBOSE:
        print("Sensors used:", len(sensors))
        print(metrics.head(10).to_string(index=False))
        print("Objective J_total:", float(j_total))
        print("Objective breakdown:")
        for k in sorted(breakdown.keys()):
            if k.startswith("J_") or k.startswith("wJ_"):
                print(f"  {k}: {breakdown[k]:.6g}")
        print(f"Wrote: {metrics_csv}")
        print(f"Wrote: {obs_csv}")
        print(f"Wrote: {sim_csv}")
        print(f"Wrote: {objective_json}")


if __name__ == "__main__":
    main()

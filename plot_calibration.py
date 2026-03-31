"""Plot observed vs simulated pressures at sensor nodes.

- Loads calibrated parameters from outputs/reports/best_params.json by default.
- Loads observed pressures from config.OBSERVED_PRESSURE_CSVS (multi-day) or OBSERVED_PRESSURE_CSV.
- Runs one simulation, aligns simulated pressures onto the observed time index, and saves PNG plots.

Outputs:
- outputs/reports/plots/sensors_overlay.png
- outputs/reports/plots/<SENSOR_ID>.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import pandas as pd

import config
from calibration.objective import load_observed_pressure_csv
from calibration.runner import RunResults, build_runner


def _ensure_dirs() -> Path:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    config.DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_dir = Path(config.REPORTS_DIR) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def _resolve_existing_path(path: str) -> str:
    p = Path(path)
    if p.exists():
        return str(p)
    alt = Path(__file__).resolve().parent / path
    if alt.exists():
        return str(alt)
    raise FileNotFoundError(
        f"Observed CSV not found: {path!r}. Tried: {str(p.resolve())!r} and {str(alt.resolve())!r}"
    )


def _coerce_index_to_seconds(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.dtype.kind in {"i", "u", "f"}:
        df.index = pd.Index(pd.to_numeric(df.index.to_numpy()))
        return df

    dt = pd.to_datetime(df.index)
    t0 = dt.min()
    seconds = (dt - t0).total_seconds()
    df.index = pd.Index(pd.to_numeric(np.asarray(seconds)))
    return df


def _load_observed_one(path: str) -> pd.DataFrame:
    path = _resolve_existing_path(path)
    if config.OBSERVED_TIME_COLUMN is None:
        df = load_observed_pressure_csv(path)
    else:
        raw = pd.read_csv(path)
        if config.OBSERVED_TIME_COLUMN not in raw.columns:
            raise ValueError(
                f"OBSERVED_TIME_COLUMN={config.OBSERVED_TIME_COLUMN!r} not found in {path}"
            )
        df = raw.set_index(config.OBSERVED_TIME_COLUMN)
        try:
            df.index = pd.Index(pd.to_numeric(df.index.to_numpy()))
        except Exception:
            df.index = pd.to_datetime(df.index)

    return _coerce_index_to_seconds(df)


def load_observed_multi_day() -> tuple[pd.DataFrame, int]:
    """Return (observed_df, n_days)."""

    if config.OBSERVED_PRESSURE_CSVS:
        dfs = []
        for i, p in enumerate(config.OBSERVED_PRESSURE_CSVS):
            d = _load_observed_one(p)
            d = d.copy()
            d.index = pd.Index(pd.to_numeric(d.index.to_numpy()) - float(d.index.min()) + float(i * 86400))
            dfs.append(d)
        out = pd.concat(dfs, axis=0).sort_index()
        return out, len(dfs)

    if not config.OBSERVED_PRESSURE_CSV:
        raise ValueError("Set OBSERVED_PRESSURE_CSV or OBSERVED_PRESSURE_CSVS in config.py")

    df = _load_observed_one(config.OBSERVED_PRESSURE_CSV)
    span = float(df.index.max() - df.index.min()) if len(df.index) else 0.0
    n_days = int(max(1, int(np.ceil((span + 1.0) / 86400.0))))
    return df, n_days


def _align_sim_to_obs_index(obs: pd.DataFrame, sim: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = obs.sort_index()
    sim = sim.sort_index()

    sim_r = sim.reindex(obs.index)
    try:
        sim_r = sim_r.interpolate(method="index", limit_direction="both")
    except Exception:
        sim_r = sim_r.ffill().bfill()

    return obs, sim_r


def _load_raw_params_from_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Params JSON not found: {str(p.resolve())}")

    data = json.loads(p.read_text())
    if isinstance(data, dict) and "best_raw_params" in data:
        rp = data["best_raw_params"]
        if not isinstance(rp, dict):
            raise ValueError("best_raw_params must be a dict")
        return cast(Dict[str, Any], rp)

    if isinstance(data, dict) and any(k in data for k in ("pda", "demand", "leakage", "time", "solver")):
        return cast(Dict[str, Any], data)

    raise ValueError(
        "Unrecognized params JSON format. Expected optimize.py best_params.json or a raw params dict."
    )


def _hours(t_s: np.ndarray) -> np.ndarray:
    return np.asarray(t_s, dtype=float) / 3600.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot observed vs simulated sensor pressures")
    parser.add_argument(
        "--params-json",
        default=str(config.OPT_BEST_PARAMS_JSON),
        help="Path to best_params.json (optimizer output) or a raw-params JSON dict",
    )
    parser.add_argument(
        "--max-sensors",
        type=int,
        default=12,
        help="Max number of sensors to include in the combined overlay plot",
    )
    args = parser.parse_args()

    plot_dir = _ensure_dirs()

    observed, n_days = load_observed_multi_day()
    metadata = config.build_default_metadata()

    try:
        raw_params = _load_raw_params_from_json(args.params_json)
    except FileNotFoundError:
        # Allow plotting even before optimization has been run.
        raw_params = config.build_default_raw_params()
    raw_params = dict(raw_params)
    raw_params.setdefault("time", {})
    raw_params["time"]["duration_days"] = int(max(1, n_days))

    runner = build_runner(inp_path=config.MODEL_INP, metadata=metadata)
    _, results, _ = runner.build_and_run_once(raw_params)
    results = cast(RunResults, results)

    sim_pressure = results["pressure"].copy()
    sim_pressure = _coerce_index_to_seconds(sim_pressure)

    sensors = [s for s in metadata.sensor_nodes if s in observed.columns and s in sim_pressure.columns]
    if not sensors:
        sensors = [c for c in observed.columns if c in sim_pressure.columns]

    if not sensors:
        raise ValueError("No overlapping sensor columns between observed data and simulated pressure.")

    obs_s = observed[sensors]
    sim_s = sim_pressure[sensors]
    obs_aligned, sim_aligned = _align_sim_to_obs_index(obs_s, sim_s)

    # Lazy import so the rest of the repo can run without matplotlib.
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from e

    # Per-sensor plots
    for s in sensors:
        fig = plt.figure(figsize=(10, 4))
        t = _hours(obs_aligned.index.to_numpy())
        plt.plot(t, obs_aligned[s].to_numpy(dtype=float), "o-", label="Observed", linewidth=1.5, markersize=3)
        plt.plot(t, sim_aligned[s].to_numpy(dtype=float), "-", label="Simulated", linewidth=2.0)
        plt.title(f"Sensor {s}: Observed vs Simulated")
        plt.xlabel("Time (hours since start)")
        plt.ylabel("Pressure")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out = plot_dir / f"{s}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)

    # Combined overlay plot (subset)
    max_sensors = int(max(1, args.max_sensors))
    sel = sensors[:max_sensors]
    fig = plt.figure(figsize=(12, 6))
    t = _hours(obs_aligned.index.to_numpy())
    for s in sel:
        plt.plot(t, obs_aligned[s].to_numpy(dtype=float), "--", linewidth=1.0, alpha=0.9)
        plt.plot(t, sim_aligned[s].to_numpy(dtype=float), "-", linewidth=1.5, alpha=0.9)
    plt.title(f"Observed (dashed) vs Simulated (solid) for {len(sel)} sensors")
    plt.xlabel("Time (hours since start)")
    plt.ylabel("Pressure")
    plt.grid(True, alpha=0.3)
    out = plot_dir / "sensors_overlay.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

    if config.VERBOSE:
        print(f"Sensors plotted: {len(sensors)}")
        print(f"Wrote plots to: {plot_dir}")
        print(f"Overlay: {out}")


if __name__ == "__main__":
    main()

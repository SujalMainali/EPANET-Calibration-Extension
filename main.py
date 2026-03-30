"""Production entrypoint: run one normal model workflow and write outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import config
from calibration.runner import RunResults, build_runner
from typing import cast


def _ensure_dirs() -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    config.DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _sensor_summary(pressure_df: pd.DataFrame, sensor_nodes: list[str]) -> pd.DataFrame:
    sensors = [n for n in sensor_nodes if n in pressure_df.columns]
    if not sensors:
        sensors = list(pressure_df.columns[: min(5, len(pressure_df.columns))])

    rows = []
    for n in sensors:
        rows.append(
            {
                "node": n,
                "min_pressure": float(pressure_df[n].min()),
                "max_pressure": float(pressure_df[n].max()),
                "mean_pressure": float(pressure_df[n].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    _ensure_dirs()

    inp_path = config.MODEL_INP
    metadata = config.build_default_metadata()
    raw_params = config.build_default_raw_params()

    runner = build_runner(inp_path=inp_path, metadata=metadata)
    _, results, _ = runner.build_and_run_once(raw_params)
    results = cast(RunResults, results)

    pressure_df = results["pressure"]
    demand_df = results["demand"]
    debug = results["debug"]

    pressure_csv = Path(config.RUNS_DIR) / "pressure_timeseries.csv"
    demand_csv = Path(config.RUNS_DIR) / "demand_timeseries.csv"
    summary_csv = Path(config.RUNS_DIR) / "run_summary.csv"
    debug_json = Path(config.RUNS_DIR) / "run_debug.json"

    if config.SAVE_CSV:
        pressure_df.to_csv(pressure_csv, index=True)
        demand_df.to_csv(demand_csv, index=True)
        _sensor_summary(pressure_df, metadata.sensor_nodes).to_csv(summary_csv, index=False)

    if config.SAVE_DEBUG_JSON:
        debug_json.write_text(json.dumps(debug, indent=2, default=str))

    if config.VERBOSE:
        print(f"Run complete. Pressure rows={len(pressure_df)}, cols={len(pressure_df.columns)}")
        print(f"Wrote: {pressure_csv}")
        print(f"Wrote: {demand_csv}")
        print(f"Wrote: {summary_csv}")
        print(f"Wrote: {debug_json}")


if __name__ == "__main__":
    main()

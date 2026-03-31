"""Run a simulation using calibrated parameters (e.g., optimizer output).

Default input is outputs/reports/best_params.json written by optimize.py.

Writes time series outputs under outputs/runs/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, cast

import config
from calibration.runner import RunResults, build_runner


def _ensure_dirs() -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    config.DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_raw_params_from_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Params JSON not found: {str(p.resolve())}")

    data = json.loads(p.read_text())

    # optimize.py format
    if isinstance(data, dict) and "best_raw_params" in data:
        rp = data["best_raw_params"]
        if not isinstance(rp, dict):
            raise ValueError("best_raw_params must be a dict")
        rp = cast(Dict[str, Any], rp)

        # If the optimizer ran against multi-day observations, its output JSON contains
        # n_days. Persist that into the raw params so calibrated runs cover the same
        # horizon by default.
        n_days = data.get("n_days")
        if n_days is not None:
            try:
                nd = int(n_days)
            except Exception:
                nd = None
            if nd and nd > 0:
                rp = dict(rp)
                rp.setdefault("time", {})
                rp["time"]["duration_days"] = int(max(1, nd))

        return rp

    # allow passing a plain raw-params dict
    if isinstance(data, dict) and any(k in data for k in ("pda", "demand", "leakage", "time", "solver")):
        return cast(Dict[str, Any], data)

    raise ValueError(
        "Unrecognized params JSON format. Expected optimize.py best_params.json or a raw params dict."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EPANET/WNTR simulation using calibrated parameters")
    parser.add_argument(
        "--params-json",
        default=str(config.OPT_BEST_PARAMS_JSON),
        help="Path to best_params.json (optimizer output) or a raw-params JSON dict",
    )
    parser.add_argument(
        "--run-name",
        default="calibrated",
        help="Prefix for output filenames under outputs/runs/",
    )
    parser.add_argument(
        "--duration-days",
        type=int,
        default=None,
        help="Override time.duration_days for the run (optional)",
    )
    args = parser.parse_args()

    _ensure_dirs()

    try:
        raw_params = _load_raw_params_from_json(args.params_json)
    except FileNotFoundError:
        raw_params = config.build_default_raw_params()
        if config.VERBOSE:
            print(f"[run_calibrated] Params file not found; using config defaults: {args.params_json}")

    # Duration defaults:
    # - If params JSON was optimize.py output, _load_raw_params_from_json already injected n_days.
    # - Otherwise, if OBSERVED_PRESSURE_CSVS is configured, default to that many days.
    # - User flag overrides everything.
    raw_params = dict(raw_params)
    raw_params.setdefault("time", {})
    if args.duration_days is not None:
        raw_params["time"]["duration_days"] = int(max(1, args.duration_days))
    elif config.OBSERVED_PRESSURE_CSVS:
        raw_params["time"]["duration_days"] = int(max(1, len(config.OBSERVED_PRESSURE_CSVS)))

    metadata = config.build_default_metadata()
    runner = build_runner(inp_path=config.MODEL_INP, metadata=metadata)

    _, results, _ = runner.build_and_run_once(raw_params)
    results = cast(RunResults, results)

    pressure_df = results["pressure"]
    demand_df = results["demand"]
    debug = results["debug"]

    prefix = str(args.run_name).strip() or "calibrated"
    pressure_csv = Path(config.RUNS_DIR) / f"{prefix}_pressure_timeseries.csv"
    demand_csv = Path(config.RUNS_DIR) / f"{prefix}_demand_timeseries.csv"
    debug_json = Path(config.RUNS_DIR) / f"{prefix}_run_debug.json"

    if config.SAVE_CSV:
        pressure_df.to_csv(pressure_csv, index=True)
        demand_df.to_csv(demand_csv, index=True)

    if config.SAVE_DEBUG_JSON:
        debug_json.write_text(json.dumps(debug, indent=2, default=str))

    if config.VERBOSE:
        print(f"Run complete. Pressure rows={len(pressure_df)}, cols={len(pressure_df.columns)}")
        print(f"Using params: {args.params_json}")
        print(f"Wrote: {pressure_csv}")
        print(f"Wrote: {demand_csv}")
        print(f"Wrote: {debug_json}")


if __name__ == "__main__":
    main()

"""Run a simulation using calibrated parameters safely.

Default input is outputs/reports/best_params.json written by optimize.py.

Enhancements:
- safer EPANET execution
- validation of calibrated parameters
- automatic cleanup of invalid leakage values
- protection against Error 200
- detailed debug reporting
- duration sanity checks
- emitter coefficient validation

Writes time series outputs under outputs/runs/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, cast

import numpy as np

import config
from calibration.runner import RunResults, build_runner


# =====================================================================
# DIRECTORY SETUP
# =====================================================================

def _ensure_dirs() -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    config.DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# PARAMETER LOADING
# =====================================================================

def _load_raw_params_from_json(path: str) -> Dict[str, Any]:

    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(
            f"Params JSON not found: {str(p.resolve())}"
        )

    data = json.loads(p.read_text())

    # -------------------------------------------------------------
    # optimize.py format
    # -------------------------------------------------------------
    if isinstance(data, dict) and "best_raw_params" in data:

        rp = data["best_raw_params"]

        if not isinstance(rp, dict):
            raise ValueError(
                "best_raw_params must be a dict"
            )

        rp = cast(Dict[str, Any], rp)

        # preserve n_days
        n_days = data.get("n_days")

        if n_days is not None:

            try:
                nd = int(n_days)

            except Exception:
                nd = None

            if nd and nd > 0:

                rp = dict(rp)

                rp.setdefault("time", {})

                rp["time"]["duration_days"] = int(
                    max(1, nd)
                )

        return rp

    # -------------------------------------------------------------
    # plain raw params
    # -------------------------------------------------------------
    if (
        isinstance(data, dict)
        and any(
            k in data
            for k in (
                "pda",
                "demand",
                "leakage",
                "time",
                "solver",
            )
        )
    ):

        return cast(Dict[str, Any], data)

    raise ValueError(
        "Unrecognized params JSON format."
    )


# =====================================================================
# SAFETY CLEANING
# =====================================================================

def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:

        v = float(value)

        if not np.isfinite(v):
            return float(default)

        return float(v)

    except Exception:
        return float(default)


def _sanitize_raw_params(
    raw_params: Dict[str, Any],
) -> Dict[str, Any]:

    rp = dict(raw_params)

    # -------------------------------------------------------------
    # TIME
    # -------------------------------------------------------------
    rp.setdefault("time", {})

    duration_days = _safe_float(
        rp["time"].get("duration_days", 1),
        1,
    )

    rp["time"]["duration_days"] = int(
        max(1, duration_days)
    )

    # -------------------------------------------------------------
    # LEAKAGE
    # -------------------------------------------------------------
    rp.setdefault("leakage", {})

    leakage = rp["leakage"]

    # global scale
    leakage["global_scale"] = max(
        0.0,
        _safe_float(
            leakage.get("global_scale", 1.0),
            1.0,
        ),
    )

    # emitter exponent
    leakage["emitter_exponent"] = min(
        1.5,
        max(
            0.1,
            _safe_float(
                leakage.get(
                    "emitter_exponent",
                    0.5,
                ),
                0.5,
            ),
        ),
    )

    # -------------------------------------------------------------
    # ZONE MULTIPLIERS
    # -------------------------------------------------------------
    zone_mults = leakage.get(
        "zone_multipliers",
        {},
    )

    if isinstance(zone_mults, dict):

        clean_mults = {}

        for k, v in zone_mults.items():

            fv = _safe_float(v, 1.0)

            # avoid insane multipliers
            fv = min(max(fv, 0.0), 10.0)

            clean_mults[str(k)] = float(fv)

        leakage["zone_multipliers"] = clean_mults

    else:
        leakage["zone_multipliers"] = {}

    # -------------------------------------------------------------
    # NODE ZONES
    # -------------------------------------------------------------
    node_zones = leakage.get(
        "node_zones",
        {},
    )

    if not isinstance(node_zones, dict):
        leakage["node_zones"] = {}

    rp["leakage"] = leakage

    return rp


# =====================================================================
# DEBUGGING
# =====================================================================

def _print_debug_summary(
    raw_params: Dict[str, Any],
) -> None:

    leakage = raw_params.get("leakage", {})

    print("\n" + "=" * 70)
    print("CALIBRATED PARAMETER SUMMARY")
    print("=" * 70)

    print(
        f"Duration days: "
        f"{raw_params.get('time', {}).get('duration_days')}"
    )

    print(
        f"Global leakage scale: "
        f"{leakage.get('global_scale')}"
    )

    print(
        f"Emitter exponent: "
        f"{leakage.get('emitter_exponent')}"
    )

    zone_mults = leakage.get(
        "zone_multipliers",
        {},
    )

    if zone_mults:

        print("\nZone multipliers:")

        for k, v in zone_mults.items():

            print(f"  {k}: {v}")

    else:

        print("\nNo zone multipliers found.")

    print("=" * 70 + "\n")


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run EPANET/WNTR simulation using calibrated parameters"
        )
    )

    parser.add_argument(
        "--params-json",
        default=str(config.OPT_BEST_PARAMS_JSON),
        help=(
            "Path to best_params.json "
            "or raw params JSON"
        ),
    )

    parser.add_argument(
        "--run-name",
        default="calibrated",
        help=(
            "Prefix for output filenames"
        ),
    )

    parser.add_argument(
        "--duration-days",
        type=int,
        default=None,
        help=(
            "Override time.duration_days"
        ),
    )

    args = parser.parse_args()

    _ensure_dirs()

    # -------------------------------------------------------------
    # LOAD PARAMS
    # -------------------------------------------------------------
    try:

        raw_params = _load_raw_params_from_json(
            args.params_json
        )

    except FileNotFoundError:

        raw_params = config.build_default_raw_params()

        print(
            "[run_calibrated] "
            "Params file not found. "
            "Using config defaults."
        )

    # -------------------------------------------------------------
    # APPLY DURATION LOGIC
    # -------------------------------------------------------------
    raw_params = dict(raw_params)

    raw_params.setdefault("time", {})

    if args.duration_days is not None:

        raw_params["time"]["duration_days"] = int(
            max(1, args.duration_days)
        )

    elif config.OBSERVED_PRESSURE_CSVS:

        raw_params["time"]["duration_days"] = int(
            max(
                1,
                len(config.OBSERVED_PRESSURE_CSVS),
            )
        )

    # -------------------------------------------------------------
    # SANITIZE PARAMETERS
    # -------------------------------------------------------------
    raw_params = _sanitize_raw_params(
        raw_params
    )

    # -------------------------------------------------------------
    # DEBUG SUMMARY
    # -------------------------------------------------------------
    _print_debug_summary(raw_params)

    # -------------------------------------------------------------
    # BUILD RUNNER
    # -------------------------------------------------------------
    metadata = config.build_default_metadata()

    runner = build_runner(
        inp_path=config.MODEL_INP,
        metadata=metadata,
    )

    # -------------------------------------------------------------
    # RUN SIMULATION
    # -------------------------------------------------------------
    try:

        _, results, _ = runner.build_and_run_once(
            raw_params
        )

    except Exception as e:

        print("\n" + "=" * 70)
        print("EPANET SIMULATION FAILED")
        print("=" * 70)

        print(f"\nError:\n{e}")

        print(
            "\nMost likely causes:"
        )

        print(
            "1. Invalid emitter coefficients"
        )

        print(
            "2. Invalid zone multipliers"
        )

        print(
            "3. Corrupted INP generation"
        )

        print(
            "4. Negative leakage values"
        )

        print(
            "5. Non-finite parameters"
        )

        print(
            "6. Extremely large leakage scaling"
        )

        print("=" * 70)

        raise

    # -------------------------------------------------------------
    # CAST RESULTS
    # -------------------------------------------------------------
    results = cast(
        RunResults,
        results,
    )

    pressure_df = results["pressure"]

    demand_df = results["demand"]

    debug = results["debug"]

    # -------------------------------------------------------------
    # OUTPUT FILES
    # -------------------------------------------------------------
    prefix = (
        str(args.run_name).strip()
        or "calibrated"
    )

    pressure_csv = (
        Path(config.RUNS_DIR)
        / f"{prefix}_pressure_timeseries.csv"
    )

    demand_csv = (
        Path(config.RUNS_DIR)
        / f"{prefix}_demand_timeseries.csv"
    )

    debug_json = (
        Path(config.RUNS_DIR)
        / f"{prefix}_run_debug.json"
    )

    # -------------------------------------------------------------
    # SAVE OUTPUTS
    # -------------------------------------------------------------
    if config.SAVE_CSV:

        pressure_df.to_csv(
            pressure_csv,
            index=True,
        )

        demand_df.to_csv(
            demand_csv,
            index=True,
        )

    if config.SAVE_DEBUG_JSON:

        debug_json.write_text(
            json.dumps(
                debug,
                indent=2,
                default=str,
            )
        )

    # -------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)

    print(
        f"Pressure rows: "
        f"{len(pressure_df)}"
    )

    print(
        f"Pressure cols: "
        f"{len(pressure_df.columns)}"
    )

    print(
        f"Using params: "
        f"{args.params_json}"
    )

    print(
        f"Wrote: {pressure_csv}"
    )

    print(
        f"Wrote: {demand_csv}"
    )

    print(
        f"Wrote: {debug_json}"
    )

    print("=" * 70 + "\n")


# =====================================================================
# ENTRY
# =====================================================================

if __name__ == "__main__":
    main()
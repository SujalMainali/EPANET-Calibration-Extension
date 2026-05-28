"""Verify that emitter coefficients actually affect pressures.

This script builds a tiny EPANET network (reservoir -> junction) and runs it
through the repo's toolkit-based hydraulic layer while sweeping:
- emitter exponent (n)
- emitter coefficient (C)

It prints mean junction pressure for each case and flags if changing C produces
no change in pressure (which would indicate emitters are not being applied).

Run (using your existing env):
  source ~/GlobalPython/bin/activate && python scripts/verify_emitter_addition.py
"""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import wntr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration.datamodels import ModelMetadata
from calibration.runner import build_runner


def _build_tiny_inp(path: Path) -> None:
    wn = wntr.network.WaterNetworkModel()
    wn.add_reservoir("R1", base_head=50.0)
    wn.add_junction("J1", base_demand=0.0, elevation=0.0)
    wn.add_pipe("P1", "R1", "J1", length=1000.0, diameter=0.2, roughness=110, minor_loss=0.0)

    # Use metric LPM to match the dataset assumptions.
    wn.options.hydraulic.inpfile_units = "LPM"
    wn.options.hydraulic.headloss = "H-W"

    # 1-hour report/hyd step for a quick run.
    wn.options.time.duration = 6 * 3600
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.report_timestep = 3600
    wn.options.time.report_start = 0

    wntr.network.io.write_inpfile(wn, str(path))


def _run_case(inp_path: str, *, exponent: float, coeff: float) -> float:
    metadata = ModelMetadata(service_nodes={}, leak_nodes={}, sensor_nodes=["J1"], leak_check_node=None, pda_check_node=None)
    runner = build_runner(inp_path=inp_path, metadata=metadata)

    raw = {
        "pda": {
            "demand_model": "PDA",
            "minimum_pressure": 0.0,
            "required_pressure": 10.0,
            "pressure_exponent": 0.5,
        },
        "leakage": {
            "global_scale": 0.0,
            "zone_multipliers": {},
            "emitter_exponent": float(exponent),
        },
        "demand": {"demand_multiplier": 1.0},
        "time": {
            "duration_days": 1,
            "hydraulic_timestep_s": 3600,
            "report_timestep_s": 3600,
            "report_start_s": 0,
        },
        "solver": {
            "trials": 200,
            "accuracy": 0.001,
            "unbalanced": "STOP",
            "damplimit": 0.0,
            "checkfreq": 2,
            "maxcheck": 10,
        },
    }

    # Apply an always-on emitter for the whole simulation.
    _, results, _ = runner.build_and_run_once(
        raw,
        nodes_to_read={"J1"},
        emitter_window_overrides={"J1": (0, 24 * 3600, float(coeff))},
        emitter_window_override_mode="set",
    )

    p = results["pressure"]["J1"]
    return float(p.mean())


def main() -> None:
    tmp = Path(tempfile.gettempdir()) / "_verify_emitter_addition.inp"
    _build_tiny_inp(tmp)

    # Focus on the reported problematic regime: exponent < 1.
    # Use larger coefficients for smaller exponents to make the effect visible.
    exponents = [0.3, 0.5, 0.8, 0.95]

    print("INP:", tmp)
    print("Sweeping emitter exponent and coefficient...")
    print()

    failures = 0

    for n in exponents:
        high_c = 200.0 if n <= 0.5 else 50.0
        try:
            base = _run_case(str(tmp), exponent=n, coeff=0.0)
            high = _run_case(str(tmp), exponent=n, coeff=high_c)
        except Exception as exc:
            failures += 1
            print(f"n={n:<4}  [FAIL] simulation error: {exc}")
            continue

        delta = high - base
        print(f"n={n:<4}  meanP(C={0.0:>6.1f})={base:9.4f}   meanP(C={high_c:>6.1f})={high:9.4f}   Δ={delta:9.4f}")

        # Increasing leak should reduce pressure (delta should be negative and non-trivial)
        if abs(delta) < 1e-2:
            failures += 1
            print("  [FAIL] Pressure did not meaningfully change when C changed.")
        elif delta > 0:
            failures += 1
            print("  [WARN] Pressure increased with larger emitter.")

    print()
    if failures:
        raise SystemExit(f"Emitter verification failed for {failures} exponent(s).")
    print("PASS: Emitters affect pressures across all tested exponents.")


if __name__ == "__main__":
    main()

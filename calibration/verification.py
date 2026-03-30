"""Debug/test-oriented verification routines.

These helpers compare pairs of runs and attach debug metadata.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from calibration.runner import LayeredModelRunner


def verify_emitter_effect(
    runner: LayeredModelRunner,
    raw_params_off: dict,
    raw_params_on: dict,
    check_node: Optional[str] = None,
) -> pd.DataFrame:
    check_node = check_node or runner.metadata.leak_check_node
    if not check_node:
        raise ValueError("Set metadata.leak_check_node or pass check_node explicitly.")

    _, res_off, _ = runner.build_and_run_once(raw_params_off)
    _, res_on, _ = runner.build_and_run_once(raw_params_on)

    p_off = res_off["pressure"][check_node].rename("pressure_off")
    p_on = res_on["pressure"][check_node].rename("pressure_on")

    out = pd.concat([p_off, p_on], axis=1)
    out.index.name = "time_s"
    out = out.reset_index()
    out["delta_on_minus_off"] = out["pressure_on"] - out["pressure_off"]

    out.attrs["debug_off"] = res_off["debug"]
    out.attrs["debug_on"] = res_on["debug"]
    out.attrs["length_off"] = len(p_off)
    out.attrs["length_on"] = len(p_on)
    return out


def verify_pda_effect(
    runner: LayeredModelRunner,
    raw_params_low_req: dict,
    raw_params_high_req: dict,
    check_node: Optional[str] = None,
) -> pd.DataFrame:
    check_node = check_node or runner.metadata.pda_check_node
    if not check_node:
        raise ValueError("Set metadata.pda_check_node or pass check_node explicitly.")

    _, res_low, _ = runner.build_and_run_once(raw_params_low_req)
    _, res_high, _ = runner.build_and_run_once(raw_params_high_req)

    p_low = res_low["pressure"][check_node].rename("pressure_lowReq")
    p_high = res_high["pressure"][check_node].rename("pressure_highReq")

    d_low = (
        res_low["demand"][check_node].rename("demand_lowReq")
        if check_node in res_low["demand"].columns
        else pd.Series(dtype=float, name="demand_lowReq")
    )
    d_high = (
        res_high["demand"][check_node].rename("demand_highReq")
        if check_node in res_high["demand"].columns
        else pd.Series(dtype=float, name="demand_highReq")
    )

    out = pd.concat([p_low, p_high, d_low, d_high], axis=1)
    out.index.name = "time_s"
    out = out.reset_index()
    out["delta_demand_high_minus_low"] = out["demand_highReq"] - out["demand_lowReq"]

    out.attrs["debug_low"] = res_low["debug"]
    out.attrs["debug_high"] = res_high["debug"]
    out.attrs["length_low"] = len(p_low)
    out.attrs["length_high"] = len(p_high)
    return out

"""Hydraulic model layer implemented via direct ENepanet() toolkit calls."""

from __future__ import annotations

import copy
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import wntr
from wntr.epanet.toolkit import ENepanet

from calibration.behavior_layer import BehaviorLayer
from calibration.datamodels import ModelMetadata, ModelParameters
from utils.dataframe_helpers import expand_to_report_times
from utils.epanet_helpers import EN_const, call_en_get, call_en_get_int, safe_float
from utils.time_helpers import expected_report_times, summarize_time_index


class HydraulicModelLayerENepanet:
    def __init__(self, inp_path: str):
        self.inp_path = Path(inp_path).resolve()
        if not self.inp_path.exists():
            raise FileNotFoundError(f"INP file not found: {self.inp_path}")
        self.base_wn = wntr.network.WaterNetworkModel(str(self.inp_path))

    def clone_network(self) -> "wntr.network.WaterNetworkModel":
        return copy.deepcopy(self.base_wn)

    def apply_pda_settings_to_inp_model(self, wn_model: "wntr.network.WaterNetworkModel", params: ModelParameters) -> None:
        demand_model = params.pda.demand_model.upper()
        if demand_model == "PDD":
            demand_model = "PDA"
        wn_model.options.hydraulic.demand_model = demand_model
        wn_model.options.hydraulic.minimum_pressure = params.pda.minimum_pressure
        wn_model.options.hydraulic.required_pressure = params.pda.required_pressure
        wn_model.options.hydraulic.pressure_exponent = params.pda.pressure_exponent

        if params.leakage.emitter_exponent is not None:
            wn_model.options.hydraulic.emitter_exponent = params.leakage.emitter_exponent

        wn_model.options.time.duration = params.time.duration_days * 24 * 3600
        wn_model.options.time.hydraulic_timestep = params.time.hydraulic_timestep_s
        wn_model.options.time.report_timestep = params.time.report_timestep_s
        wn_model.options.time.report_start = params.time.report_start_s
        wn_model.options.time.pattern_timestep = params.time.report_timestep_s
        wn_model.options.time.pattern_start = 0

        # Solver/convergence controls
        wn_model.options.hydraulic.trials = int(params.solver.trials)
        wn_model.options.hydraulic.accuracy = float(params.solver.accuracy)
        wn_model.options.hydraulic.unbalanced = str(params.solver.unbalanced).upper()
        wn_model.options.hydraulic.damplimit = float(params.solver.damplimit)
        wn_model.options.hydraulic.checkfreq = int(params.solver.checkfreq)
        wn_model.options.hydraulic.maxcheck = int(params.solver.maxcheck)

    def apply_service_node_demands(
        self,
        wn_model: "wntr.network.WaterNetworkModel",
        metadata: ModelMetadata,
        behavior: BehaviorLayer,
        params: ModelParameters,
    ) -> None:
        for node_name, meta in metadata.service_nodes.items():
            j = wn_model.get_node(node_name)
            hourly_m3ph = behavior.build_node_hourly_demands_m3ph(params, meta)
            avg_hourly_m3ph = float(np.mean(hourly_m3ph))
            if avg_hourly_m3ph <= 0:
                avg_hourly_m3ph = 1e-9
            pattern = (hourly_m3ph / avg_hourly_m3ph).tolist()
            pattern_name = f"dynpat_{node_name}"

            if pattern_name in wn_model.pattern_name_list:
                wn_model.remove_pattern(pattern_name)
            wn_model.add_pattern(pattern_name, pattern)

            avg_m3ps = avg_hourly_m3ph / 3600.0
            j.demand_timeseries_list.clear()
            j.add_demand(avg_m3ps, pattern_name, category="customer_dynamic")

    def write_temp_inp(self, wn_model: "wntr.network.WaterNetworkModel") -> Tuple[Path, Path, Path]:
        run_dir = Path("outputs/debug/temp_runs")
        run_dir.mkdir(parents=True, exist_ok=True)

        inp_tmp = run_dir / "model_tmp.inp"
        rpt_tmp = run_dir / "model_tmp.rpt"
        out_tmp = run_dir / "model_tmp.bin"
        wntr.network.io.write_inpfile(wn_model, inp_tmp)
        return inp_tmp, rpt_tmp, out_tmp

    def run(
        self,
        wn_model: "wntr.network.WaterNetworkModel",
        metadata: ModelMetadata,
        behavior: BehaviorLayer,
        params: ModelParameters,
    ) -> Dict[str, object]:
        inp_tmp, rpt_tmp, out_tmp = self.write_temp_inp(wn_model)

        en = ENepanet()
        en.ENopen(str(inp_tmp), str(rpt_tmp), str(out_tmp))
        try:
            C_DURATION = EN_const("DURATION")
            C_HYDSTEP = EN_const("HYDSTEP")
            C_REPORTSTEP = EN_const("REPORTSTEP")
            C_REPORTSTART = EN_const("REPORTSTART")
            C_PRESSURE = EN_const("PRESSURE")
            C_EMITTER = EN_const("EMITTER")

            C_DEMAND = None
            try:
                C_DEMAND = EN_const("DEMAND")
            except Exception:
                pass

            en.ENsettimeparam(C_DURATION, params.time.duration_days * 24 * 3600)
            en.ENsettimeparam(C_HYDSTEP, params.time.hydraulic_timestep_s)
            en.ENsettimeparam(C_REPORTSTEP, params.time.report_timestep_s)
            en.ENsettimeparam(C_REPORTSTART, params.time.report_start_s)

            # Some bindings behave more predictably when pattern timing is set explicitly.
            # These params exist in EPANET 2.x but constant names can vary.
            for cname, value in (
                ("PATTERNSTEP", params.time.report_timestep_s),
                ("PATTERNSTART", 0),
            ):
                try:
                    en.ENsettimeparam(EN_const(cname), int(value))
                except Exception:
                    pass

            nodes_to_read = sorted(
                set(metadata.sensor_nodes)
                | set(metadata.service_nodes.keys())
                | set(metadata.leak_nodes.keys())
            )
            node_index = {n: call_en_get(en, "ENgetnodeindex", n) for n in nodes_to_read}

            emitter_coeffs = behavior.grouped_emitter_coefficients(params)
            for node_name, coeff in emitter_coeffs.items():
                idx = node_index[node_name]
                en.ENsetnodevalue(idx, C_EMITTER, float(coeff))

            en.ENopenH()
            en.ENinitH(0)

            pressure_rows = []
            demand_rows = []
            times_s = []
            next_steps = []

            while True:
                t = call_en_get_int(en, "ENrunH")
                times_s.append(t)

                prow = {}
                drow = {}

                for n in nodes_to_read:
                    idx = node_index[n]
                    prow[n] = safe_float(call_en_get(en, "ENgetnodevalue", idx, C_PRESSURE))
                    if C_DEMAND is not None:
                        try:
                            drow[n] = safe_float(call_en_get(en, "ENgetnodevalue", idx, C_DEMAND))
                        except Exception:
                            drow[n] = np.nan

                pressure_rows.append(prow)
                if C_DEMAND is not None:
                    demand_rows.append(drow)

                tstep = call_en_get_int(en, "ENnextH")
                next_steps.append(tstep)
                if tstep <= 0:
                    break

            en.ENcloseH()

            pressure_df = pd.DataFrame(pressure_rows, index=times_s)
            pressure_df.index.name = "time_s"

            if demand_rows:
                demand_df = pd.DataFrame(demand_rows, index=times_s)
                demand_df.index.name = "time_s"
            else:
                demand_df = pd.DataFrame(index=times_s, columns=sorted(metadata.service_nodes.keys()), dtype=float)

            duration_s = params.time.duration_days * 24 * 3600
            expected_times = expected_report_times(
                duration_s=duration_s,
                report_step_s=params.time.report_timestep_s,
                report_start_s=params.time.report_start_s,
            )

            pressure_df_raw = pressure_df.copy()
            demand_df_raw = demand_df.copy()

            pressure_df = expand_to_report_times(pressure_df, expected_times)
            demand_df = expand_to_report_times(demand_df, expected_times)

            actual_summary = summarize_time_index(times_s)
            expected_summary = summarize_time_index(expected_times)

            warnings_list = []
            if len(times_s) <= 1:
                msg = (
                    f"Only {len(times_s)} timestep(s) returned. "
                    f"Expected around {len(expected_times)} for duration={duration_s}s "
                    f"and report_step={params.time.report_timestep_s}s."
                )
                warnings_list.append(msg)
                warnings.warn(msg)

            if len(times_s) != len(expected_times):
                msg = (
                    f"Timestep count mismatch: got {len(times_s)}, expected {len(expected_times)}. "
                    f"Actual first/last={actual_summary['first_times']} / {actual_summary['last_times']}, "
                    f"Expected first/last={expected_summary['first_times']} / {expected_summary['last_times']}."
                )
                warnings_list.append(msg)

            if len(times_s) > 1:
                unique_diffs = sorted(set(np.diff(times_s)))
                if params.time.report_timestep_s not in unique_diffs:
                    msg = (
                        f"Observed timestep diffs {unique_diffs} do not contain the configured "
                        f"report step {params.time.report_timestep_s}."
                    )
                    warnings_list.append(msg)

            run_debug = {
                "temp_inp": str(inp_tmp),
                "temp_rpt": str(rpt_tmp),
                "temp_out": str(out_tmp),
                "emitter_coeffs": emitter_coeffs,
                "pda_settings_written_to_inp": {
                    "demand_model": wn_model.options.hydraulic.demand_model,
                    "minimum_pressure": wn_model.options.hydraulic.minimum_pressure,
                    "required_pressure": wn_model.options.hydraulic.required_pressure,
                    "pressure_exponent": wn_model.options.hydraulic.pressure_exponent,
                    "emitter_exponent": getattr(wn_model.options.hydraulic, "emitter_exponent", None),
                },
                "solver_settings_written_to_inp": {
                    "trials": wn_model.options.hydraulic.trials,
                    "accuracy": wn_model.options.hydraulic.accuracy,
                    "unbalanced": wn_model.options.hydraulic.unbalanced,
                    "damplimit": wn_model.options.hydraulic.damplimit,
                    "checkfreq": wn_model.options.hydraulic.checkfreq,
                    "maxcheck": wn_model.options.hydraulic.maxcheck,
                },
                "expected_time_index": expected_times,
                "actual_time_index": times_s,
                "raw_pressure_shape": pressure_df_raw.shape,
                "raw_demand_shape": demand_df_raw.shape,
                "expanded_pressure_shape": pressure_df.shape,
                "expanded_demand_shape": demand_df.shape,
                "actual_time_summary": actual_summary,
                "expected_time_summary": expected_summary,
                "next_steps": next_steps[:20],
                "warnings": warnings_list,
            }

            return {
                "pressure": pressure_df,
                "demand": demand_df,
                "debug": run_debug,
            }
        finally:
            try:
                en.ENclose()
            except Exception:
                pass

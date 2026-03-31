"""Parameterization layer: raw dicts -> typed ModelParameters with validation."""

from __future__ import annotations

from calibration.datamodels import (
    CarryoverSettings,
    GlobalDemandSettings,
    HydraulicSolverSettings,
    ModelMetadata,
    ModelParameters,
    PatternFamilyParams,
    PDASettings,
    TimeSettings,
    ZoneLeakageSettings,
)


class ParameterizationLayer:
    def __init__(self, metadata: ModelMetadata):
        self.metadata = metadata

    def from_dict(self, raw: dict) -> ModelParameters:
        params = ModelParameters()

        pda = raw.get("pda", {})
        params.pda = PDASettings(
            demand_model=str(pda.get("demand_model", "PDA")).upper(),
            minimum_pressure=float(pda.get("minimum_pressure", 3.0)),
            required_pressure=float(pda.get("required_pressure", 12.0)),
            pressure_exponent=float(pda.get("pressure_exponent", 0.5)),
        )

        pf = raw.get("pattern_family", {})
        params.pattern_family = PatternFamilyParams(
            morning_center=float(pf.get("morning_center", 7.5)),
            morning_width=float(pf.get("morning_width", 1.5)),
            morning_weight=float(pf.get("morning_weight", 0.72)),
            evening_center=float(pf.get("evening_center", 19.0)),
            evening_width=float(pf.get("evening_width", 1.2)),
            evening_weight=float(pf.get("evening_weight", 0.18)),
            background_weight=float(pf.get("background_weight", 0.10)),
            floor=float(pf.get("floor", 0.02)),
        )

        co = raw.get("carryover", {})
        params.carryover = CarryoverSettings(
            enabled=bool(co.get("enabled", False)),
            catchup_factor=float(co.get("catchup_factor", 0.50)),
            max_carryover_multiplier=float(co.get("max_carryover_multiplier", 2.0)),
        )

        lk = raw.get("leakage", {})
        params.leakage = ZoneLeakageSettings(
            global_scale=float(lk.get("global_scale", 0.0)),
            zone_multipliers={str(k): float(v) for k, v in lk.get("zone_multipliers", {}).items()},
            emitter_exponent=None
            if lk.get("emitter_exponent") is None
            else float(lk.get("emitter_exponent", 0.5)),
        )

        dm = raw.get("demand", {})
        params.demand = GlobalDemandSettings(
            demand_multiplier=float(dm.get("demand_multiplier", 1.0)),
        )

        tm = raw.get("time", {})
        params.time = TimeSettings(
            duration_days=int(tm.get("duration_days", 1)),
            hydraulic_timestep_s=int(tm.get("hydraulic_timestep_s", 3600)),
            report_timestep_s=int(tm.get("report_timestep_s", 3600)),
            report_start_s=int(tm.get("report_start_s", 0)),
        )

        sv = raw.get("solver", {})
        params.solver = HydraulicSolverSettings(
            trials=int(sv.get("trials", params.solver.trials)),
            accuracy=float(sv.get("accuracy", params.solver.accuracy)),
            unbalanced=str(sv.get("unbalanced", params.solver.unbalanced)).upper(),
            damplimit=float(sv.get("damplimit", params.solver.damplimit)),
            checkfreq=int(sv.get("checkfreq", params.solver.checkfreq)),
            maxcheck=int(sv.get("maxcheck", params.solver.maxcheck)),
        )

        self._validate(params)
        return params

    def _validate(self, params: ModelParameters) -> None:
        if params.pda.required_pressure <= params.pda.minimum_pressure:
            raise ValueError("required_pressure must be greater than minimum_pressure")
        if params.pda.pressure_exponent <= 0:
            raise ValueError("pressure_exponent must be > 0")
        if params.leakage.global_scale < 0:
            raise ValueError("leakage.global_scale must be >= 0")
        if params.leakage.emitter_exponent is not None and params.leakage.emitter_exponent <= 0:
            raise ValueError("leakage.emitter_exponent must be > 0")
        if params.pattern_family.morning_width <= 0 or params.pattern_family.evening_width <= 0:
            raise ValueError("pattern widths must be > 0")
        if params.demand.demand_multiplier <= 0:
            raise ValueError("demand_multiplier must be > 0")
        if params.time.hydraulic_timestep_s <= 0 or params.time.report_timestep_s <= 0:
            raise ValueError("timesteps must be > 0")
        if params.solver.trials <= 0:
            raise ValueError("solver.trials must be > 0")
        if params.solver.accuracy <= 0:
            raise ValueError("solver.accuracy must be > 0")
        if params.solver.unbalanced not in {"STOP", "CONTINUE"}:
            raise ValueError("solver.unbalanced must be STOP or CONTINUE")
        if params.solver.damplimit < 0:
            raise ValueError("solver.damplimit must be >= 0")
        if params.solver.checkfreq <= 0 or params.solver.maxcheck <= 0:
            raise ValueError("solver.checkfreq and solver.maxcheck must be > 0")

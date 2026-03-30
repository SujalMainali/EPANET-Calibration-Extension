"""Orchestration for the layered calibration scaffold."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Tuple, TypedDict, cast

import pandas as pd

from calibration.behavior_layer import BehaviorLayer
from calibration.datamodels import ModelMetadata, ModelParameters
from calibration.hydraulic_layer_enepanet import HydraulicModelLayerENepanet
from calibration.parameterization_layer import ParameterizationLayer


if TYPE_CHECKING:
    from calibration.objective import ObjectiveConfig


class RunResults(TypedDict):
    pressure: pd.DataFrame
    demand: pd.DataFrame
    debug: Dict[str, Any]


class LayeredModelRunner:
    def __init__(
        self,
        hydraulic: HydraulicModelLayerENepanet,
        parameterization: ParameterizationLayer,
        behavior: BehaviorLayer,
        metadata: ModelMetadata,
    ):
        self.hydraulic = hydraulic
        self.parameterization = parameterization
        self.behavior = behavior
        self.metadata = metadata

    def build_and_run_once(self, raw_params: dict) -> Tuple["object", RunResults, ModelParameters]:
        params = self.parameterization.from_dict(raw_params)
        wn_model = self.hydraulic.clone_network()
        self.hydraulic.apply_pda_settings_to_inp_model(wn_model, params)
        self.hydraulic.apply_service_node_demands(wn_model, self.metadata, self.behavior, params)
        results = cast(RunResults, self.hydraulic.run(wn_model, self.metadata, self.behavior, params))
        return wn_model, results, params

    def smoke_test(self, raw_params: dict) -> pd.DataFrame:
        _, results, params = self.build_and_run_once(raw_params)

        pressure_df = results["pressure"]
        demand_df = results["demand"]

        sensor_list = [n for n in self.metadata.sensor_nodes if n in pressure_df.columns]
        if not sensor_list:
            sensor_list = list(pressure_df.columns[: min(5, len(pressure_df.columns))])

        summary_rows = []
        for n in sensor_list:
            summary_rows.append(
                {
                    "node": n,
                    "min_pressure": float(pressure_df[n].min()),
                    "max_pressure": float(pressure_df[n].max()),
                    "mean_pressure": float(pressure_df[n].mean()),
                }
            )

        unmet = {}
        if not demand_df.empty:
            service_cols = [n for n in self.metadata.service_nodes if n in demand_df.columns]
            unmet = self.behavior.update_unmet_from_demand_df(
                demand_df=demand_df.loc[:, service_cols],
                params=params,
                report_timestep_s=params.time.report_timestep_s,
            )

        summary_df = pd.DataFrame(summary_rows)
        summary_df.attrs["unmet_daily_m3"] = unmet
        summary_df.attrs["run_debug"] = results["debug"]
        return summary_df

    def evaluate_objective(
        self,
        raw_params: dict,
        observed_pressure: pd.DataFrame,
        *,
        config: "ObjectiveConfig | None" = None,
        sensor_weights: "dict[str, float] | None" = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Run once and evaluate the composite objective J(θ).

        This is a convenience wrapper intended for optimizers.
        """

        from calibration.objective import compute_objective

        _, results, params = self.build_and_run_once(raw_params)
        return compute_objective(
            params=params,
            metadata=self.metadata,
            results=results,
            observed_pressure=observed_pressure,
            config=config,
            sensor_weights=sensor_weights,
        )


def build_runner(inp_path: str, metadata: ModelMetadata) -> LayeredModelRunner:
    hydraulic = HydraulicModelLayerENepanet(inp_path)
    parameterization = ParameterizationLayer(metadata)
    behavior = BehaviorLayer(metadata)
    return LayeredModelRunner(
        hydraulic=hydraulic,
        parameterization=parameterization,
        behavior=behavior,
        metadata=metadata,
    )

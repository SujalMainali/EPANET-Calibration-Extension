"""Behavior layer: dynamic patterns, carryover demand, grouped leak emitters."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from calibration.datamodels import ModelMetadata, ModelParameters, PatternFamilyParams, ServiceNodeMeta
from utils.pattern_helpers import gaussian, normalize24


class BehaviorLayer:
    def __init__(self, metadata: ModelMetadata):
        self.metadata = metadata
        self.unmet_daily_m3: Dict[str, float] = {n: 0.0 for n in metadata.service_nodes}

    def build_preferred_pattern(self, pf: PatternFamilyParams) -> np.ndarray:
        hrs = np.arange(24, dtype=float)
        raw = (
            pf.morning_weight * gaussian(hrs, pf.morning_center, pf.morning_width)
            + pf.evening_weight * gaussian(hrs, pf.evening_center, pf.evening_width)
            + np.full(24, pf.background_weight, dtype=float)
            + pf.floor
        )
        return normalize24(raw)

    def build_node_hourly_demands_m3ph(self, params: ModelParameters, node_meta: ServiceNodeMeta) -> np.ndarray:
        daily_target = node_meta.base_daily_volume_m3 * params.demand.demand_multiplier
        preferred_pattern = self.build_preferred_pattern(params.pattern_family)
        preferred_hourly = daily_target * preferred_pattern / 24.0

        if not params.carryover.enabled:
            return preferred_hourly

        unmet = self.unmet_daily_m3.get(node_meta.node_name, 0.0)
        extra = params.carryover.catchup_factor * unmet
        max_extra = daily_target * (params.carryover.max_carryover_multiplier - 1.0)
        extra = min(extra, max_extra)

        late_weights = np.array(
            [
                0.01,
                0.01,
                0.01,
                0.01,
                0.03,
                0.05,
                0.10,
                0.12,
                0.10,
                0.08,
                0.05,
                0.03,
                0.02,
                0.02,
                0.02,
                0.03,
                0.05,
                0.08,
                0.10,
                0.10,
                0.08,
                0.05,
                0.03,
                0.02,
            ],
            dtype=float,
        )
        late_weights /= late_weights.sum()
        return preferred_hourly + extra * late_weights

    def grouped_emitter_coefficients(self, params: ModelParameters) -> Dict[str, float]:
        out = {}
        for node_name, meta in self.metadata.leak_nodes.items():
            zone_mult = params.leakage.zone_multipliers.get(meta.zone, 0.0)
            out[node_name] = zone_mult * meta.weight
        return out

    def update_unmet_from_demand_df(
        self,
        demand_df: pd.DataFrame,
        params: ModelParameters,
        report_timestep_s: int,
    ) -> Dict[str, float]:
        updated = {}
        for node_name, meta in self.metadata.service_nodes.items():
            target_m3 = meta.base_daily_volume_m3 * params.demand.demand_multiplier
            if node_name in demand_df.columns:
                delivered_m3 = float(demand_df[node_name].fillna(0.0).sum()) * report_timestep_s
            else:
                delivered_m3 = 0.0
            updated[node_name] = max(0.0, target_m3 - delivered_m3)
        self.unmet_daily_m3.update(updated)
        return updated

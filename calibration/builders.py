"""Local builder helpers for metadata and default raw parameters.

These are examples/placeholders you should edit to match your network.
"""

from __future__ import annotations

from calibration.datamodels import LeakNodeMeta, ModelMetadata, ServiceNodeMeta


def build_example_metadata() -> ModelMetadata:
    metadata = ModelMetadata()

    metadata.service_nodes = {
        "HOUSEEND_16438": ServiceNodeMeta("HOUSEEND_16438", base_daily_volume_m3=0.75, zone="Z1"),
        "HOUSEEND_16604": ServiceNodeMeta("HOUSEEND_16604", base_daily_volume_m3=0.65, zone="Z1"),
        "HOUSEEND_17872": ServiceNodeMeta("HOUSEEND_17872", base_daily_volume_m3=0.90, zone="Z2"),
    }

    metadata.leak_nodes = {
        "NODE_3004": LeakNodeMeta("NODE_3004", zone="Z1", weight=0.5),
        "NODE_3005": LeakNodeMeta("NODE_3005", zone="Z1", weight=0.5),
        "NODE_3006": LeakNodeMeta("NODE_3006", zone="Z2", weight=0.5),
    }

    metadata.sensor_nodes = ["HOUSEEND_16032", "HOUSEEND_16426", "HOUSEEND_16702"]
    metadata.leak_check_node = "NODE_3004"
    metadata.pda_check_node = "HOUSEEND_16438"
    return metadata


def build_example_raw_params() -> dict:
    return {
        "pda": {
            "demand_model": "PDA",
            "minimum_pressure": 3.0,
            "required_pressure": 12.0,
            "pressure_exponent": 0.5,
        },
        "pattern_family": {
            "morning_center": 7.5,
            "morning_width": 1.5,
            "morning_weight": 0.72,
            "evening_center": 19.0,
            "evening_width": 1.2,
            "evening_weight": 0.18,
            "background_weight": 0.10,
            "floor": 0.02,
        },
        "carryover": {
            "enabled": False,
            "catchup_factor": 0.50,
            "max_carryover_multiplier": 2.0,
        },
        "leakage": {
            "zone_multipliers": {"Z1": 1.0, "Z2": 1.3},
            "emitter_exponent": 0.5,
        },
        "demand": {
            "demand_multiplier": 1.0,
        },
        "time": {
            "duration_days": 1,
            "hydraulic_timestep_s": 3600,
            "report_timestep_s": 3600,
            "report_start_s": 0,
        },
    }

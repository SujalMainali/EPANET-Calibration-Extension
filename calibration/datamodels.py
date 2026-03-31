"""Shared dataclasses used across the layered calibration scaffold.

Keep this module free of simulation logic to avoid circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PDASettings:
    demand_model: str = "PDA"
    minimum_pressure: float = 3.0
    required_pressure: float = 12.0
    pressure_exponent: float = 0.5


@dataclass
class PatternFamilyParams:
    morning_center: float = 7.5
    morning_width: float = 1.5
    morning_weight: float = 0.72

    # Optional third spike to better match mid-day pressure/demand behavior.
    noon_center: float = 12.5
    noon_width: float = 1.5
    noon_weight: float = 0.0

    evening_center: float = 19.0
    evening_width: float = 1.2
    evening_weight: float = 0.18
    background_weight: float = 0.10
    floor: float = 0.02


@dataclass
class CarryoverSettings:
    enabled: bool = False
    catchup_factor: float = 0.50
    max_carryover_multiplier: float = 2.0


@dataclass
class ZoneLeakageSettings:
    # Global scaling applied to all emitter coefficients. Keep at 0.0 to effectively disable
    # emitters even if leak nodes are present.
    global_scale: float = 0.0
    zone_multipliers: Dict[str, float] = field(default_factory=dict)
    emitter_exponent: Optional[float] = 0.5


@dataclass
class GlobalDemandSettings:
    demand_multiplier: float = 1.0


@dataclass
class TimeSettings:
    duration_days: int = 1
    hydraulic_timestep_s: int = 3600
    report_timestep_s: int = 3600
    report_start_s: int = 0


@dataclass
class HydraulicSolverSettings:
    """Hydraulic solver controls that can affect convergence and halting.

    These map to EPANET [OPTIONS] fields like TRIALS, ACCURACY, UNBALANCED, etc.
    """

    trials: int = 40
    accuracy: float = 0.001
    unbalanced: str = "STOP"  # STOP or CONTINUE
    damplimit: float = 0.0
    checkfreq: int = 2
    maxcheck: int = 10


@dataclass
class ModelParameters:
    pda: PDASettings = field(default_factory=PDASettings)
    pattern_family: PatternFamilyParams = field(default_factory=PatternFamilyParams)
    carryover: CarryoverSettings = field(default_factory=CarryoverSettings)
    leakage: ZoneLeakageSettings = field(default_factory=ZoneLeakageSettings)
    demand: GlobalDemandSettings = field(default_factory=GlobalDemandSettings)
    time: TimeSettings = field(default_factory=TimeSettings)
    solver: HydraulicSolverSettings = field(default_factory=HydraulicSolverSettings)


@dataclass
class ServiceNodeMeta:
    node_name: str
    base_daily_volume_m3: float
    zone: str = "Z1"
    pattern_family_name: str = "DEFAULT_REFILL"


@dataclass
class LeakNodeMeta:
    node_name: str
    zone: str
    weight: float


@dataclass
class ModelMetadata:
    service_nodes: Dict[str, ServiceNodeMeta] = field(default_factory=dict)
    leak_nodes: Dict[str, LeakNodeMeta] = field(default_factory=dict)
    sensor_nodes: List[str] = field(default_factory=list)
    leak_check_node: Optional[str] = None
    pda_check_node: Optional[str] = None

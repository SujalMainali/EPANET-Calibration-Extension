"""Local builder helpers for metadata and default raw parameters.

These are examples/placeholders you should edit to match your network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import wntr

from calibration.datamodels import LeakNodeMeta, ModelMetadata, ServiceNodeMeta


def inspect_inp_units(inp_path: str) -> Dict[str, object]:
    """Inspect key unit/time settings from the INP via WNTR.

    Notes:
    - WNTR stores demands internally in SI (m^3/s) even if the INP uses LPM/GPM/etc.
    - The original INP units are available as wn.options.hydraulic.inpfile_units.
    """

    wn = wntr.network.WaterNetworkModel(str(Path(inp_path)))
    return {
        "inpfile_units": wn.options.hydraulic.inpfile_units,
        "pattern_timestep_s": int(getattr(wn.options.time, "pattern_timestep", 0) or 0),
        "duration_s": int(getattr(wn.options.time, "duration", 0) or 0),
        "report_timestep_s": int(getattr(wn.options.time, "report_timestep", 0) or 0),
    }


def extract_houseend_junctions_from_inp(
    inp_path: str,
    *,
    houseend_prefix: str = "HOUSEEND_",
) -> pd.DataFrame:
    """Extract HOUSEEND junction base demands + implied daily volumes.

    Returns a table with:
    - node_name
    - base_demand_m3s (WNTR internal units)
    - base_demand_lpm_equiv (derived from base_demand_m3s; useful when INP units are LPM)
    - pattern_name (if any)
    - daily_volume_m3 (computed using pattern multipliers over the INP duration)
    - inpfile_units (e.g., LPM)
    """

    wn = wntr.network.WaterNetworkModel(str(Path(inp_path)))
    inp_units = wn.options.hydraulic.inpfile_units
    duration_s = int(getattr(wn.options.time, "duration", 0) or 0)
    pattern_step_s = int(getattr(wn.options.time, "pattern_timestep", 0) or 0)
    if duration_s <= 0:
        duration_s = 24 * 3600
    if pattern_step_s <= 0:
        # Default to hourly; aligns with your typical report step.
        pattern_step_s = 3600

    rows = []
    for node_name in wn.junction_name_list:
        if not node_name.startswith(houseend_prefix):
            continue
        j = wn.get_node(node_name)
        if j is None:
            continue
        demand_list = getattr(j, "demand_timeseries_list", None)
        if not demand_list:
            continue

        d0 = demand_list[0]
        base_m3s = float(d0.base_value)
        base_lpm_equiv = base_m3s * 60.0 * 1000.0
        pattern_name = getattr(d0, "pattern_name", None)

        daily_volume_m3 = base_m3s * float(duration_s)
        if pattern_name and pattern_name in wn.pattern_name_list and pattern_step_s > 0:
            pat = wn.get_pattern(pattern_name)
            mult_raw = getattr(pat, "multipliers", None)
            if mult_raw is None:
                mult = np.asarray([], dtype=float)
            else:
                mult = np.asarray(mult_raw, dtype=float)
            if mult.size > 0:
                n_steps = int(duration_s // pattern_step_s)
                if n_steps <= 0:
                    n_steps = 24
                idx = np.arange(n_steps) % mult.size
                daily_volume_m3 = base_m3s * float(pattern_step_s) * float(mult[idx].sum())

        rows.append(
            {
                "node_name": node_name,
                "base_demand_m3s": base_m3s,
                "base_demand_lpm_equiv": float(base_lpm_equiv),
                "pattern_name": pattern_name,
                "daily_volume_m3": float(daily_volume_m3),
                "inpfile_units": inp_units,
            }
        )

    return pd.DataFrame(rows)


def build_service_nodes_from_inp(
    inp_path: str,
    *,
    sensor_nodes: Iterable[str],
    houseend_prefix: str = "HOUSEEND_",
    zone_by_node: Optional[Dict[str, str]] = None,
) -> Tuple[ModelMetadata, Dict[str, object]]:
    """Build ModelMetadata.service_nodes by reading the INP.

    - Service nodes: all junction IDs starting with HOUSEEND_
    - base_daily_volume_m3: computed from base demand and its pattern over the INP duration
    - Units: recorded in the returned info dict for consistency checks

    Leak nodes are left empty here (configure later once you decide leak calibration strategy).
    """

    zone_by_node = zone_by_node or {}
    info = inspect_inp_units(inp_path)
    df = extract_houseend_junctions_from_inp(inp_path, houseend_prefix=houseend_prefix)

    metadata = ModelMetadata()
    metadata.sensor_nodes = list(sensor_nodes)

    service_nodes: Dict[str, ServiceNodeMeta] = {}
    for _, r in df.iterrows():
        node_name = str(r["node_name"])
        zone = zone_by_node.get(node_name, "UNKNOWN")
        family = str(r.get("pattern_name") or "DEFAULT")
        service_nodes[node_name] = ServiceNodeMeta(
            node_name=node_name,
            base_daily_volume_m3=float(r["daily_volume_m3"]),
            zone=zone,
            pattern_family_name=family,
        )
    metadata.service_nodes = service_nodes
    metadata.leak_nodes = {}
    metadata.leak_check_node = None
    metadata.pda_check_node = (metadata.sensor_nodes[0] if metadata.sensor_nodes else None)

    info.update(
        {
            "service_node_count": int(len(metadata.service_nodes)),
            "service_daily_volume_m3_min": float(df["daily_volume_m3"].min()) if not df.empty else 0.0,
            "service_daily_volume_m3_max": float(df["daily_volume_m3"].max()) if not df.empty else 0.0,
        }
    )
    return metadata, info


def load_zone_mapping_csv(path: str) -> Dict[str, str]:
    """Load a node->zone mapping from a CSV.

    Expected columns (case-sensitive, but multiple aliases supported):
    - node_name (or id/ID/epanet_id/EPANET_ID)
    - zone (or Zone/ZONE)
    """

    df = pd.read_csv(path)

    def pick_col(candidates: list[str]) -> str:
        for c in candidates:
            if c in df.columns:
                return c
        raise ValueError(f"Missing required column; tried: {candidates}")

    id_col = pick_col(["node_name", "id", "ID", "epanet_id", "EPANET_ID"])
    zone_col = pick_col(["zone", "Zone", "ZONE"])

    out: Dict[str, str] = {}
    for _, r in df.iterrows():
        out[str(r[id_col])] = str(r[zone_col])
    return out


def build_zone_mapping_from_inp(
    inp_path: str,
    *,
    mode: str = "grid",
    node_prefix: str | None = None,
    zone_labels: tuple[str, str, str, str] = ("Z_NW", "Z_NE", "Z_SW", "Z_SE"),
) -> Dict[str, str]:
    """Build a node->zone mapping using coordinates embedded in the INP.

    Parameters
    - mode:
        - 'grid': split into 4 zones by median X/Y of the selected nodes
    - node_prefix: if provided, only nodes whose IDs start with this prefix are zoned
    - zone_labels: labels for (NW, NE, SW, SE)

    Notes
    - Requires the INP to contain a [COORDINATES] section.
    - Nodes missing coordinates are skipped.
    """

    if mode != "grid":
        raise ValueError("mode must be 'grid'")

    wn = wntr.network.WaterNetworkModel(str(Path(inp_path)))

    coords: dict[str, tuple[float, float]] = {}
    for node_name in wn.node_name_list:
        if node_prefix and not str(node_name).startswith(node_prefix):
            continue
        node = wn.get_node(node_name)
        if node is None:
            continue

        xy = getattr(node, "coordinates", None)
        if not xy or len(xy) < 2:
            continue
        try:
            x = float(xy[0])
            y = float(xy[1])
        except Exception:
            continue
        if not np.isfinite(x) or not np.isfinite(y):
            continue

        coords[str(node_name)] = (x, y)

    if not coords:
        return {}

    xs = np.asarray([v[0] for v in coords.values()], dtype=float)
    ys = np.asarray([v[1] for v in coords.values()], dtype=float)
    x_med = float(np.median(xs))
    y_med = float(np.median(ys))

    z_nw, z_ne, z_sw, z_se = zone_labels

    out: Dict[str, str] = {}
    for name, (x, y) in coords.items():
        north = y >= y_med
        east = x >= x_med
        if north and (not east):
            out[name] = z_nw
        elif north and east:
            out[name] = z_ne
        elif (not north) and (not east):
            out[name] = z_sw
        else:
            out[name] = z_se

    return out


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
        "solver": {
            "trials": 200,
            "accuracy": 0.001,
            "unbalanced": "CONTINUE",
            "damplimit": 0.1,
            "checkfreq": 2,
            "maxcheck": 10,
        },
    }

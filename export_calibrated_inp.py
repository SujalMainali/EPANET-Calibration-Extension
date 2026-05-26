"""Export a plain EPANET INP equivalent of the calibrated layered model.

This repository runs EPANET through a layered scaffold:
- ParameterizationLayer: raw dict -> typed ModelParameters
- BehaviorLayer: builds per-node dynamic demand patterns and leak grouping
- HydraulicModelLayerENepanet: applies options to a WNTR WaterNetworkModel

The exported INP aims to reproduce the same pressure heads/patterns when run in
standard EPANET, by materializing:
- [OPTIONS] time, PDA/PDD, solver controls, emitter exponent
- [PATTERNS] per-node dynamic patterns (dynpat_<node>)
- [JUNCTIONS] base demands + pattern assignment
- [EMITTERS] baseline leak emitters (derived from calibrated zone multipliers)

Limitations
-----------
- If carryover.enabled is True, the model demand depends on unmet demand from
  previous runs. The exporter writes the "unmet=0" (first-run) equivalent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import wntr

import config
from calibration.behavior_layer import BehaviorLayer
from calibration.builders import (
    build_leak_nodes_from_ids,
    build_service_nodes_from_inp,
    build_zone_mapping_from_inp,
    load_zone_mapping_csv,
)
from calibration.hydraulic_layer_enepanet import HydraulicModelLayerENepanet
from calibration.parameterization_layer import ParameterizationLayer


PSI_PER_M = 1.422334330  # 1 mH2O = 1.42233 psi


def _flow_to_m3s_factor(inp_units: str | None) -> float:
    """Return multiplier to convert EPANET flow/demand units -> m^3/s."""

    if not inp_units:
        return 1.0
    u = str(inp_units).strip().upper()

    # Metric
    if u == "CMS":
        return 1.0
    if u == "CMD":
        return 1.0 / 86400.0
    if u == "MLD":
        return 1000.0 / 86400.0
    if u == "LPS":
        return 1.0 / 1000.0
    if u == "LPM":
        return 1.0 / (1000.0 * 60.0)

    # US/Imperial
    if u == "CFS":
        return 0.028316846592
    if u == "GPM":
        return 0.003785411784 / 60.0
    if u == "MGD":
        return (1_000_000.0 * 0.003785411784) / 86400.0
    if u == "IMGD":
        return (1_000_000.0 * 0.00454609) / 86400.0
    if u == "AFD":
        return 1233.48183754752 / 86400.0

    return 1.0


def _pressure_unit_factor_for_inp_units(inp_units: str | None) -> float:
    """Return multiplier for converting internal meters -> INP pressure units.

    WNTR stores pressure/head internally in meters.
    - Metric flow units (LPS/LPM/CMS/...) use meters in INP.
    - US flow units (GPM/MGD/...) use psi in INP.
    """

    if not inp_units:
        return 1.0

    u = str(inp_units).strip().upper()
    if u in {"CFS", "GPM", "MGD", "IMGD", "AFD"}:
        return PSI_PER_M
    return 1.0


def _load_raw_params(path: str) -> Dict[str, Any]:
    """Load raw params from either optimize.py output or a raw params dict."""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Params JSON not found: {str(p.resolve())}")

    data = json.loads(p.read_text())

    if isinstance(data, dict) and "best_raw_params" in data:
        rp = data["best_raw_params"]
        if not isinstance(rp, dict):
            raise ValueError("best_raw_params must be a dict")

        # Preserve multi-day calibration horizon if present.
        n_days = data.get("n_days")
        try:
            n_days_i = int(n_days) if n_days is not None else None
        except Exception:
            n_days_i = None
        if n_days_i and n_days_i > 0:
            rp = dict(rp)
            rp.setdefault("time", {})
            rp["time"]["duration_days"] = int(max(1, n_days_i))

        return rp

    if isinstance(data, dict) and any(k in data for k in ("pda", "demand", "leakage", "time", "solver")):
        return data

    raise ValueError("Unrecognized params JSON format")


def _build_metadata_for_inp(inp_path: str):
    """Build ModelMetadata the same way as config.build_default_metadata(), but for a specific INP."""

    zone_by_node = None

    if config.ZONE_MAP_CSV:
        zone_by_node = load_zone_mapping_csv(config.ZONE_MAP_CSV)
    elif config.AUTO_ZONES_FROM_INP:
        zone_by_node = build_zone_mapping_from_inp(
            inp_path,
            mode="voronoi",
            node_prefix=config.ZONE_NODE_PREFIX,
            sensor_nodes=config.SENSOR_NODES,
        )

    metadata, info = build_service_nodes_from_inp(
        inp_path,
        sensor_nodes=config.SENSOR_NODES,
        zone_by_node=zone_by_node,
        ignore_unmapped_zones=bool(config.ZONE_IGNORE_UNASSIGNED_NODES),
    )

    leak_ids: list[str] = []
    if config.LEAK_NODES_FROM_ZONES and zone_by_node:
        wn = wntr.network.WaterNetworkModel(inp_path)
        zoned = set(str(k) for k in zone_by_node.keys())
        for jname in wn.junction_name_list:
            sj = str(jname)
            if sj in zoned:
                leak_ids.append(sj)

    leak_ids.extend(list(config.LEAK_NODE_IDS))

    if config.LEAK_NODE_PREFIXES:
        wn = wntr.network.WaterNetworkModel(inp_path)
        for nid in wn.node_name_list:
            sid = str(nid)
            if any(sid.startswith(pref) for pref in config.LEAK_NODE_PREFIXES):
                leak_ids.append(sid)

    seen = set()
    leak_ids = [x for x in leak_ids if not (x in seen or seen.add(x))]

    if leak_ids:
        metadata.leak_nodes = build_leak_nodes_from_ids(
            inp_path,
            leak_ids,
            zone_by_node=zone_by_node,
            default_weight=float(config.LEAK_NODE_DEFAULT_WEIGHT),
        )
        metadata.leak_check_node = next(iter(metadata.leak_nodes.keys()), None)

    if config.VERBOSE:
        print("[metadata]", info)

    return metadata


def _apply_baseline_emitters_to_wn(
    wn_model: "wntr.network.WaterNetworkModel",
    behavior: BehaviorLayer,
    params,
) -> Tuple[int, int]:
    """Apply baseline leak emitter coefficients to junctions and return stats.

    The calibration scaffold computes emitter coefficients in EPANET INP units,
    because it applies them via the toolkit (ENsetnodevalue(EMITTER)).
    WNTR stores emitters internally in SI (m^3/s per m^n), so we convert.
    """

    emitter_coeffs_inp = behavior.grouped_emitter_coefficients(params)
    if not emitter_coeffs_inp:
        return 0, 0

    inp_units = getattr(wn_model.options.hydraulic, "inpfile_units", None)
    q_to_m3s = _flow_to_m3s_factor(inp_units)
    p_factor = _pressure_unit_factor_for_inp_units(inp_units)
    n = float(getattr(wn_model.options.hydraulic, "emitter_exponent", 0.5) or 0.5)

    applied = 0
    missing = 0

    for node_name, c_inp in emitter_coeffs_inp.items():
        try:
            node = wn_model.get_node(node_name)
        except Exception:
            node = None

        if node is None:
            missing += 1
            continue

        c_inp_f = float(c_inp)
        if c_inp_f == 0.0:
            # Explicitly clear in case the base INP had emitters.
            node.emitter_coefficient = 0.0
            applied += 1
            continue

        # Convert coefficient from INP units -> WNTR internal SI:
        # C_inp = C_si * (flow_factor) / (pressure_factor^n)
        # where flow_factor converts m^3/s -> flow_units.
        # Since q_to_m3s converts flow_units -> m^3/s, flow_factor = 1/q_to_m3s.
        # => C_si = C_inp * q_to_m3s * (pressure_factor^n)
        c_si = c_inp_f * float(q_to_m3s) * float(p_factor**n)
        node.emitter_coefficient = float(c_si)
        applied += 1

    return applied, missing


def export_calibrated_inp(
    *,
    base_inp_path: str,
    params_json_path: str,
    out_inp_path: str,
) -> Path:
    base_inp = Path(base_inp_path)
    if not base_inp.exists():
        raise FileNotFoundError(f"Base INP not found: {str(base_inp.resolve())}")

    raw_params = _load_raw_params(params_json_path)

    metadata = _build_metadata_for_inp(str(base_inp))
    parameterization = ParameterizationLayer(metadata)
    params = parameterization.from_dict(raw_params)
    behavior = BehaviorLayer(metadata)

    hydraulic = HydraulicModelLayerENepanet(str(base_inp))
    wn_model = hydraulic.clone_network()
    hydraulic.apply_pda_settings_to_inp_model(wn_model, params)
    hydraulic.apply_service_node_demands(wn_model, metadata, behavior, params)

    applied, missing = _apply_baseline_emitters_to_wn(wn_model, behavior, params)
    if params.carryover.enabled:
        print("[warn] carryover.enabled=True: exported INP uses first-run (unmet=0) demands")
    if missing:
        print(f"[warn] emitter nodes missing in INP: {missing} (skipped)")

    out_path = Path(out_inp_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wntr.network.io.write_inpfile(wn_model, out_path)

    print(f"Wrote calibrated INP: {str(out_path)}")
    print(f"Emitters applied: {applied}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Export an EPANET INP equivalent of calibrated model parameters")
    ap.add_argument(
        "--inp",
        default=config.MODEL_INP,
        help="Base EPANET INP (defaults to config.MODEL_INP)",
    )
    ap.add_argument(
        "--params",
        default=str(Path(config.REPORTS_DIR) / "best_params.json"),
        help="Calibration params JSON (optimize.py output best_params.json)",
    )
    ap.add_argument(
        "--out",
        default=str(Path(config.REPORTS_DIR) / "calibrated_model.inp"),
        help="Output INP path",
    )
    args = ap.parse_args()

    export_calibrated_inp(base_inp_path=args.inp, params_json_path=args.params, out_inp_path=args.out)


if __name__ == "__main__":
    main()

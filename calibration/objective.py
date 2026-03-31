"""Objective function for calibrating the Kathmandu-style double-buffered system.

This module defines a composite objective

    J(θ) = w_ts * J_timeseries + w_feat * J_features + w_sp * J_spatial
           + w_vol * J_volume + w_reg * J_regularization

It is written to work directly with this repo's outputs:
- simulated pressure: RunResults['pressure'] (DataFrame indexed by time_s)
- simulated demand:   RunResults['demand']   (DataFrame indexed by time_s, m^3/s)

Observed pressures should be provided as a DataFrame with columns containing sensor
node names and an index representing time in seconds (preferred) or datetimes.

The objective returns both the scalar value and a breakdown dict for diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from calibration.datamodels import ModelMetadata, ModelParameters



_EPS = 1e-12


@dataclass(frozen=True)
class ObjectiveWeights:
    w_ts: float = 0.40
    w_feat: float = 0.30
    w_sp: float = 0.15
    w_vol: float = 0.10
    w_reg: float = 0.05


@dataclass(frozen=True)
class TimeseriesFitConfig:
    """Pointwise pressure fit config."""

    alpha_rmse: float = 0.5
    eps: float = 1e-6


@dataclass(frozen=True)
class FeatureFitConfig:
    """Curve feature extraction and weighting."""

    drop_frac_of_range: float = 0.20
    # feature weights (a1..a6 in your write-up)
    w_pmin: float = 1.0
    w_pmax: float = 0.5
    w_tdip: float = 1.0
    w_trec: float = 0.75
    w_dlow: float = 0.75
    w_auc: float = 0.5
    # include slopes (optional but useful for double-buffered dynamics)
    w_slope_drop: float = 0.25
    w_slope_rec: float = 0.25


@dataclass(frozen=True)
class SpatialFitConfig:
    """Spatial pressure relationship fit config."""

    max_pairs: Optional[int] = None  # None = all pairs


@dataclass(frozen=True)
class VolumeFitConfig:
    """Delivered-volume (PDA + carryover consistency) term."""

    eps: float = 1e-6


@dataclass(frozen=True)
class RegularizationPrior:
    target: float
    scale: float


@dataclass
class RegularizationConfig:
    """Simple L2 priors on parameters (dimensionless after scaling)."""

    priors: Dict[str, RegularizationPrior] = field(default_factory=dict)


@dataclass
class ObjectiveConfig:
    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    timeseries: TimeseriesFitConfig = field(default_factory=TimeseriesFitConfig)
    features: FeatureFitConfig = field(default_factory=FeatureFitConfig)
    spatial: SpatialFitConfig = field(default_factory=SpatialFitConfig)
    volume: VolumeFitConfig = field(default_factory=VolumeFitConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)


def load_observed_pressure_csv(path: str) -> pd.DataFrame:
    """Load observed pressures from CSV.

    Expected format:
    - first column is time (either 'time_s' as integer seconds, or a datetime string)
    - remaining columns are sensor node pressures

    The returned DataFrame uses the first column as index.
    """

    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError("Observed pressure CSV must have >= 2 columns (time + sensors)")

    time_col = df.columns[0]
    df = df.set_index(time_col)

    # Try to coerce to numeric seconds; fallback to datetime.
    # (Pass a numpy array to satisfy type checkers and pandas overloads.)
    try:
        df.index = pd.Index(pd.to_numeric(df.index.to_numpy()))
    except Exception:
        df.index = pd.to_datetime(df.index)

    return df


def build_default_regularization_from_params(params: ModelParameters, metadata: ModelMetadata) -> RegularizationConfig:
    """Create conservative priors centered on the provided params.

    This makes the regularization term usable immediately without the user
    needing to specify priors. You should adjust scales to match local knowledge.
    """

    priors: Dict[str, RegularizationPrior] = {
        "pda.minimum_pressure": RegularizationPrior(params.pda.minimum_pressure, scale=5.0),
        "pda.required_pressure": RegularizationPrior(params.pda.required_pressure, scale=10.0),
        "pda.pressure_exponent": RegularizationPrior(params.pda.pressure_exponent, scale=0.5),
        "demand.demand_multiplier": RegularizationPrior(params.demand.demand_multiplier, scale=0.5),
        "carryover.catchup_factor": RegularizationPrior(params.carryover.catchup_factor, scale=0.5),
        "carryover.max_carryover_multiplier": RegularizationPrior(params.carryover.max_carryover_multiplier, scale=1.0),
        "pattern_family.morning_center": RegularizationPrior(params.pattern_family.morning_center, scale=2.0),
        "pattern_family.morning_width": RegularizationPrior(params.pattern_family.morning_width, scale=1.0),
        "pattern_family.morning_weight": RegularizationPrior(params.pattern_family.morning_weight, scale=0.3),
        "pattern_family.noon_center": RegularizationPrior(params.pattern_family.noon_center, scale=2.0),
        "pattern_family.noon_width": RegularizationPrior(params.pattern_family.noon_width, scale=1.0),
        "pattern_family.noon_weight": RegularizationPrior(params.pattern_family.noon_weight, scale=0.3),
        "pattern_family.evening_center": RegularizationPrior(params.pattern_family.evening_center, scale=2.0),
        "pattern_family.evening_width": RegularizationPrior(params.pattern_family.evening_width, scale=1.0),
        "pattern_family.evening_weight": RegularizationPrior(params.pattern_family.evening_weight, scale=0.3),
        "pattern_family.background_weight": RegularizationPrior(params.pattern_family.background_weight, scale=0.3),
        "pattern_family.floor": RegularizationPrior(params.pattern_family.floor, scale=0.05),
    }

    # Zone leakage multipliers: default prior is current value if present,
    # otherwise 1.0 (meaning "nominal"), with a fairly wide scale.
    zones: set[str] = set()
    for svc in metadata.service_nodes.values():
        zones.add(svc.zone)
    for lk in metadata.leak_nodes.values():
        zones.add(lk.zone)

    for zone in sorted(zones):
        zval = float(params.leakage.zone_multipliers.get(zone, 1.0))
        priors[f"leakage.zone_multipliers.{zone}"] = RegularizationPrior(zval, scale=2.0)

    if params.leakage.emitter_exponent is not None:
        priors["leakage.emitter_exponent"] = RegularizationPrior(float(params.leakage.emitter_exponent), scale=0.5)

    return RegularizationConfig(priors=priors)


def compute_objective(
    *,
    params: ModelParameters,
    metadata: ModelMetadata,
    results: Mapping[str, Any],
    observed_pressure: pd.DataFrame,
    config: Optional[ObjectiveConfig] = None,
    sensor_weights: Optional[Mapping[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Compute J(θ) and return (scalar, breakdown).

    breakdown contains unweighted and weighted components.
    """

    if config is None:
        config = ObjectiveConfig()

    if not config.regularization.priors:
        config.regularization = build_default_regularization_from_params(params, metadata)

    sim_pressure = results["pressure"]
    sim_demand = results["demand"]

    if not isinstance(sim_pressure, pd.DataFrame) or not isinstance(sim_demand, pd.DataFrame):
        raise TypeError("results must contain DataFrames under keys 'pressure' and 'demand'")

    obs_pressure = _coerce_index_to_seconds(observed_pressure.copy())
    sim_pressure = _coerce_index_to_seconds(sim_pressure.copy())
    sim_demand = _coerce_index_to_seconds(sim_demand.copy())

    sensors = _select_sensors(metadata, obs_pressure, sim_pressure)
    sw = _build_sensor_weights(sensors, sensor_weights)

    obs_aligned, sim_aligned = _align_on_observed_index(obs_pressure[sensors], sim_pressure[sensors])

    j_ts = _j_timeseries(obs_aligned, sim_aligned, sw, config.timeseries)
    j_feat = _j_features(obs_aligned, sim_aligned, sw, config.features)
    j_sp = _j_spatial(obs_aligned, sim_aligned, sw, config.spatial)
    j_vol = _j_volume(params, metadata, sim_demand, config.volume)
    j_reg = _j_regularization(params, config.regularization)

    w = config.weights
    j_total = w.w_ts * j_ts + w.w_feat * j_feat + w.w_sp * j_sp + w.w_vol * j_vol + w.w_reg * j_reg

    breakdown = {
        "J_total": float(j_total),
        "J_timeseries": float(j_ts),
        "J_features": float(j_feat),
        "J_spatial": float(j_sp),
        "J_volume": float(j_vol),
        "J_regularization": float(j_reg),
        "w_ts": float(w.w_ts),
        "w_feat": float(w.w_feat),
        "w_sp": float(w.w_sp),
        "w_vol": float(w.w_vol),
        "w_reg": float(w.w_reg),
        "wJ_timeseries": float(w.w_ts * j_ts),
        "wJ_features": float(w.w_feat * j_feat),
        "wJ_spatial": float(w.w_sp * j_sp),
        "wJ_volume": float(w.w_vol * j_vol),
        "wJ_regularization": float(w.w_reg * j_reg),
    }

    return float(j_total), breakdown


def _coerce_index_to_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce an index of time into numeric seconds since start."""

    if df.index.dtype.kind in {"i", "u", "f"}:
        df.index = pd.Index(pd.to_numeric(df.index.to_numpy()))
        return df

    # datetime-like
    try:
        dt = pd.to_datetime(df.index)
    except Exception as e:
        raise TypeError("Unsupported time index; use numeric seconds or datetime index") from e

    t0 = dt.min()
    seconds = (dt - t0).total_seconds()
    df.index = pd.Index(pd.to_numeric(np.asarray(seconds)))
    return df


def _select_sensors(metadata: ModelMetadata, obs: pd.DataFrame, sim: pd.DataFrame) -> list[str]:
    wanted = list(metadata.sensor_nodes)
    sensors = [n for n in wanted if (n in obs.columns and n in sim.columns)]
    if sensors:
        return sensors

    # fallback: intersection of columns
    inter = [n for n in obs.columns if n in sim.columns]
    if not inter:
        raise ValueError("No overlapping sensor columns between observed_pressure and simulated pressure")
    return inter


def _build_sensor_weights(sensors: Iterable[str], sensor_weights: Optional[Mapping[str, float]]) -> Dict[str, float]:
    if sensor_weights is None:
        out = {s: 1.0 for s in sensors}
    else:
        out = {s: float(sensor_weights.get(s, 1.0)) for s in sensors}

    total = sum(max(0.0, v) for v in out.values())
    if total <= 0:
        return {s: 1.0 / max(1, len(out)) for s in out}
    return {k: max(0.0, v) / total for k, v in out.items()}


def _align_on_observed_index(obs: pd.DataFrame, sim: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align sim to obs times using index interpolation on numeric seconds."""

    obs = obs.sort_index()
    sim = sim.sort_index()

    # Reindex sim to obs index and interpolate using numeric index.
    # For flat extrapolation outside sim range, use nearest then ffill/bfill.
    sim_r = sim.reindex(obs.index)
    try:
        sim_r = sim_r.interpolate(method="index", limit_direction="both")
    except Exception:
        sim_r = sim_r.ffill().bfill()

    return obs, sim_r


def _finite_mask(a: pd.Series, b: pd.Series) -> np.ndarray:
    av = a.to_numpy(dtype=float)
    bv = b.to_numpy(dtype=float)
    return np.isfinite(av) & np.isfinite(bv)


def _nrmse(obs: pd.Series, sim: pd.Series, eps: float) -> float:
    mask = _finite_mask(obs, sim)
    if not mask.any():
        return 0.0
    o = obs.to_numpy(dtype=float)[mask]
    s = sim.to_numpy(dtype=float)[mask]
    r = float(np.nanmax(o) - np.nanmin(o))
    denom = r + eps
    return float(np.sqrt(np.mean((s - o) ** 2)) / denom)


def _nmae(obs: pd.Series, sim: pd.Series, eps: float) -> float:
    mask = _finite_mask(obs, sim)
    if not mask.any():
        return 0.0
    o = obs.to_numpy(dtype=float)[mask]
    s = sim.to_numpy(dtype=float)[mask]
    r = float(np.nanmax(o) - np.nanmin(o))
    denom = r + eps
    return float(np.mean(np.abs(s - o)) / denom)


def _j_timeseries(obs: pd.DataFrame, sim: pd.DataFrame, sw: Mapping[str, float], cfg: TimeseriesFitConfig) -> float:
    total = 0.0
    for n, w in sw.items():
        nrmse = _nrmse(obs[n], sim[n], cfg.eps)
        nmae = _nmae(obs[n], sim[n], cfg.eps)
        total += float(w) * (cfg.alpha_rmse * nrmse + (1.0 - cfg.alpha_rmse) * nmae)
    return float(total)


def _extract_features(time_s: np.ndarray, p: np.ndarray, drop_frac: float) -> Dict[str, float]:
    """Extract pressure curve features from a single node timeseries."""

    mask = np.isfinite(time_s) & np.isfinite(p)
    if not mask.any():
        return {
            "pmin": 0.0,
            "pmax": 0.0,
            "pmean": 0.0,
            "tmin_s": 0.0,
            "tdip_s": 0.0,
            "trec_s": 0.0,
            "dlow_s": 0.0,
            "auc": 0.0,
            "slope_drop": 0.0,
            "slope_rec": 0.0,
            "range": 0.0,
            "duration_s": 0.0,
        }

    t = np.asarray(time_s[mask], dtype=float)
    y = np.asarray(p[mask], dtype=float)
    order = np.argsort(t)
    t = t[order]
    y = y[order]

    pmin = float(np.min(y))
    pmax = float(np.max(y))
    pmean = float(np.mean(y))
    prange = float(pmax - pmin)

    duration_s = float(max(_EPS, t[-1] - t[0]))

    imin = int(np.argmin(y))
    tmin = float(t[imin])

    thresh = pmean - drop_frac * prange

    # time of first major drop: first time going below threshold, else use tmin.
    below = np.where(y <= thresh)[0]
    if below.size > 0:
        tdip = float(t[int(below[0])])
    else:
        tdip = tmin

    # recovery time: first time after min when back above threshold, else end.
    after_min = np.where((t >= tmin) & (y >= thresh))[0]
    if after_min.size > 0:
        trec = float(t[int(after_min[0])])
    else:
        trec = float(t[-1])

    # duration below threshold: approximate using median dt.
    if len(t) >= 2:
        dt = float(np.median(np.diff(t)))
    else:
        dt = duration_s
    dlow = float(np.sum(y <= thresh) * dt)

    # AUC using trapezoidal integration.
    # NumPy 2.x removed np.trapz; prefer np.trapezoid, with a fallback.
    trapz_fn = getattr(np, "trapezoid", None)
    if trapz_fn is None:  # pragma: no cover
        trapz_fn = getattr(np, "trapz", None)
    if trapz_fn is None:  # pragma: no cover
        raise RuntimeError("No trapezoidal integration function available in NumPy")
    auc = float(trapz_fn(y, t))

    # Slopes into dip and out of dip.
    # Drop slope from tdip to tmin.
    y_dip = float(np.interp(tdip, t, y))
    y_rec = float(np.interp(trec, t, y))

    slope_drop = float((pmin - y_dip) / max(_EPS, (tmin - tdip)))
    slope_rec = float((y_rec - pmin) / max(_EPS, (trec - tmin)))

    return {
        "pmin": pmin,
        "pmax": pmax,
        "pmean": pmean,
        "tmin_s": tmin,
        "tdip_s": tdip,
        "trec_s": trec,
        "dlow_s": dlow,
        "auc": auc,
        "slope_drop": slope_drop,
        "slope_rec": slope_rec,
        "range": prange,
        "duration_s": duration_s,
    }


def _j_features(obs: pd.DataFrame, sim: pd.DataFrame, sw: Mapping[str, float], cfg: FeatureFitConfig) -> float:
    weights = {
        "pmin": cfg.w_pmin,
        "pmax": cfg.w_pmax,
        "tdip": cfg.w_tdip,
        "trec": cfg.w_trec,
        "dlow": cfg.w_dlow,
        "auc": cfg.w_auc,
        "slope_drop": cfg.w_slope_drop,
        "slope_rec": cfg.w_slope_rec,
    }
    wsum = float(sum(weights.values()))
    if wsum <= 0:
        return 0.0

    t = obs.index.to_numpy(dtype=float)

    total = 0.0
    for n, w in sw.items():
        fo = _extract_features(t, obs[n].to_numpy(dtype=float), cfg.drop_frac_of_range)
        fs = _extract_features(t, sim[n].to_numpy(dtype=float), cfg.drop_frac_of_range)

        pr = float(fo["range"]) + 1e-6
        dur = float(max(fo["duration_s"], 24 * 3600.0))

        # Normalize each delta into a dimensionless quantity.
        d_pmin = abs(fs["pmin"] - fo["pmin"]) / pr
        d_pmax = abs(fs["pmax"] - fo["pmax"]) / pr
        d_tdip = abs(fs["tdip_s"] - fo["tdip_s"]) / dur
        d_trec = abs(fs["trec_s"] - fo["trec_s"]) / dur
        d_dlow = abs(fs["dlow_s"] - fo["dlow_s"]) / dur

        denom_auc = abs(fo["auc"]) + 1e-6
        d_auc = abs(fs["auc"] - fo["auc"]) / denom_auc

        denom_slope = (pr / dur) + 1e-6
        d_slope_drop = abs(fs["slope_drop"] - fo["slope_drop"]) / denom_slope
        d_slope_rec = abs(fs["slope_rec"] - fo["slope_rec"]) / denom_slope

        term = (
            weights["pmin"] * d_pmin
            + weights["pmax"] * d_pmax
            + weights["tdip"] * d_tdip
            + weights["trec"] * d_trec
            + weights["dlow"] * d_dlow
            + weights["auc"] * d_auc
            + weights["slope_drop"] * d_slope_drop
            + weights["slope_rec"] * d_slope_rec
        ) / wsum

        total += float(w) * float(term)

    return float(total)


def _pairs(items: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            out.append((items[i], items[j]))
    return out


def _j_spatial(obs: pd.DataFrame, sim: pd.DataFrame, sw: Mapping[str, float], cfg: SpatialFitConfig) -> float:
    sensors = list(sw.keys())
    if len(sensors) < 2:
        return 0.0

    pairs = _pairs(sensors)
    if cfg.max_pairs is not None and len(pairs) > cfg.max_pairs:
        # deterministic downsample: take first N pairs
        pairs = pairs[: cfg.max_pairs]

    per_pair = []
    for i, j in pairs:
        o = (obs[i] - obs[j]).to_numpy(dtype=float)
        s = (sim[i] - sim[j]).to_numpy(dtype=float)
        mask = np.isfinite(o) & np.isfinite(s)
        if not mask.any():
            continue

        # Normalize by typical observed pressure range of the pair.
        oi = obs[i].to_numpy(dtype=float)
        oj = obs[j].to_numpy(dtype=float)
        r = float(max(np.nanmax(oi) - np.nanmin(oi), np.nanmax(oj) - np.nanmin(oj)))
        denom = r + 1e-6
        per_pair.append(float(np.mean(np.abs(s[mask] - o[mask])) / denom))

    if not per_pair:
        return 0.0

    return float(np.mean(per_pair))


def _j_volume(params: ModelParameters, metadata: ModelMetadata, demand_df: pd.DataFrame, cfg: VolumeFitConfig) -> float:
    if demand_df.empty or not metadata.service_nodes:
        return 0.0

    dt = float(params.time.report_timestep_s)
    days = float(max(1, int(getattr(params.time, "duration_days", 1) or 1)))

    errs = []
    for node_name, meta in metadata.service_nodes.items():
        # Compare delivered vs target total volume over the simulated horizon.
        target_m3 = float(meta.base_daily_volume_m3) * float(params.demand.demand_multiplier) * days
        if target_m3 <= 0:
            continue

        if node_name in demand_df.columns:
            delivered_m3 = float(demand_df[node_name].fillna(0.0).sum()) * dt
        else:
            delivered_m3 = 0.0

        rel = (delivered_m3 - target_m3) / (target_m3 + cfg.eps)
        errs.append(float(rel * rel))

    if not errs:
        return 0.0
    return float(np.mean(errs))


def _get_param_by_path(params: ModelParameters, path: str) -> float:
    # Special-case zone multipliers: leakage.zone_multipliers.Z1
    if path.startswith("leakage.zone_multipliers."):
        zone = path.split(".")[-1]
        return float(params.leakage.zone_multipliers.get(zone, 0.0))

    # Standard nested dataclass field lookup.
    cur: Any = params
    for part in path.split("."):
        if not hasattr(cur, part):
            raise KeyError(f"Unknown parameter path: {path}")
        cur = getattr(cur, part)
    if cur is None:
        return 0.0
    return float(cur)


def _j_regularization(params: ModelParameters, cfg: RegularizationConfig) -> float:
    if not cfg.priors:
        return 0.0

    terms = []
    for path, prior in cfg.priors.items():
        try:
            x = _get_param_by_path(params, path)
        except KeyError:
            continue
        scale = float(prior.scale) if float(prior.scale) != 0.0 else 1.0
        z = (float(x) - float(prior.target)) / scale
        terms.append(float(z * z))

    if not terms:
        return 0.0
    return float(np.mean(terms))

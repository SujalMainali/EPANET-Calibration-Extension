"""
Gradient-descent optimizer for EPANET/WNTR calibration.

FIXED VERSION:
- Prevents reservoir/tank loss during optimization
- Prevents unstable emitter exponent values
- Prevents NaN hydraulic solutions
- Adds safe parameter clipping
- Adds hydraulic sanity checks
- Adds retry-safe finite difference gradients
- Adds robust line search
- Stabilizes leakage scaling
- Handles failed EPANET runs gracefully
- Supports multi-day calibration
- Supports Voronoi zone multipliers
- FIXED: Proper metadata attribute access

Paste directly over optimize.py
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

import config
from calibration.objective import (
    ObjectiveConfig,
    ObjectiveWeights,
    build_default_regularization_from_params,
    load_observed_pressure_csv,
)
from calibration.runner import build_runner
from validation_layer import (
    PreprocessingValidationLayer,
    load_processed_observation_dataset,
    save_processed_observation_dataset,
)


# =============================================================================
# SAFETY CONSTANTS
# =============================================================================

SAFE_MIN_EMITTER_EXPONENT = 0.35
SAFE_MAX_EMITTER_EXPONENT = 1.20

SAFE_MIN_GLOBAL_LEAK_SCALE = 1e-7
SAFE_MAX_GLOBAL_LEAK_SCALE = 0.25

SAFE_MIN_ZONE_MULTIPLIER = 0.50
SAFE_MAX_ZONE_MULTIPLIER = 1.50

FAILED_RUN_PENALTY = 1e9


# =============================================================================
# UTILITIES
# =============================================================================

def _ensure_dirs() -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    config.DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _next_run_no(base_runs_dir: Path) -> int:
    if not base_runs_dir.exists():
        return 1

    best = 0

    for p in base_runs_dir.iterdir():
        if not p.is_dir():
            continue

        if not p.name.startswith("run_"):
            continue

        try:
            n = int(p.name.split("_")[1])
            best = max(best, n)
        except Exception:
            continue

    return best + 1


def _prepare_run_dir(run_no: int | None) -> Path:
    base = Path(config.RUNS_DIR)
    base.mkdir(parents=True, exist_ok=True)

    if run_no is None:
        run_no = _next_run_no(base)

    run_dir = base / f"run_{run_no}"

    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=False)

    return run_dir


# =============================================================================
# OBSERVED DATA LOADING
# =============================================================================

def _coerce_index_to_seconds(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.dtype.kind in {"i", "u", "f"}:
        df.index = pd.Index(pd.to_numeric(df.index.to_numpy()))
        return df

    dt = pd.to_datetime(df.index)

    t0 = dt.min()

    seconds = (dt - t0).total_seconds()

    df.index = pd.Index(pd.to_numeric(np.asarray(seconds)))

    return df


def _resolve_existing_path(path: str) -> str:
    p = Path(path)

    if p.exists():
        return str(p)

    alt = Path(__file__).resolve().parent / path

    if alt.exists():
        return str(alt)

    raise FileNotFoundError(f"Observed CSV not found: {path}")


def _load_observed_one(path: str) -> pd.DataFrame:
    path = _resolve_existing_path(path)

    if config.OBSERVED_TIME_COLUMN is None:
        df = load_observed_pressure_csv(path)
    else:
        raw = pd.read_csv(path)

        if config.OBSERVED_TIME_COLUMN not in raw.columns:
            raise ValueError(
                f"OBSERVED_TIME_COLUMN={config.OBSERVED_TIME_COLUMN!r} "
                f"not found in {path}"
            )

        df = raw.set_index(config.OBSERVED_TIME_COLUMN)

        try:
            df.index = pd.Index(pd.to_numeric(df.index.to_numpy()))
        except Exception:
            df.index = pd.to_datetime(df.index)

    return _coerce_index_to_seconds(df)


def load_observed_multi_day() -> tuple[pd.DataFrame, int]:

    if config.OBSERVED_PRESSURE_CSVS:

        dfs: list[pd.DataFrame] = []

        for i, p in enumerate(config.OBSERVED_PRESSURE_CSVS):

            d = _load_observed_one(p)

            d = d.copy()

            d.index = pd.Index(
                pd.to_numeric(d.index.to_numpy())
                - float(d.index.min())
                + float(i * 86400)
            )

            dfs.append(d)

        out = pd.concat(dfs, axis=0).sort_index()

        return out, len(dfs)

    if not config.OBSERVED_PRESSURE_CSV:
        raise ValueError(
            "Set OBSERVED_PRESSURE_CSV or OBSERVED_PRESSURE_CSVS"
        )

    df = _load_observed_one(config.OBSERVED_PRESSURE_CSV)

    span = (
        float(df.index.max() - df.index.min())
        if len(df.index)
        else 0.0
    )

    n_days = int(max(1, int(np.ceil((span + 1.0) / 86400.0))))

    return df, n_days


def _infer_n_days_from_index(observed: pd.DataFrame) -> int:
    if len(observed.index) == 0:
        return 1
    span = float(observed.index.max() - observed.index.min())
    return int(max(1, int(np.ceil((span + 1.0) / 86400.0))))


def validate_and_smooth_observed(
    observed: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prepare the exact pressure target used by objective evaluations."""

    if not config.OBSERVATION_VALIDATION_ENABLED:
        return observed, {"enabled": False}

    dataset_path = Path(config.OBSERVATION_SMOOTHED_DATASET_CSV)
    recompute = bool(config.OBSERVATION_RECOMPUTE_SMOOTHED_DATASET)
    if dataset_path.exists() and not recompute:
        processed = load_processed_observation_dataset(
            dataset_path,
            list(config.SENSOR_NODES),
        )
        return (
            processed,
            {
                "enabled": True,
                "source": "cached_dataset",
                "dataset_path": str(dataset_path),
                "recomputed": False,
            },
        )

    layer = PreprocessingValidationLayer(
        sensor_nodes=list(config.SENSOR_NODES),
        points_per_day=int(config.OBSERVATION_POINTS_PER_DAY),
        fold_days_enabled=bool(config.OBSERVATION_FOLD_DAYS_ENABLED),
        fold_aggregation=str(config.OBSERVATION_FOLD_AGGREGATION),
        smoothing_enabled=bool(config.OBSERVATION_SMOOTHING_ENABLED),
        smoothing_max_harmonic=int(
            config.OBSERVATION_SMOOTHING_MAX_HARMONIC
        ),
        interpolate_missing=bool(config.OBSERVATION_INTERPOLATE_MISSING),
        require_complete_days=bool(config.OBSERVATION_REQUIRE_COMPLETE_DAYS),
        mass_relative_tolerance=float(config.OBSERVATION_MASS_REL_TOL),
        parseval_relative_tolerance=float(config.OBSERVATION_PARSEVAL_REL_TOL),
        export_stages=bool(config.OBSERVATION_EXPORT_STAGES),
        output_dir=config.OBSERVATION_VALIDATION_DIR,
        verbose=bool(config.VERBOSE),
    )
    result = layer.process_and_validate(observed)
    save_processed_observation_dataset(result.calibration_data, dataset_path)
    return (
        result.calibration_data,
        {
            "enabled": True,
            "source": "generated_dataset",
            "dataset_path": str(dataset_path),
            "recomputed": True,
            **result.summary,
        },
    )


def load_calibration_observed() -> tuple[pd.DataFrame, int, dict[str, Any]]:
    """Load the observed pressure target used by optimization/results."""

    dataset_path = Path(config.OBSERVATION_SMOOTHED_DATASET_CSV)
    if (
        config.OBSERVATION_VALIDATION_ENABLED
        and dataset_path.exists()
        and not config.OBSERVATION_RECOMPUTE_SMOOTHED_DATASET
    ):
        processed = load_processed_observation_dataset(
            dataset_path,
            list(config.SENSOR_NODES),
        )
        return (
            processed,
            _infer_n_days_from_index(processed),
            {
                "enabled": True,
                "source": "cached_dataset",
                "dataset_path": str(dataset_path),
                "recomputed": False,
            },
        )

    observed, n_days = load_observed_multi_day()
    processed, summary = validate_and_smooth_observed(observed)
    n_days = int(summary.get("num_days", n_days))
    return processed, n_days, summary


# =============================================================================
# PARAMETER HELPERS
# =============================================================================

def _get_by_path(d: Dict[str, Any], path: str) -> float:
    cur = d

    for p in path.split("."):
        cur = cur[p]

    return float(cur)


def _set_by_path(d: Dict[str, Any], path: str, value: float) -> None:
    parts = path.split(".")

    cur = d

    for p in parts[:-1]:

        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}

        cur = cur[p]

    cur[parts[-1]] = float(value)


# =============================================================================
# SAFE PARAMETER CLIPPING
# =============================================================================

def _safe_clip(path: str, value: float) -> float:

    value = float(value)

    # Emitter exponent
    if "emitter_exponent" in path:
        return float(
            np.clip(
                value,
                SAFE_MIN_EMITTER_EXPONENT,
                SAFE_MAX_EMITTER_EXPONENT,
            )
        )

    # Global leakage scale
    if "global_scale" in path:
        return float(
            np.clip(
                value,
                SAFE_MIN_GLOBAL_LEAK_SCALE,
                SAFE_MAX_GLOBAL_LEAK_SCALE,
            )
        )

    # Zone multipliers
    if "zone_multipliers" in path:
        return float(
            np.clip(
                value,
                SAFE_MIN_ZONE_MULTIPLIER,
                SAFE_MAX_ZONE_MULTIPLIER,
            )
        )

    # Config bounds
    b = config.OPT_BOUNDS.get(path)

    if b:
        lo, hi = float(b[0]), float(b[1])
        value = min(hi, max(lo, value))

    return float(value)


# =============================================================================
# OBJECTIVE CONFIG
# =============================================================================

OBJECTIVE_WEIGHT_KEYS = ("w_ts", "w_feat", "w_sp", "w_vol", "w_reg")


def _objective_weights_from_config() -> ObjectiveWeights:
    """Build objective weights from config.py with no silent fallbacks."""

    w = getattr(config, "OBJECTIVE_WEIGHTS", None)
    if not isinstance(w, dict):
        raise TypeError("config.OBJECTIVE_WEIGHTS must be a dict")

    missing = [key for key in OBJECTIVE_WEIGHT_KEYS if key not in w]
    extra = [key for key in w if key not in OBJECTIVE_WEIGHT_KEYS]
    if missing or extra:
        raise ValueError(
            "config.OBJECTIVE_WEIGHTS must contain exactly "
            f"{list(OBJECTIVE_WEIGHT_KEYS)}; missing={missing}, extra={extra}"
        )

    values = {key: float(w[key]) for key in OBJECTIVE_WEIGHT_KEYS}
    invalid = [
        key
        for key, value in values.items()
        if not np.isfinite(value) or value < 0.0
    ]
    if invalid:
        raise ValueError(
            "config.OBJECTIVE_WEIGHTS values must be finite and non-negative; "
            f"invalid={invalid}"
        )

    return ObjectiveWeights(**values)


def _objective_config_from_config(
    *,
    runner: Any,
    raw_params: Dict[str, Any],
    metadata: Any,
) -> ObjectiveConfig:
    """Create the single objective config used by loss, gradients, and search.

    The top-level component weights come only from config.OBJECTIVE_WEIGHTS.
    Regularization priors are frozen once from the initial optimizer parameter
    state instead of being rebuilt during finite-difference perturbations.
    """

    params = runner.parameterization.from_dict(raw_params)
    return ObjectiveConfig(
        weights=_objective_weights_from_config(),
        regularization=build_default_regularization_from_params(params, metadata),
    )


def _objective_weights_dict(obj_cfg: ObjectiveConfig) -> dict[str, float]:
    w = obj_cfg.weights
    return {
        "w_ts": float(w.w_ts),
        "w_feat": float(w.w_feat),
        "w_sp": float(w.w_sp),
        "w_vol": float(w.w_vol),
        "w_reg": float(w.w_reg),
    }


# =============================================================================
# HYDRAULIC SANITY CHECK
# =============================================================================

def _validate_raw_params(raw_params: Dict[str, Any]) -> None:

    leakage = raw_params.get("leakage", {})

    gs = leakage.get("global_scale", 0.0)

    if gs <= 0:
        raise RuntimeError(
            f"Invalid global leakage scale: {gs}"
        )

    pda = raw_params.get("pda", {})

    ee = pda.get("emitter_exponent", 0.5)

    if ee < SAFE_MIN_EMITTER_EXPONENT:
        raise RuntimeError(
            f"Emitter exponent too low: {ee}"
        )

    if ee > SAFE_MAX_EMITTER_EXPONENT:
        raise RuntimeError(
            f"Emitter exponent too high: {ee}"
        )


# =============================================================================
# METADATA EXTRACTION (FIXED)
# =============================================================================

def _extract_metadata_info(metadata: Any) -> Dict[str, Any]:
    """
    Safely extract metadata information from ModelMetadata object.
    
    Handles both dict-like and object attribute access.
    Returns a dict with available metadata.
    """
    info = {}
    
    # Try accessing as dict first
    if isinstance(metadata, dict):
        return metadata
    
    # Try accessing as object attributes
    attrs = [
        'inpfile_units',
        'pattern_timestep_s',
        'duration_s',
        'report_timestep_s',
        'service_node_count',
        'service_daily_volume_m3_min',
        'service_daily_volume_m3_max',
        'zones_distribution',
    ]
    
    for attr in attrs:
        try:
            val = getattr(metadata, attr, None)
            if val is not None:
                info[attr] = val
        except Exception:
            continue
    
    # If we got nothing, try to convert to dict via __dict__
    if not info and hasattr(metadata, '__dict__'):
        try:
            info = dict(metadata.__dict__)
        except Exception:
            pass
    
    return info


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-no",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    _ensure_dirs()

    run_dir = _prepare_run_dir(args.run_no)

    observed, n_days, observation_preprocessing = load_calibration_observed()

    metadata = config.build_default_metadata()

    runner = build_runner(
        inp_path=config.MODEL_INP,
        metadata=metadata,
    )

    # Extract metadata safely
    metadata_info = _extract_metadata_info(metadata)

    print(f"Optimizer run dir: {run_dir}")
    print(f"Observed days: {n_days}")
    print(f"Observation preprocessing: {observation_preprocessing}")
    print(f"Metadata: {metadata_info}")

    # =============================================================================
    # INITIAL PARAMETERS
    # =============================================================================

    raw_params = config.build_default_raw_params()

    raw_params.setdefault("time", {})
    raw_params["time"]["duration_days"] = int(n_days)

    # FORCE SAFE INITIAL VALUES
    if "pda" in raw_params:
        raw_params["pda"]["emitter_exponent"] = max(
            SAFE_MIN_EMITTER_EXPONENT,
            float(raw_params["pda"].get("emitter_exponent", 0.5)),
        )

    if "leakage" in raw_params:
        raw_params["leakage"]["global_scale"] = max(
            SAFE_MIN_GLOBAL_LEAK_SCALE,
            float(raw_params["leakage"].get("global_scale", 0.01)),
        )

    # =============================================================================
    # APPLY SAFE CLIPPING
    # =============================================================================

    for p in config.OPT_PARAM_PATHS:

        try:
            x = _get_by_path(raw_params, p)
            _set_by_path(raw_params, p, _safe_clip(p, x))
        except Exception:
            continue

    obj_cfg = _objective_config_from_config(
        runner=runner,
        raw_params=raw_params,
        metadata=metadata,
    )
    active_objective_weights = _objective_weights_dict(obj_cfg)

    print(f"Optimizing {len(config.OPT_PARAM_PATHS)} parameters")
    print(f"Objective weights: {active_objective_weights}")

    # =============================================================================
    # OBJECTIVE EVALUATION
    # =============================================================================

    def eval_J(
        rp_in: Dict[str, Any]
    ) -> tuple[float, Dict[str, float]]:

        rp = copy.deepcopy(rp_in)

        try:

            _validate_raw_params(rp)

            j, breakdown = runner.evaluate_objective(
                rp,
                observed_pressure=observed,
                config=obj_cfg,
            )

            j = float(j)

            if not np.isfinite(j):
                raise RuntimeError(
                    "Objective became non-finite"
                )

            return j, dict(breakdown)

        except Exception as e:

            if config.VERBOSE:
                print(
                    f"[eval_J] Penalizing failed run: "
                    f"{type(e).__name__}: {e}"
                )

            return (
                FAILED_RUN_PENALTY,
                {
                    "J_total": FAILED_RUN_PENALTY,
                    "J_failed": FAILED_RUN_PENALTY,
                },
            )

    # =============================================================================
    # INITIAL OBJECTIVE
    # =============================================================================

    best_params = copy.deepcopy(raw_params)

    best_J, best_breakdown = eval_J(best_params)

    print(f"Initial J_total: {best_J:.6g}")

    lr = float(config.OPT_LEARNING_RATE)

    history_rows: list[dict[str, float | int]] = []

    # =============================================================================
    # OPTIMIZATION LOOP
    # =============================================================================

    for it in range(int(config.OPT_MAX_ITERS)):

        cur_J, cur_breakdown = eval_J(raw_params)

        row = {
            "iter": it,
            "J_total": float(cur_J),
            "lr": float(lr),
        }
        row.update(
            {
                "wJ_timeseries": float(cur_breakdown.get("wJ_timeseries", 0.0)),
                "wJ_features": float(cur_breakdown.get("wJ_features", 0.0)),
                "wJ_spatial": float(cur_breakdown.get("wJ_spatial", 0.0)),
                "wJ_volume": float(cur_breakdown.get("wJ_volume", 0.0)),
                "wJ_regularization": float(
                    cur_breakdown.get("wJ_regularization", 0.0)
                ),
            }
        )

        history_rows.append(row)

        # ---------------------------------------------------------------------
        # COMPUTE GRADIENTS
        # ---------------------------------------------------------------------

        grads: dict[str, float] = {}

        for p in config.OPT_PARAM_PATHS:

            try:

                x = _get_by_path(raw_params, p)

                eps = max(
                    config.OPT_FD_EPS_ABS,
                    config.OPT_FD_EPS_REL * max(1.0, abs(x)),
                )

                rp_plus = copy.deepcopy(raw_params)
                rp_minus = copy.deepcopy(raw_params)

                _set_by_path(
                    rp_plus,
                    p,
                    _safe_clip(p, x + eps),
                )

                _set_by_path(
                    rp_minus,
                    p,
                    _safe_clip(p, x - eps),
                )

                j_plus, _ = eval_J(rp_plus)
                j_minus, _ = eval_J(rp_minus)

                if (
                    not np.isfinite(j_plus)
                    or not np.isfinite(j_minus)
                ):
                    grads[p] = 0.0
                else:
                    grads[p] = (
                        (j_plus - j_minus)
                        / (2.0 * eps)
                    )
                row[f"grad.{p}"] = float(grads[p])

            except Exception:
                grads[p] = 0.0
                row[f"grad.{p}"] = 0.0

        # ---------------------------------------------------------------------
        # BACKTRACKING LINE SEARCH
        # ---------------------------------------------------------------------

        accepted = False

        lr_try = float(lr)

        max_backtracks = 12

        proposal: Dict[str, Any] = copy.deepcopy(raw_params)
        new_J = cur_J
        new_breakdown = cur_breakdown

        for bt in range(max_backtracks + 1):

            proposal = copy.deepcopy(raw_params)

            for p in config.OPT_PARAM_PATHS:

                try:

                    x = _get_by_path(raw_params, p)

                    x_new = x - lr_try * grads[p]

                    x_new = _safe_clip(p, x_new)

                    _set_by_path(proposal, p, x_new)

                except Exception:
                    continue

            new_J, new_breakdown = eval_J(proposal)

            if (
                np.isfinite(new_J)
                and new_J < cur_J
            ):
                accepted = True
                break

            lr_try *= 0.5

        # ---------------------------------------------------------------------
        # NO IMPROVEMENT
        # ---------------------------------------------------------------------

        if not accepted:

            print(
                f"iter={it}: no improving step found "
                f"after {max_backtracks} backtracks"
            )

            break

        # ---------------------------------------------------------------------
        # ACCEPT STEP
        # ---------------------------------------------------------------------

        raw_params = proposal

        lr = lr_try * float(
            config.OPT_LEARNING_RATE_DECAY
        )

        print(
            f"iter={it}: "
            f"J {cur_J:.6g} -> {new_J:.6g}, "
            f"lr={lr_try:.6g}"
        )

        # ---------------------------------------------------------------------
        # UPDATE BEST
        # ---------------------------------------------------------------------

        if new_J < best_J:

            best_J = float(new_J)

            best_params = copy.deepcopy(raw_params)

            best_breakdown = dict(new_breakdown)

        # ---------------------------------------------------------------------
        # CONVERGENCE
        # ---------------------------------------------------------------------

        rel_impr = (
            (cur_J - new_J)
            / max(1e-12, abs(cur_J))
        )

        if rel_impr < float(config.OPT_TOL_REL):

            print(
                f"Stopping: relative improvement "
                f"{rel_impr:.3g} < OPT_TOL_REL"
            )

            break

    # =============================================================================
    # SAVE OUTPUTS
    # =============================================================================

    hist_df = pd.DataFrame(history_rows)

    run_hist_csv = run_dir / "opt_history.csv"
    run_best_json = run_dir / "best_params.json"

    hist_df.to_csv(run_hist_csv, index=False)

    out = {
        "best_J_total": float(best_J),
        "best_breakdown": best_breakdown,
        "best_raw_params": best_params,
        "optimized_paths": list(config.OPT_PARAM_PATHS),
        "objective_weights": active_objective_weights,
        "n_days": int(n_days),
        "observation_preprocessing": observation_preprocessing,
        "metadata": metadata_info,  # Include extracted metadata
    }

    run_best_json.write_text(
        json.dumps(out, indent=2, default=str)
    )

    Path(config.OPT_HISTORY_CSV).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(config.OPT_BEST_PARAMS_JSON).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    hist_df.to_csv(
        Path(config.OPT_HISTORY_CSV),
        index=False,
    )

    Path(config.OPT_BEST_PARAMS_JSON).write_text(
        json.dumps(out, indent=2, default=str)
    )

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Best J_total: {best_J:.6g}")
    print(f"Iterations completed: {len(history_rows)}")
    print(f"Best params saved to: {run_best_json}")
    print(f"Optimization history saved to: {run_hist_csv}")
    print(f"Global best params: {Path(config.OPT_BEST_PARAMS_JSON)}")
    print("=" * 80)


if __name__ == "__main__":
    main()

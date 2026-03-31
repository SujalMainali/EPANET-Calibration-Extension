"""Gradient-descent optimizer for the calibration objective.

Reads all settings from config.py:
- observed pressure CSV(s)
- list of parameter paths to optimize
- learning rate, finite-difference steps, bounds

Supports multi-day observations:
- If config.OBSERVED_PRESSURE_CSVS is set: each CSV is treated as one day.
  The times are offset by +k*86400 so the concatenated observations span N days.
- Otherwise config.OBSERVED_PRESSURE_CSV can be a single-day or multi-day CSV.

Implementation notes
- Uses finite-difference gradients (central differences).
- Clips parameters to optional bounds.
- Keeps the best-seen parameters and writes:
  - outputs/reports/opt_history.csv
  - outputs/reports/best_params.json
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

import config
from calibration.objective import load_observed_pressure_csv
from calibration.runner import build_runner


def _ensure_dirs() -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    config.DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)


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
    raise FileNotFoundError(
        f"Observed CSV not found: {path!r}. Tried: {str(p.resolve())!r} and {str(alt.resolve())!r}"
    )


def _load_observed_one(path: str) -> pd.DataFrame:
    path = _resolve_existing_path(path)
    if config.OBSERVED_TIME_COLUMN is None:
        df = load_observed_pressure_csv(path)
    else:
        raw = pd.read_csv(path)
        if config.OBSERVED_TIME_COLUMN not in raw.columns:
            raise ValueError(
                f"OBSERVED_TIME_COLUMN={config.OBSERVED_TIME_COLUMN!r} not found in {path}"
            )
        df = raw.set_index(config.OBSERVED_TIME_COLUMN)
        try:
            df.index = pd.Index(pd.to_numeric(df.index.to_numpy()))
        except Exception:
            df.index = pd.to_datetime(df.index)

    return _coerce_index_to_seconds(df)


def load_observed_multi_day() -> tuple[pd.DataFrame, int]:
    """Return (observed_df, n_days)."""

    if config.OBSERVED_PRESSURE_CSVS:
        dfs = []
        for i, p in enumerate(config.OBSERVED_PRESSURE_CSVS):
            d = _load_observed_one(p)
            # normalize each day to start at 0 and then offset by i*86400
            d = d.copy()
            d.index = pd.Index(pd.to_numeric(d.index.to_numpy()) - float(d.index.min()) + float(i * 86400))
            dfs.append(d)
        out = pd.concat(dfs, axis=0).sort_index()
        return out, len(dfs)

    if not config.OBSERVED_PRESSURE_CSV:
        raise ValueError(
            "Set OBSERVED_PRESSURE_CSV or OBSERVED_PRESSURE_CSVS in config.py for optimization."
        )

    df = _load_observed_one(config.OBSERVED_PRESSURE_CSV)
    span = float(df.index.max() - df.index.min()) if len(df.index) else 0.0
    n_days = int(max(1, int(np.ceil((span + 1.0) / 86400.0))))
    return df, n_days


def _get_by_path(d: Dict[str, Any], path: str) -> float:
    parts = path.split(".")
    cur: Any = d
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            raise KeyError(f"Path not found in raw_params: {path}")
    return float(cur)


def _set_by_path(d: Dict[str, Any], path: str, value: float) -> None:
    parts = path.split(".")
    cur: Any = d
    for p in parts[:-1]:
        if not isinstance(cur, dict):
            raise KeyError(f"Cannot set path (non-dict encountered): {path}")
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    if not isinstance(cur, dict):
        raise KeyError(f"Cannot set path (non-dict encountered): {path}")
    cur[parts[-1]] = float(value)


def _apply_bounds(path: str, x: float) -> float:
    b = config.OPT_BOUNDS.get(path)
    if not b:
        return float(x)
    lo, hi = float(b[0]), float(b[1])
    return float(min(hi, max(lo, x)))


def _finite_difference_eps(x: float) -> float:
    return float(max(config.OPT_FD_EPS_ABS, config.OPT_FD_EPS_REL * max(1.0, abs(float(x)))))


def _finite_difference_grad(
    eval_J,
    raw_params: Dict[str, Any],
    path: str,
    x: float,
    eps: float,
) -> float:
    """Bounds-aware finite-difference gradient for one scalar parameter.

    Uses the *actual* (possibly clipped) step size in the denominator.
    This matters a lot near bounds (e.g., leakage.global_scale at 0).
    """

    x_plus = _apply_bounds(path, x + eps)
    x_minus = _apply_bounds(path, x - eps)

    if np.isclose(x_plus, x_minus):
        return 0.0

    rp_plus = copy.deepcopy(raw_params)
    rp_minus = copy.deepcopy(raw_params)
    _set_by_path(rp_plus, path, x_plus)
    _set_by_path(rp_minus, path, x_minus)

    j_plus, _ = eval_J(rp_plus)
    j_minus, _ = eval_J(rp_minus)
    return float((j_plus - j_minus) / (x_plus - x_minus))


def main() -> None:
    _ensure_dirs()

    observed, n_days = load_observed_multi_day()

    metadata = config.build_default_metadata()
    runner = build_runner(inp_path=config.MODEL_INP, metadata=metadata)

    def eval_J(rp_in: Dict[str, Any]) -> tuple[float, Dict[str, float]]:
        rp = copy.deepcopy(rp_in)
        rp.setdefault("time", {})
        rp["time"]["duration_days"] = int(max(1, n_days))
        try:
            j, breakdown = runner.evaluate_objective(rp, observed_pressure=observed)
            j = float(j)
            if not np.isfinite(j):
                raise RuntimeError(f"Non-finite objective returned: {j!r}")
            return j, dict(breakdown)
        except Exception as e:
            # Penalize invalid runs heavily so gradient descent backs away.
            # Keep this finite so comparisons behave predictably.
            penalty = 1e9
            if config.VERBOSE:
                print(f"[eval_J] Penalizing failed run: {type(e).__name__}: {e}")
            return float(penalty), {"J_total": float(penalty), "J_failed": float(penalty)}

    # Start from config defaults
    raw_params: Dict[str, Any] = config.build_default_raw_params()

    # Clip init to bounds
    for p in config.OPT_PARAM_PATHS:
        x0 = _get_by_path(raw_params, p)
        _set_by_path(raw_params, p, _apply_bounds(p, x0))

    lr = float(config.OPT_LEARNING_RATE)

    best_params = copy.deepcopy(raw_params)
    best_J, best_breakdown = eval_J(best_params)

    history_rows = []

    if config.VERBOSE:
        print(f"Observed days: {n_days}, rows: {len(observed)}")
        print("Initial J_total:", best_J)

    for it in range(int(config.OPT_MAX_ITERS)):
        cur_J, cur_breakdown = eval_J(raw_params)

        row = {"iter": it, "J_total": float(cur_J), "lr": float(lr)}
        for k in ("J_timeseries", "J_features", "J_spatial", "J_volume", "J_regularization"):
            if k in cur_breakdown:
                row[k] = float(cur_breakdown[k])
        for p in config.OPT_PARAM_PATHS:
            row[p] = float(_get_by_path(raw_params, p))
        history_rows.append(row)

        # Track best
        if cur_J < best_J:
            best_J = float(cur_J)
            best_params = copy.deepcopy(raw_params)
            best_breakdown = dict(cur_breakdown)

        # Compute gradient by central finite differences
        grads: Dict[str, float] = {}
        for p in config.OPT_PARAM_PATHS:
            x = _get_by_path(raw_params, p)
            eps = _finite_difference_eps(x)

            grads[p] = _finite_difference_grad(eval_J, raw_params, p, x, eps)

        # Backtracking line search: shrink lr until we find an improving step.
        max_backtracks = 12
        lr_try = float(lr)
        accepted = False
        new_J = float("inf")
        new_breakdown: Dict[str, float] = {}

        for bt in range(max_backtracks + 1):
            proposal = copy.deepcopy(raw_params)
            for p in config.OPT_PARAM_PATHS:
                x = _get_by_path(raw_params, p)
                x_new = x - lr_try * grads[p]
                _set_by_path(proposal, p, _apply_bounds(p, x_new))

            new_J, new_breakdown = eval_J(proposal)
            if new_J <= cur_J:
                accepted = True
                break
            lr_try *= 0.5

        if not accepted:
            if config.VERBOSE:
                print(
                    f"iter={it}: no improving step found after {max_backtracks} backtracks; stopping (lr={lr_try:.6g})"
                )
            break

        if config.VERBOSE:
            suffix = f" (backtracks={bt})" if bt else ""
            print(f"iter={it}: J {cur_J:.6g} -> {new_J:.6g} (improved), lr={lr_try:.6g}{suffix}")

        raw_params = proposal

        # Update best immediately on acceptance (important when OPT_MAX_ITERS is small).
        if new_J < best_J:
            best_J = float(new_J)
            best_params = copy.deepcopy(raw_params)
            best_breakdown = dict(new_breakdown)

        # Update learning rate after a successful step.
        lr = lr_try * float(config.OPT_LEARNING_RATE_DECAY)

        rel_impr = (cur_J - new_J) / max(1e-12, abs(cur_J))
        if rel_impr < float(config.OPT_TOL_REL):
            if config.VERBOSE:
                print(f"Stopping: relative improvement {rel_impr:.3g} < OPT_TOL_REL")
            break

    # Write outputs
    hist_df = pd.DataFrame(history_rows)
    hist_df.to_csv(Path(config.OPT_HISTORY_CSV), index=False)

    out = {
        "best_J_total": float(best_J),
        "best_breakdown": best_breakdown,
        "best_raw_params": best_params,
        "optimized_paths": list(config.OPT_PARAM_PATHS),
        "n_days": int(n_days),
    }
    Path(config.OPT_BEST_PARAMS_JSON).write_text(json.dumps(out, indent=2, default=str))

    if config.VERBOSE:
        print("Best J_total:", float(best_J))
        print(f"Wrote: {config.OPT_HISTORY_CSV}")
        print(f"Wrote: {config.OPT_BEST_PARAMS_JSON}")


if __name__ == "__main__":
    main()

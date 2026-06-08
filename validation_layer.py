"""Validation and smoothing for observed calibration pressure signals.

Multi-day observations are folded into one representative daily profile per
sensor. A Fourier low-pass filter can then remove short-period noise before the
daily profile is repeated across the calibration horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def load_processed_observation_dataset(
    path: str | Path,
    sensor_nodes: list[str],
) -> pd.DataFrame:
    """Load a saved calibration-ready observed pressure dataset."""

    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"Processed observation dataset must have time + sensors: {path}")

    time_col = df.columns[0]
    df = df.set_index(time_col)
    df.index = pd.Index(pd.to_numeric(df.index.to_numpy()), name=str(time_col))

    missing = [sensor for sensor in sensor_nodes if sensor not in df.columns]
    if missing:
        raise ValueError(
            f"Processed observation dataset {path} is missing sensor columns: {missing}"
        )

    out = df[sensor_nodes].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(out.to_numpy(dtype=float)).all():
        raise ValueError(f"Processed observation dataset contains non-finite values: {path}")
    return out


def save_processed_observation_dataset(
    observed: pd.DataFrame,
    path: str | Path,
) -> None:
    """Save calibration-ready observed pressure data with a reusable time index."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    observed.to_csv(out_path, index=True, index_label=observed.index.name or "time_seconds")


@dataclass(frozen=True)
class ValidationResult:
    calibration_data: pd.DataFrame
    folded_profile: pd.DataFrame
    smoothed_profile: pd.DataFrame
    frequency_magnitudes: pd.DataFrame
    summary: dict[str, Any]


class PreprocessingValidationLayer:
    """Fold, validate, and smooth observed pressure data."""

    def __init__(
        self,
        csv_filepaths: str | list[str] | None = None,
        sensor_nodes: list[str] | None = None,
        *,
        points_per_day: int = 24,
        fold_days_enabled: bool = True,
        fold_aggregation: str = "mean",
        smoothing_enabled: bool = True,
        smoothing_max_harmonic: int = 6,
        interpolate_missing: bool = True,
        require_complete_days: bool = True,
        mass_relative_tolerance: float = 1e-10,
        parseval_relative_tolerance: float = 1e-10,
        export_stages: bool = True,
        output_dir: str | Path = Path("outputs") / "debug" / "validation",
        verbose: bool = True,
    ):
        if points_per_day < 2:
            raise ValueError("points_per_day must be at least 2")
        if fold_aggregation not in {"mean", "median"}:
            raise ValueError("fold_aggregation must be 'mean' or 'median'")
        if not 0 <= smoothing_max_harmonic <= points_per_day // 2:
            raise ValueError(
                "smoothing_max_harmonic must be between 0 and the Nyquist harmonic "
                f"({points_per_day // 2})"
            )

        self.sensor_nodes = list(sensor_nodes or [])
        self._observed: pd.DataFrame | None = None
        if csv_filepaths is not None:
            paths = (
                [csv_filepaths]
                if isinstance(csv_filepaths, str)
                else list(csv_filepaths)
            )
            self._observed = pd.concat(
                [pd.read_csv(path) for path in paths],
                ignore_index=True,
            )
        self.points_per_day = int(points_per_day)
        self.fold_days_enabled = bool(fold_days_enabled)
        self.fold_aggregation = fold_aggregation
        self.smoothing_enabled = bool(smoothing_enabled)
        self.smoothing_max_harmonic = int(smoothing_max_harmonic)
        self.interpolate_missing = bool(interpolate_missing)
        self.require_complete_days = bool(require_complete_days)
        self.mass_relative_tolerance = float(mass_relative_tolerance)
        self.parseval_relative_tolerance = float(parseval_relative_tolerance)
        self.export_stages = bool(export_stages)
        self.output_dir = Path(output_dir)
        self.verbose = bool(verbose)

    def process_and_validate(
        self,
        observed: pd.DataFrame | None = None,
    ) -> ValidationResult:
        """Return calibration-ready, per-sensor pressure observations."""

        if observed is None:
            observed = self._observed
        if observed is None:
            raise ValueError(
                "Provide observed data to process_and_validate() or CSV paths "
                "when constructing the validation layer"
            )
        if not self.sensor_nodes:
            raise ValueError("sensor_nodes cannot be empty")

        missing = [sensor for sensor in self.sensor_nodes if sensor not in observed.columns]
        if missing:
            raise ValueError(f"Missing sensor columns in observed data: {missing}")

        pressure = observed[self.sensor_nodes].apply(pd.to_numeric, errors="coerce")
        if self.interpolate_missing:
            pressure = pressure.interpolate(method="linear", limit_direction="both")

        non_finite = ~np.isfinite(pressure.to_numpy(dtype=float))
        if non_finite.any():
            bad_columns = [
                sensor
                for sensor, has_bad_value in zip(
                    self.sensor_nodes,
                    non_finite.any(axis=0),
                    strict=True,
                )
                if bool(has_bad_value)
            ]
            raise ValueError(f"Non-finite pressure values remain in sensors: {bad_columns}")

        total_points = len(pressure)
        num_days, remainder = divmod(total_points, self.points_per_day)
        if num_days < 1:
            raise ValueError(
                f"Insufficient observations: need at least {self.points_per_day} points"
            )
        if remainder and self.require_complete_days:
            raise ValueError(
                f"Observed data has {total_points} points, which is not a whole number "
                f"of {self.points_per_day}-point days"
            )

        used_points = num_days * self.points_per_day
        pressure = pressure.iloc[:used_points].copy()
        values = pressure.to_numpy(dtype=float).reshape(
            num_days,
            self.points_per_day,
            len(self.sensor_nodes),
        )

        if self.fold_days_enabled:
            if self.fold_aggregation == "mean":
                profile_values = values.mean(axis=0)
            else:
                profile_values = np.median(values, axis=0)

            profile_index = pd.RangeIndex(self.points_per_day, name="hour")
            folded = pd.DataFrame(
                profile_values,
                index=profile_index,
                columns=self.sensor_nodes,
            )
            mass_delta, mass_tolerance = self._validate_folding(
                values,
                profile_values,
            )
            frequency_magnitudes = self._validate_frequency_energy(
                profile_values,
                profile="folded",
            )
            smoothed_values = self._smooth_profile(profile_values)
            smoothed = pd.DataFrame(
                smoothed_values,
                index=profile_index.copy(),
                columns=self.sensor_nodes,
            )
            calibration_values = np.tile(smoothed_values, (num_days, 1))
            roughness_input = profile_values
            roughness_output = smoothed_values
        else:
            folded = pressure.copy()
            smoothed_days: list[FloatArray] = []
            frequency_frames: list[pd.DataFrame] = []
            for day in range(num_days):
                day_values = values[day]
                smoothed_days.append(self._smooth_profile(day_values))
                frequency_frames.append(
                    self._validate_frequency_energy(
                        day_values,
                        profile=f"day_{day + 1}",
                    )
                )

            smoothed_values = np.stack(smoothed_days, axis=0)
            calibration_values = smoothed_values.reshape(
                used_points,
                len(self.sensor_nodes),
            )
            smoothed = pd.DataFrame(
                calibration_values,
                index=pressure.index.copy(),
                columns=self.sensor_nodes,
            )
            frequency_magnitudes = pd.concat(
                frequency_frames,
                ignore_index=True,
            )
            mass_delta, mass_tolerance = 0.0, 0.0
            roughness_input = values
            roughness_output = smoothed_values

        calibration_data = pd.DataFrame(
            calibration_values,
            index=pressure.index.copy(),
            columns=self.sensor_nodes,
        )
        calibration_data.index.name = observed.index.name or "time_seconds"

        raw_roughness = self._roughness(roughness_input)
        smoothed_roughness = self._roughness(roughness_output)
        summary: dict[str, Any] = {
            "num_days": int(num_days),
            "points_per_day": int(self.points_per_day),
            "fold_days_enabled": self.fold_days_enabled,
            "fold_aggregation": self.fold_aggregation,
            "smoothing_enabled": self.smoothing_enabled,
            "smoothing_max_harmonic": int(self.smoothing_max_harmonic),
            "mass_delta": float(mass_delta),
            "mass_tolerance": float(mass_tolerance),
            "raw_roughness": float(raw_roughness),
            "smoothed_roughness": float(smoothed_roughness),
            "roughness_reduction_fraction": float(
                0.0
                if raw_roughness <= 0.0
                else 1.0 - (smoothed_roughness / raw_roughness)
            ),
        }

        if self.export_stages:
            self._export(
                folded=folded,
                smoothed=smoothed,
                frequencies=frequency_magnitudes,
                calibration_data=calibration_data,
            )

        if self.verbose:
            smoothing_state = (
                f"harmonics 0-{self.smoothing_max_harmonic}"
                if self.smoothing_enabled
                else "disabled"
            )
            print(
                "[validation] "
                f"days={num_days}; folding={'enabled' if self.fold_days_enabled else 'disabled'}; "
                f"smoothing={smoothing_state}; "
                f"roughness reduction={summary['roughness_reduction_fraction']:.1%}"
            )

        return ValidationResult(
            calibration_data=calibration_data,
            folded_profile=folded,
            smoothed_profile=smoothed,
            frequency_magnitudes=frequency_magnitudes,
            summary=summary,
        )

    def _validate_folding(
        self,
        daily_values: FloatArray,
        folded_values: FloatArray,
    ) -> tuple[float, float]:
        if self.fold_aggregation != "mean":
            return 0.0, 0.0

        raw_sum = float(np.sum(daily_values))
        reconstructed_sum = float(np.sum(folded_values) * daily_values.shape[0])
        delta = abs(raw_sum - reconstructed_sum)
        tolerance = max(
            np.finfo(float).eps * max(1.0, abs(raw_sum)),
            self.mass_relative_tolerance * max(1.0, abs(raw_sum)),
        )
        if delta > tolerance:
            raise AssertionError(
                f"Mean-fold mass conservation failed: delta={delta:.6g}, "
                f"tolerance={tolerance:.6g}"
            )
        return delta, tolerance

    def _validate_frequency_energy(
        self,
        profile_values: FloatArray,
        *,
        profile: str,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        n = self.points_per_day

        for sensor_index, sensor in enumerate(self.sensor_nodes):
            centered = profile_values[:, sensor_index] - np.mean(
                profile_values[:, sensor_index]
            )
            coefficients = np.fft.rfft(centered)
            time_energy = float(np.mean(centered**2))

            spectral_terms = (np.abs(coefficients) ** 2) / (n**2)
            weights = np.ones_like(spectral_terms)
            if n % 2 == 0:
                weights[1:-1] = 2.0
            else:
                weights[1:] = 2.0
            spectral_energy = float(np.sum(weights * spectral_terms))

            delta = abs(time_energy - spectral_energy)
            tolerance = max(
                np.finfo(float).eps,
                self.parseval_relative_tolerance * max(1.0, time_energy),
            )
            if delta > tolerance:
                raise AssertionError(
                    f"Parseval energy check failed for {sensor}: "
                    f"delta={delta:.6g}, tolerance={tolerance:.6g}"
                )

            amplitudes = np.abs(coefficients) / n
            if len(amplitudes) > 1:
                if n % 2 == 0:
                    amplitudes[1:-1] *= 2.0
                else:
                    amplitudes[1:] *= 2.0

            for harmonic, amplitude in enumerate(amplitudes):
                rows.append(
                    {
                        "profile": profile,
                        "sensor": sensor,
                        "harmonic": int(harmonic),
                        "cycles_per_day": int(harmonic),
                        "magnitude": float(amplitude),
                        "retained": bool(
                            not self.smoothing_enabled
                            or harmonic <= self.smoothing_max_harmonic
                        ),
                    }
                )

        return pd.DataFrame(rows)

    def _smooth_profile(self, profile_values: FloatArray) -> FloatArray:
        if not self.smoothing_enabled:
            return profile_values.copy()

        coefficients = np.fft.rfft(profile_values, axis=0)
        coefficients[self.smoothing_max_harmonic + 1 :, :] = 0.0
        smoothed = np.fft.irfft(
            coefficients,
            n=self.points_per_day,
            axis=0,
        )

        # The DC component preserves the mean. Correct only for roundoff so the
        # calibration target never receives a shifted pressure level.
        smoothed += np.mean(profile_values, axis=0) - np.mean(smoothed, axis=0)
        if not np.isfinite(smoothed).all():
            raise AssertionError("Smoothing produced non-finite pressure values")
        return smoothed

    @staticmethod
    def _roughness(values: FloatArray) -> float:
        profiles = values[np.newaxis, ...] if values.ndim == 2 else values
        cyclic = np.concatenate(
            [profiles[:, -1:, :], profiles, profiles[:, :1, :]],
            axis=1,
        )
        second_difference = np.diff(cyclic, n=2, axis=1)
        return float(np.mean(second_difference**2))

    def _export(
        self,
        *,
        folded: pd.DataFrame,
        smoothed: pd.DataFrame,
        frequencies: pd.DataFrame,
        calibration_data: pd.DataFrame,
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        folded.to_csv(self.output_dir / "observed_folded_raw.csv", index=True)
        smoothed.to_csv(self.output_dir / "observed_folded_smoothed.csv", index=True)
        frequencies.to_csv(
            self.output_dir / "observed_fourier_frequencies.csv",
            index=False,
        )
        calibration_data.to_csv(
            self.output_dir / "observed_calibration_target.csv",
            index=True,
        )

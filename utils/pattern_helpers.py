"""Pure math helpers for building demand patterns."""

from __future__ import annotations

import numpy as np


def normalize24(values: np.ndarray) -> np.ndarray:
    """Normalize a length-24 vector so its sum is 24."""
    s = float(values.sum())
    if s <= 0:
        raise ValueError("Pattern sum must be positive.")
    return 24.0 * values / s


def gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Gaussian bell curve values for array x."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

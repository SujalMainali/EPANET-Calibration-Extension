"""Time grid helpers for expected report indices and diagnostics."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def expected_report_times(duration_s: int, report_step_s: int, report_start_s: int = 0) -> List[int]:
    if report_step_s <= 0:
        return [0]
    if duration_s < report_start_s:
        return []

    times = [0]
    t = report_start_s + report_step_s
    while t <= duration_s:
        times.append(int(t))
        t += report_step_s

    return sorted(set(times))


def summarize_time_index(times_s: List[int]) -> Dict[str, object]:
    if not times_s:
        return {
            "num_timesteps": 0,
            "first_times": [],
            "last_times": [],
            "step_diffs": [],
        }

    diffs = list(np.diff(times_s)) if len(times_s) > 1 else []
    return {
        "num_timesteps": len(times_s),
        "first_times": times_s[:5],
        "last_times": times_s[-5:],
        "step_diffs": diffs[:10],
    }

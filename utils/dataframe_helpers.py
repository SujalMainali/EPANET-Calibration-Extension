"""Pandas helpers for post-processing and aligning timeseries outputs."""

from __future__ import annotations

import pandas as pd


def expand_to_report_times(df: pd.DataFrame, expected_times: list[int]) -> pd.DataFrame:
    """Reindex to expected_times and fill missing values.

    Intended to align irregular toolkit time indices to a fixed reporting grid.
    """
    if df.empty:
        return pd.DataFrame(index=expected_times, columns=df.columns)

    out = df.copy()
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    out = out.reindex(expected_times).ffill().bfill()
    out.index.name = "time_s"
    return out

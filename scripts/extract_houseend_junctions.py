"""Export HOUSEEND junction base demands from an EPANET INP.

The output CSV is useful for:
- checking units
- checking base demand magnitudes
- building service_nodes metadata

This script intentionally uses local/module-level variables (not argparse) so it’s easy
to edit paths and rerun.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibration.builders import extract_houseend_junctions_from_inp, inspect_inp_units


def main() -> None:
    # ------------------------------
    # Edit these local variables
    # ------------------------------

    inp_path: str | None = None
    out_csv: str = "outputs/houseend_demands.csv"
    prefix: str = "HOUSEEND_"

    if inp_path is None:
        raise ValueError("Set inp_path inside main() to your EPANET INP path.")

    info = inspect_inp_units(inp_path)
    df = extract_houseend_junctions_from_inp(inp_path, houseend_prefix=prefix)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("Wrote:", out_path)
    print("INP info:", info)
    print("Rows:", len(df))
    if len(df) > 0:
        print(
            "daily_volume_m3 min/max:",
            float(df["daily_volume_m3"].min()),
            float(df["daily_volume_m3"].max()),
        )


if __name__ == "__main__":
    main()

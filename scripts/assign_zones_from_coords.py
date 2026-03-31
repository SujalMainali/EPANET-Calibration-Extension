"""Assign zones to junctions from an EPANET INP.

This script intentionally uses local/module-level variables (not argparse) so it’s easy
to edit paths and rerun.

Supported modes:
- grid:  split into 4 zones by median X/Y (Z_NW, Z_NE, Z_SW, Z_SE)

Notes:
- Requires a [COORDINATES] section in the INP.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from calibration.builders import build_zone_mapping_from_inp


def main() -> None:
    # ------------------------------
    # Edit these local variables
    # ------------------------------

    inp_path: str | None = None
    mode: str = "grid"  # currently only "grid" supported
    node_prefix: str | None = "HOUSEEND_"  # set to None to zone all nodes with coordinates
    out_csv: str = "outputs/node_zones.csv"

    if inp_path is None:
        raise ValueError("Set inp_path inside main() to your EPANET INP path.")

    zone_by_node = build_zone_mapping_from_inp(
        inp_path,
        mode=mode,
        node_prefix=node_prefix,
    )

    out = pd.DataFrame(
        {
            "node_name": list(zone_by_node.keys()),
            "zone": list(zone_by_node.values()),
        }
    )

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print("Wrote:", out_path)
    print("Rows:", len(out))


if __name__ == "__main__":
    main()

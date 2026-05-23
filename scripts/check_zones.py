import json
import re
from pathlib import Path
from collections import Counter

path = Path("outputs/reports/best_params.json")

data = json.loads(path.read_text())

mapping = {}

# =========================================================
# metadata.leak_nodes  ← YOUR ACTUAL STRUCTURE
# =========================================================
metadata = data.get("metadata", {})

leak_nodes = metadata.get("leak_nodes", {})

print(f"Found metadata.leak_nodes: {len(leak_nodes)} entries")

if leak_nodes:

    parsed = {}

    for node_id, meta in leak_nodes.items():

        zone = None

        # Case 1 — dict
        if isinstance(meta, dict):

            zone = meta.get("zone")

        # Case 2 — object repr string
        else:

            m = re.search(r"zone='([^']+)'", str(meta))

            if m:
                zone = m.group(1)

        if zone:
            parsed[str(node_id)] = str(zone)

    mapping = parsed

# =========================================================
# FINAL
# =========================================================
if not mapping:

    print()
    print("ERROR: Could not extract zones")
    raise SystemExit(1)

counts = Counter(mapping.values())

total = sum(counts.values())

print()
print("=" * 60)
print(f"TOTAL NODES = {total}")
print("=" * 60)

header = f"{'Zone':<10}{'Nodes':>10}{'Share':>12}{'Expected/50':>18}"

print(header)
print("-" * len(header))

for zone in sorted(counts.keys()):

    n = counts[zone]

    share = n / total

    expected = share * 50

    print(
        f"{zone:<10}{n:>10}{share:>11.1%}{expected:>18.1f}"
    )

print()
print("SUCCESS")
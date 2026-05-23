"""Drill into metadata.leak_nodes to see exact structure."""
import json
from pathlib import Path

raw = json.loads(Path("outputs/reports/best_params.json").read_text())
leak_nodes = raw["metadata"]["leak_nodes"]

print(f"Type: {type(leak_nodes).__name__}")
print(f"Length: {len(leak_nodes)}")
print()

if isinstance(leak_nodes, dict):
    keys = list(leak_nodes.keys())
    print(f"First 10 keys: {keys[:10]}")
    print()
    print("First 5 entries (full value):")
    for k in keys[:5]:
        print(f"  {k!r}: {leak_nodes[k]}")

elif isinstance(leak_nodes, list):
    print(f"First 5 entries:")
    for item in leak_nodes[:5]:
        print(f"  {item}")
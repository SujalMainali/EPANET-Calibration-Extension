"""
Quick diagnostic — run this BEFORE generate_leak_dataset_many.py.
It prints the exact structure of your best_params.json so we can
see why zone mapping is failing.

Usage:
    python scripts/diagnose_zone_mapping.py
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

BEST_PARAMS_PATH = "outputs/reports/best_params.json"  # adjust if needed

def main():
    p = Path(BEST_PARAMS_PATH)
    if not p.exists():
        print(f"❌ File not found: {p.resolve()}")
        sys.exit(1)

    raw = json.loads(p.read_text())

    print("=" * 80)
    print(f"FILE: {p.resolve()}  ({p.stat().st_size} bytes)")
    print("=" * 80)

    # ── Top-level keys ────────────────────────────────────────────────────────
    print(f"\n[1] TOP-LEVEL KEYS ({len(raw)}):")
    for k, v in raw.items():
        vtype = type(v).__name__
        vlen  = f"  len={len(v)}" if hasattr(v, "__len__") else ""
        print(f"    '{k}': {vtype}{vlen}")

    # ── Unwrap best_raw_params if present ─────────────────────────────────────
    inner = raw.get("best_raw_params", raw)
    if "best_raw_params" in raw:
        print("\n[2] UNWRAPPED best_raw_params keys:")
        for k, v in inner.items():
            vtype = type(v).__name__
            vlen  = f"  len={len(v)}" if hasattr(v, "__len__") else ""
            print(f"    '{k}': {vtype}{vlen}")

    # ── leakage block ─────────────────────────────────────────────────────────
    leakage = inner.get("leakage", {})
    print(f"\n[3] leakage keys: {list(leakage.keys())}")
    print(f"    global_scale     = {leakage.get('global_scale')}")
    print(f"    zone_multipliers = {leakage.get('zone_multipliers')}")

    leak_nodes = leakage.get("leak_nodes", None)
    if leak_nodes is None:
        leak_nodes = raw.get("leak_nodes", None)
    if leak_nodes is None:
        leak_nodes = inner.get("leak_nodes", None)

    print(f"\n[4] leak_nodes location: ", end="")
    if leak_nodes is None:
        print("NOT FOUND in leakage, best_raw_params, or top-level!")
    else:
        top_keys = list(leak_nodes.keys())
        print(f"found  ({len(top_keys)} top-level keys)")
        print(f"    First 10 keys: {top_keys[:10]}")
        # Show sample values
        for k in top_keys[:3]:
            v = leak_nodes[k]
            print(f"    [{k!r}] → {type(v).__name__}: {str(v)[:200]}")

    # ── metadata block ────────────────────────────────────────────────────────
    metadata = raw.get("metadata", {})
    print(f"\n[5] metadata keys: {list(metadata.keys())}")
    nza = metadata.get("node_zone_assignments", None)
    if nza:
        print(f"    node_zone_assignments: {len(nza)} entries")
        sample = list(nza.items())[:5]
        for k, v in sample:
            print(f"      {k!r} → {v!r}")
    else:
        print("    node_zone_assignments: NOT FOUND")

    # ── Deep search for any key containing 'zone' ─────────────────────────────
    print("\n[6] ALL keys containing 'zone' anywhere in JSON (deep scan):")
    _ZONE_RE = re.compile(r"zone", re.IGNORECASE)

    def find_zone_keys(obj, path="", depth=0):
        if depth > 8:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                full = f"{path}.{k}" if path else str(k)
                if _ZONE_RE.search(str(k)):
                    vtype = type(v).__name__
                    vlen  = f" len={len(v)}" if hasattr(v, "__len__") else ""
                    vsample = (list(v.keys())[:5]
                               if isinstance(v, dict)
                               else str(v)[:120])
                    print(f"    {full}  ({vtype}{vlen})")
                    print(f"      → {vsample}")
                find_zone_keys(v, full, depth + 1)
        elif isinstance(obj, list):
            for i, item in enumerate(obj[:5]):
                find_zone_keys(item, f"{path}[{i}]", depth + 1)

    find_zone_keys(raw)

    # ── Check for dedicated mapping files ────────────────────────────────────
    print("\n[7] Mapping files on disk:")
    for f in [
        "outputs/reports/node_zone_mapping.json",
        "outputs/reports/zone_mapping.json",
        "outputs/node_zone_map.json",
    ]:
        fp = Path(f)
        exists = fp.exists()
        print(f"    {f}: {'EXISTS' if exists else 'not found'}")
        if exists:
            try:
                d = json.loads(fp.read_text())
                print(f"      → {len(d)} entries, sample: {list(d.items())[:3]}")
            except Exception as e:
                print(f"      → parse error: {e}")

    print("\n" + "=" * 80)
    print("PASTE THE OUTPUT ABOVE and share it — we'll fix it immediately.")
    print("=" * 80)

if __name__ == "__main__":
    main()
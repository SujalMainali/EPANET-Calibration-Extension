import json
from pathlib import Path

path = Path("outputs/reports/best_params.json")

print("Checking:", path.resolve())

data = json.loads(path.read_text())

print("\nTOP LEVEL KEYS:")
print("=" * 60)

for k in data.keys():
    print(k)

print("\n" + "=" * 60)
print("METADATA KEYS")
print("=" * 60)

metadata = data.get("metadata", {})

if metadata:
    for k in metadata.keys():
        print(k)
else:
    print("NO METADATA")

print("\n" + "=" * 60)
print("BEST_RAW_PARAMS KEYS")
print("=" * 60)

brp = data.get("best_raw_params", {})

if brp:
    for k in brp.keys():
        print(k)
else:
    print("NO best_raw_params")

print("\n" + "=" * 60)
print("LEAKAGE KEYS")
print("=" * 60)

leakage = brp.get("leakage", {})

if leakage:
    for k in leakage.keys():
        print(k)
else:
    print("NO leakage block")

print("\n" + "=" * 60)
print("DONE")
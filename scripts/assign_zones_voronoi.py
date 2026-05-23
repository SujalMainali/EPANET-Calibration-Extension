"""Assign zones to junctions using Voronoi tessellation (nearest-neighbor).

This script assigns each junction to its nearest sensor node, creating a
Voronoi-like partitioning. Two optional rebalancing modes are provided:

- balance_method = "none"     : plain nearest-sensor Voronoi (default)
- balance_method = "capacity" : capacity-constrained assignment to balance node counts per zone
- balance_method = "kmeans"   : simple k-means clustering (zones not tied to sensors)

Usage:
    python scripts/assign_zones_voronoi.py

Output:
    outputs/node_zones_voronoi.csv (node_name, zone)
    outputs/node_zones_voronoi_detailed.csv (node_name, zone, sensor_center, distance_to_center)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import wntr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _compute_distance_matrix(
    node_coords: np.ndarray, sensor_coords: np.ndarray
) -> np.ndarray:
    # node_coords: (N,2), sensor_coords: (K,2) -> distances: (N,K)
    dx = node_coords[:, None, 0] - sensor_coords[None, :, 0]
    dy = node_coords[:, None, 1] - sensor_coords[None, :, 1]
    return np.sqrt(dx * dx + dy * dy)


def _capacity_assign(
    node_names: List[str],
    node_coords: np.ndarray,
    sensor_list: List[str],
    sensor_coords: np.ndarray,
    quotas: List[int],
) -> Tuple[List[str], np.ndarray]:
    """
    Greedy capacity-constrained assignment:
    - For each node in order of ascending distance to its nearest sensor,
      try to assign to nearest available sensor; if saturated, try next nearest.
    - Ensures each sensor receives at most its quota (some sensors may be under quota
      if total quotas != N, unassigned nodes are filled afterwards).
    Returns:
      zones (list of zone labels for each node, same order as node_names),
      distances (N-array of distance to assigned center)
    """
    N = len(node_names)
    K = len(sensor_list)
    distances = _compute_distance_matrix(node_coords, sensor_coords)  # (N,K)
    pref_idx = np.argsort(distances, axis=1)  # each row: sensor indices sorted by distance

    # Order nodes by their nearest distance (closest first)
    nearest_dists = distances[np.arange(N), pref_idx[:, 0]]
    order = np.argsort(nearest_dists, kind="stable")

    remaining = quotas.copy()
    assigned_zone_idx = np.full(N, -1, dtype=int)
    assigned_dist = np.full(N, np.nan, dtype=float)

    for i in order:
        # go through preferences
        for sidx in pref_idx[i]:
            if remaining[sidx] > 0:
                assigned_zone_idx[i] = sidx
                assigned_dist[i] = distances[i, sidx]
                remaining[sidx] -= 1
                break
        # if none available, leave unassigned for now

    # Fill any unassigned nodes (if any) into sensors that still have capacity,
    # or if all sensors are full, assign to nearest sensor (force).
    unassigned = np.where(assigned_zone_idx == -1)[0]
    if unassigned.size:
        # sensors with remaining capacity
        sensors_with_cap = [idx for idx, r in enumerate(remaining) if r > 0]
        if sensors_with_cap:
            # assign to nearest among sensors_with_cap
            for i in unassigned:
                # pick nearest sensor among those
                dsub = distances[i, sensors_with_cap]
                choice = sensors_with_cap[int(np.argmin(dsub))]
                assigned_zone_idx[i] = choice
                assigned_dist[i] = distances[i, choice]
                remaining[sensors_with_cap.index(choice)] -= 1
        else:
            # no capacity left; assign to absolute nearest (force)
            for i in unassigned:
                sidx = pref_idx[i, 0]
                assigned_zone_idx[i] = sidx
                assigned_dist[i] = distances[i, sidx]

    # Build zone labels Z_<sensor_index> to keep consistency with sensors
    zones = [f"Z_{sidx}" for sidx in assigned_zone_idx.tolist()]
    return zones, assigned_dist


def _kmeans_simple(
    node_coords: np.ndarray, k: int, init_centers: Optional[np.ndarray] = None, max_iter: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple k-means implemented with numpy.
    - init_centers: optional (k,2) initial centers (if None centers are sampled from nodes)
    Returns:
      labels (N-array), centers (k,2)
    """
    N = node_coords.shape[0]
    if init_centers is None:
        # choose k unique samples as initial centers
        idx = np.random.choice(N, size=k, replace=False)
        centers = node_coords[idx].astype(float)
    else:
        centers = init_centers.copy().astype(float)

    labels = np.full(N, -1, dtype=int)
    for _ in range(max_iter):
        d = _compute_distance_matrix(node_coords, centers)  # (N,k)
        new_labels = np.argmin(d, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # recompute centers
        for j in range(k):
            members = node_coords[labels == j]
            if len(members) > 0:
                centers[j] = members.mean(axis=0)
            # if no members, center remains unchanged
    return labels, centers


def assign_zones_voronoi(
    inp_path: str,
    sensor_nodes: Iterable[str],
    zone_labels: Optional[List[str]] = None,
    node_prefix: Optional[str] = None,
    balance_method: str = "none",  # "none" | "capacity" | "kmeans"
    max_imbalance_ratio: float = 2.0,
) -> pd.DataFrame:
    """
    Assign zones using Voronoi (nearest-neighbor) partitioning with optional balancing.

    Arguments:
        inp_path: path to EPANET INP
        sensor_nodes: iterable of sensor node IDs (kept as zone centers for "none" and "capacity")
        zone_labels: optional list of labels (must match len(sensor_nodes) if provided)
        node_prefix: optional prefix to restrict nodes assigned
        balance_method: "none" (plain Voronoi), "capacity" (balanced counts), "kmeans" (balanced clusters)
        max_imbalance_ratio: threshold used only for reporting (no automatic correction)
    """
    wn = wntr.network.WaterNetworkModel(str(Path(inp_path)))

    # Collect node coordinates
    all_coords = {}
    for node_name in wn.node_name_list:
        if node_prefix and not str(node_name).startswith(node_prefix):
            continue
        node = wn.get_node(node_name)
        if node is None:
            continue
        xy = getattr(node, "coordinates", None)
        if not xy or len(xy) < 2:
            continue
        try:
            x = float(xy[0])
            y = float(xy[1])
        except Exception:
            continue
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        all_coords[str(node_name)] = (x, y)

    if not all_coords:
        raise ValueError("No nodes found with coordinates. node_prefix=%r" % (node_prefix,))

    # Ensure sensors exist
    sensor_list = [str(s).strip() for s in sensor_nodes]
    sensor_coords = {}
    for s in sensor_list:
        if s not in all_coords:
            raise ValueError(f"Sensor node {s!r} not found in INP or missing coordinates")
        sensor_coords[s] = all_coords[s]

    if not sensor_coords:
        raise ValueError("No valid sensor nodes with coordinates")

    # Prepare arrays
    node_names = list(all_coords.keys())
    node_array = np.array([all_coords[n] for n in node_names], dtype=float)  # (N,2)
    sensors_ordered = list(sensor_coords.keys())
    sensor_array = np.array([sensor_coords[s] for s in sensors_ordered], dtype=float)  # (K,2)

    K = sensor_array.shape[0]
    N = node_array.shape[0]

    if zone_labels is None:
        zone_labels = [f"Z_{i}" for i in range(K)]
    if len(zone_labels) != K:
        raise ValueError("zone_labels length must equal number of sensors")

    # Compute plain Voronoi (nearest sensor)
    dist_mat = _compute_distance_matrix(node_array, sensor_array)  # (N,K)
    nearest_idx = np.argmin(dist_mat, axis=1)
    nearest_sensor = [sensors_ordered[i] for i in nearest_idx]
    nearest_zone = [zone_labels[i] for i in nearest_idx]
    nearest_dist = dist_mat[np.arange(N), nearest_idx]

    # Default assignment
    zones = nearest_zone
    distances = nearest_dist.copy()

    # If capacity balancing requested
    if balance_method == "capacity":
        # Compute quotas: distribute N as evenly as possible
        base = N // K
        remainder = N % K
        quotas = [base + (1 if i < remainder else 0) for i in range(K)]
        # But if any sensor is one of the sensor nodes that MUST have at least 1, quotas already ensure >=0
        # Run capacity-constrained greedy assignment
        zones_bal, distances_bal = _capacity_assign(node_names, node_array, sensors_ordered, sensor_array, quotas)
        zones = zones_bal
        distances = distances_bal

    elif balance_method == "kmeans":
        # Run kmeans with k=K (simple numpy implementation)
        labels, centers = _kmeans_simple(node_array, k=K, init_centers=sensor_array, max_iter=200)
        zones = [f"Z_{int(lbl)}" for lbl in labels]
        # compute distances to assigned centers
        assigned_center_coords = centers[labels]
        distances = np.sqrt(((node_array - assigned_center_coords) ** 2).sum(axis=1))
        # sensor_center reported as the nearest sensor to that center (optional)
        # map the center index to a sensor by nearest sensor center (so we have a sensor mapping if needed)
        # but note: kmeans zones are not strictly "sensor-based"
        # We'll keep sensor_center as nearest sensor to cluster center for downstream compatibility
        center_to_sensor_idx = np.argmin(_compute_distance_matrix(centers, sensor_array), axis=1)
        nearest_sensor_for_label = [sensors_ordered[i] for i in center_to_sensor_idx]
        # remap zone labels to sensors (Z_<sensor_idx>) if you prefer; for now keep Z_<kmeans_label>
        # but include sensor_center column below using nearest_sensor_for_label[labels[i]]

    # Build DataFrame
    df = pd.DataFrame(
        {
            "node_name": node_names,
            "zone": zones,
        }
    )

    # Add sensor_center and distance_to_center for backward compatibility:
    # For capacity and none: sensor_center is straightforward
    if balance_method in {"none", "capacity"}:
        df["sensor_center"] = [sensors_ordered[int(z.split("_", 1)[1])] for z in df["zone"]]
        df["distance_to_center"] = distances
    else:  # kmeans
        # For kmeans derive sensor_center by nearest sensor to cluster center
        labels_k = np.array([int(z.split("_", 1)[1]) for z in df["zone"]])
        # compute centers again to be safe
        labels_arr = labels_k
        centers = np.array([node_array[labels_arr == j].mean(axis=0) if np.any(labels_arr == j) else sensor_array[j] for j in range(K)])
        sensor_idx_by_center = np.argmin(_compute_distance_matrix(centers, sensor_array), axis=1)
        df["sensor_center"] = [sensors_ordered[sensor_idx_by_center[int(lbl)]] for lbl in labels_k]
        df["distance_to_center"] = distances

    # Compute diagnostics
    counts = df["zone"].value_counts().sort_index()
    min_c = int(counts.min())
    max_c = int(counts.max())
    mean_c = float(counts.mean())
    imbalance_ratio = float(max_c / max(1, min_c))

    diagnostics = {
        "n_nodes_total": int(N),
        "n_sensors": int(K),
        "counts": counts.to_dict(),
        "min": min_c,
        "max": max_c,
        "mean": mean_c,
        "imbalance_ratio": imbalance_ratio,
    }

    # Print summary
    print("\nZone assignment summary:")
    for z, c in sorted(counts.items()):
        print(f"  {z}: {int(c)} nodes")
    print(f"\nTotal nodes assigned: {N}")
    print(f"Min: {min_c}, Max: {max_c}, Mean: {mean_c:.1f}, Imbalance ratio: {imbalance_ratio:.2f}")

    if imbalance_ratio > max_imbalance_ratio:
        print(f"WARNING: imbalance_ratio {imbalance_ratio:.2f} > max_imbalance_ratio {max_imbalance_ratio:.2f}")

    return df


def main() -> None:
    """Main entry: configure variables here or call from other scripts."""
    INP_PATH = "models/PATTERN.inp"

    SENSOR_NODES = [
        "HOUSEEND_16032",
        "HOUSEEND_16239",
        "HOUSEEND_16317",
        "HOUSEEND_16426",
        "HOUSEEND_16547",
        "HOUSEEND_16598",
        "HOUSEEND_16702",
    ]

    # Choose balancing strategy: "none", "capacity", or "kmeans"
    BALANCE_METHOD = "capacity"  # recommended to reduce large imbalances
    MAX_IMBALANCE_RATIO = 2.0  # warn if ratio > 2.0

    NODE_PREFIX = None  # limit to prefix if you want

    OUT_CSV = "outputs/node_zones_voronoi.csv"

    print("Generating Voronoi zones (with optional balancing)...")
    print(f"  INP: {INP_PATH}")
    print(f"  Sensors: {len(SENSOR_NODES)}")
    print(f"  Balancing method: {BALANCE_METHOD}")

    df = assign_zones_voronoi(
        inp_path=INP_PATH,
        sensor_nodes=SENSOR_NODES,
        zone_labels=None,
        node_prefix=NODE_PREFIX,
        balance_method=BALANCE_METHOD,
        max_imbalance_ratio=MAX_IMBALANCE_RATIO,
    )

    # Save outputs
    out_path = Path(OUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_export = df[["node_name", "zone"]].copy()
    df_export.to_csv(out_path, index=False)
    out_detailed = out_path.parent / (out_path.stem + "_detailed.csv")
    df.to_csv(out_detailed, index=False)

    print(f"\nWrote: {out_path} (nodes={len(df)})")
    print(f"Wrote (detailed): {out_detailed}")
    print("\n✅ Zone assignment complete!")

if __name__ == "__main__":
    main()
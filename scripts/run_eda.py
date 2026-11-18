"""Dataset exploration script: produces the inventory table, data-quality log,
and a sample Folium map required by the hackathon's EDA deliverable.

Run with: python -m scripts.run_eda
"""

from __future__ import annotations

import json

import pandas as pd

from src.config import MAPS_DIR, RESULTS_DIR, STUDENT_TEST_DIR, TRAIN_DIR
from src.data.loader import TripDataLoader
from src.data.preprocessor import clean_track, validate_track
from src.visualization.map_plotter import create_folium_map


def inventory_partition(partition_dir, partition_name: str) -> pd.DataFrame:
    rows = []
    for trip_track in sorted(partition_dir.glob("*/*.track")):
        trip_id = trip_track.stem
        trip_dir = partition_dir / trip_id
        data = TripDataLoader.load_trip(trip_dir, trip_id)
        track_df, nodes_df, arcs_df, route = data["track"], data["nodes"], data["arcs"], data["route"]
        if track_df is None:
            continue
        report = validate_track(track_df)
        rows.append({
            "partition": partition_name,
            "trip_id": trip_id,
            "n_gps_points": report["n_points"],
            "n_nodes": len(nodes_df) if nodes_df is not None else 0,
            "n_arcs": len(arcs_df) if arcs_df is not None else 0,
            "n_route_steps": len(route) if route is not None else 0,
            "pct_1s_sampling": report.get("pct_1s_steps"),
            "max_gap_s": report.get("max_gap_s"),
            "lon_min": report["lon_range"][0],
            "lon_max": report["lon_range"][1],
            "lat_min": report["lat_range"][0],
            "lat_max": report["lat_range"][1],
        })
    return pd.DataFrame(rows)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)

    train_inv = inventory_partition(TRAIN_DIR, "train")
    test_inv = inventory_partition(STUDENT_TEST_DIR, "student_test")
    inventory = pd.concat([train_inv, test_inv], ignore_index=True)
    inventory.to_csv(RESULTS_DIR / "dataset_inventory.csv", index=False)

    print(f"Trips: train={len(train_inv)}, student_test={len(test_inv)}")
    print(inventory.groupby("partition")[["n_gps_points", "n_nodes", "n_arcs", "n_route_steps"]].describe().T)

    summary = {
        "n_train_trips": int(len(train_inv)),
        "n_student_test_trips": int(len(test_inv)),
        "mean_gps_points_per_trip": float(inventory["n_gps_points"].mean()),
        "mean_pct_1s_sampling": float(inventory["pct_1s_sampling"].mean()),
        "max_gap_seconds_observed": float(inventory["max_gap_s"].max()),
        "largest_road_network_nodes": int(inventory["n_nodes"].max()),
        "smallest_road_network_nodes": int(inventory["n_nodes"].min()),
    }
    with open(RESULTS_DIR / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # Representative Folium map: a mid-sized train trip with its reference route.
    sample_trip_id = train_inv.sort_values("n_gps_points").iloc[len(train_inv) // 2]["trip_id"]
    trip_dir = TRAIN_DIR / sample_trip_id
    data = TripDataLoader.load_trip(trip_dir, sample_trip_id)
    track_df = clean_track(data["track"])

    create_folium_map(
        track_df, data["nodes"], data["arcs"],
        predicted_arcs=None,
        ground_truth_route=list(data["route"]) if data["route"] is not None else None,
        output_html_path=MAPS_DIR / f"eda_sample_{sample_trip_id}.html",
    )
    print(f"Sample Folium map -> {MAPS_DIR / f'eda_sample_{sample_trip_id}.html'}")


if __name__ == "__main__":
    main()

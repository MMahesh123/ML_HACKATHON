"""Builds the supervised candidate-classification dataset for all trips in a partition.

For every trip: load raw files -> clean track -> align the reference .route to
the .track observations -> generate spatial candidates for every observation
-> engineer features -> label each (observation, candidate) row 1 if that
candidate is the aligned reference arc, else 0.

Output: a single parquet file per partition under data/processed/, with
columns FEATURE_COLUMNS + ["trip_id", "obs_idx", "arc_id", "label"].
Also records, per trip, the candidate miss rate (how often the reference arc
was absent from the candidate set at all) so it can be reported honestly.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    CANDIDATE_SEARCH_RADIUS_METERS,
    MAX_CANDIDATES_PER_OBSERVATION,
    PROCESSED_DATA_DIR,
    STUDENT_TEST_DIR,
    TRAIN_DIR,
)
from src.data.loader import TripDataLoader
from src.data.preprocessor import align_route_to_track, clean_track
from src.features.candidate import CandidateGenerator
from src.features.extractor import FEATURE_COLUMNS, attach_arc_headings, build_features_for_trip


def build_partition(partition_dir: Path, partition_name: str) -> pd.DataFrame:
    trip_ids = sorted(p.stem for p in partition_dir.glob("*/*.track"))
    all_rows = []
    quality_log: Dict[str, dict] = {}

    for trip_id in tqdm(trip_ids, desc=f"Building {partition_name}"):
        t0 = time.perf_counter()
        trip_data = TripDataLoader.load_trip(partition_dir / trip_id, trip_id)
        track_df, nodes_df, arcs_df, route_arr = (
            trip_data["track"], trip_data["nodes"], trip_data["arcs"], trip_data["route"]
        )
        if track_df is None or nodes_df is None or arcs_df is None:
            continue

        track_df = clean_track(track_df)
        reference_arcs = align_route_to_track(track_df, route_arr, arcs_df, nodes_df)

        cand_gen = CandidateGenerator(nodes_df, arcs_df, radius_meters=CANDIDATE_SEARCH_RADIUS_METERS)
        candidates_per_point = cand_gen.get_candidates_batch(
            track_df["longitude"].values, track_df["latitude"].values, top_k=MAX_CANDIDATES_PER_OBSERVATION
        )
        attach_arc_headings(candidates_per_point, cand_gen.projector, cand_gen.node_xy)

        feats = build_features_for_trip(track_df, candidates_per_point, arcs_df, cand_gen.projector)
        if len(feats) == 0:
            continue

        n_misses = 0
        n_labeled = 0
        if reference_arcs is not None:
            ref_series = pd.Series(reference_arcs, name="ref_arc")
            ref_series.index.name = "obs_idx"
            feats = feats.merge(ref_series.reset_index(), on="obs_idx", how="left")
            labeled_mask = feats["ref_arc"] >= 0
            feats["label"] = ((feats["arc_id"] == feats["ref_arc"]) & labeled_mask).astype(int)

            hit_per_obs = feats.groupby("obs_idx")["label"].max()
            labeled_obs = np.array([o for o in range(len(reference_arcs)) if reference_arcs[o] >= 0])
            n_labeled = len(labeled_obs)
            n_misses = int(n_labeled - hit_per_obs.reindex(labeled_obs, fill_value=0).sum())
            feats = feats.drop(columns=["ref_arc"])
        else:
            feats["label"] = -1  # unlabeled (evaluator-test style partition)

        feats["trip_id"] = trip_id
        all_rows.append(feats)

        quality_log[trip_id] = {
            "n_obs": len(track_df),
            "n_candidate_rows": len(feats),
            "candidate_miss_rate": (n_misses / n_labeled) if n_labeled else None,
            "build_time_s": time.perf_counter() - t0,
        }

    combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(
        columns=FEATURE_COLUMNS + ["arc_id", "obs_idx", "trip_id", "label"]
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / f"{partition_name}.parquet"
    combined.to_parquet(out_path, index=False)

    with open(PROCESSED_DATA_DIR / f"{partition_name}_quality_log.json", "w") as f:
        json.dump(quality_log, f, indent=2)

    print(f"[{partition_name}] {len(combined)} rows, {len(trip_ids)} trips -> {out_path}")
    miss_rates = [v["candidate_miss_rate"] for v in quality_log.values() if v["candidate_miss_rate"] is not None]
    if miss_rates:
        print(f"[{partition_name}] mean candidate miss rate: {np.mean(miss_rates):.4f}")

    return combined


if __name__ == "__main__":
    build_partition(TRAIN_DIR, "train")
    build_partition(STUDENT_TEST_DIR, "student_test")

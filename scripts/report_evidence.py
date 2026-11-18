"""Generates the supplementary evidence used in the technical report.

Only development partitions (train, student_test) are used here.

Outputs:
- artifacts/results/alignment_validation.csv: per-trip alignment error of the
  original greedy route-to-track alignment vs the order-preserving DP alignment.
- artifacts/results/final_model_student_test_by_trip.csv: classifier-only vs
  classifier+HMM for the exported tuned model, per student-test trip.
- reports/figures/alignment_error_greedy_vs_dp.png
- reports/figures/student_test_route_example.png

Run with: python -m scripts.report_evidence
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import RESULTS_DIR, STUDENT_TEST_DIR, TRAIN_DIR, PROJECT_ROOT
from src.data.loader import TripDataLoader
from src.data.preprocessor import align_route_to_track, clean_track
from src.geo import LocalProjector
from src.models.evaluate import calculate_route_continuity
from src.models.export import load_exported_model
from src.pipeline import run_inference

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def _greedy_alignment_baseline(track_df, route_arcs, arcs_df, nodes_df):
    """The original greedy nearest-midpoint alignment, kept only as a baseline."""
    projector = LocalProjector.from_coords(nodes_df["longitude"].values, nodes_df["latitude"].values)
    node_xy = projector.to_xy(nodes_df["longitude"].values, nodes_df["latitude"].values)
    arcs_indexed = arcs_df.set_index("arc_id")
    valid = [a for a in route_arcs if a in arcs_indexed.index]
    rf = arcs_indexed.loc[valid, "from_node_id"].values.astype(np.int64)
    rt = arcs_indexed.loc[valid, "to_node_id"].values.astype(np.int64)
    mid = (node_xy[rf] + node_xy[rt]) / 2.0
    track_xy = projector.to_xy(track_df["longitude"].values, track_df["latitude"].values)
    labels = np.full(len(track_df), -1, dtype=np.int64)
    last, n_route = 0, len(valid)
    window = max(20, n_route // max(len(track_df), 1) + 10)
    for i in range(len(track_df)):
        lo, hi = last, min(n_route, last + window)
        if hi <= lo:
            hi = n_route
        best = int(np.argmin(np.linalg.norm(mid[lo:hi] - track_xy[i], axis=1))) + lo
        labels[i] = valid[best]
        last = best
    return labels


def _distance_to_assigned_arc(track_df, labels, arcs_df, nodes_df):
    """Perpendicular distance (m) from each GPS fix to the arc it was aligned to."""
    projector = LocalProjector.from_coords(nodes_df["longitude"].values, nodes_df["latitude"].values)
    node_xy = projector.to_xy(nodes_df["longitude"].values, nodes_df["latitude"].values)
    gps = projector.to_xy(track_df["longitude"].values, track_df["latitude"].values)
    pos = {int(n): i for i, n in enumerate(nodes_df["node_id"].values)}
    ai = arcs_df.set_index("arc_id")
    a = node_xy[[pos[int(ai.loc[x, "from_node_id"])] for x in labels]]
    b = node_xy[[pos[int(ai.loc[x, "to_node_id"])] for x in labels]]
    ab = b - a
    l2 = np.einsum("ij,ij->i", ab, ab)
    l2 = np.where(l2 > 0, l2, 1.0)
    t = np.clip(np.einsum("ij,ij->i", gps - a, ab) / l2, 0.0, 1.0)
    return np.linalg.norm(gps - (a + t[:, None] * ab), axis=1)


def alignment_validation() -> pd.DataFrame:
    rows = []
    for partition, base in [("train", TRAIN_DIR), ("student_test", STUDENT_TEST_DIR)]:
        for trip_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            tid = trip_dir.name
            raw = TripDataLoader.load_trip(trip_dir, tid)
            if raw["route"] is None:
                continue
            track = clean_track(raw["track"])
            for method, fn in [("greedy", _greedy_alignment_baseline), ("dp", align_route_to_track)]:
                labels = fn(track, raw["route"], raw["arcs"], raw["nodes"])
                d = _distance_to_assigned_arc(track, labels, raw["arcs"], raw["nodes"])
                rows.append({
                    "partition": partition, "trip_id": tid, "method": method,
                    "n_points": len(track),
                    "median_error_m": float(np.median(d)),
                    "p90_error_m": float(np.percentile(d, 90)),
                    "pct_points_over_50m": float(np.mean(d > 50) * 100),
                })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "alignment_validation.csv", index=False)
    return df


def final_model_student_test() -> pd.DataFrame:
    loaded = load_exported_model()
    rows = []
    for trip_dir in sorted(p for p in STUDENT_TEST_DIR.iterdir() if p.is_dir()):
        tid = trip_dir.name
        r = run_inference(trip_dir, tid, loaded_model=loaded)
        raw = TripDataLoader.load_trip(trip_dir, tid)
        ref = align_route_to_track(r["track_df"], raw["route"], r["arcs_df"], r["nodes_df"])
        mask = ref >= 0
        y = ref[mask]
        clf = np.array([p if p is not None else -1 for p in r["predicted_arcs_classifier"]])[mask]
        hmm = np.array([p if p is not None else -1 for p in r["predicted_arcs_hmm"]])[mask]
        cands = r["candidates_per_point"]
        oracle = np.mean([ref[i] in [c["arc_id"] for c in cands[i]] for i in range(len(ref)) if mask[i]])
        seq_c = [a for a in r["predicted_arcs_classifier"] if a is not None]
        seq_h = [a for a in r["predicted_arcs_hmm"] if a is not None]
        rows.append({
            "trip_id": tid, "n_points": int(mask.sum()),
            "candidate_oracle_top1": float(oracle),
            "top1_classifier_only": float((y == clf).mean()),
            "top1_with_hmm": float((y == hmm).mean()),
            "continuity_classifier_only": calculate_route_continuity(seq_c, r["road_graph"]),
            "continuity_with_hmm": calculate_route_continuity(seq_h, r["road_graph"]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "final_model_student_test_by_trip.csv", index=False)
    return df


def tuned_vs_untuned() -> pd.DataFrame:
    """Evaluates untuned and tuned LightGBM with identical data, scaler and metrics."""
    import json
    import time

    from lightgbm import LGBMClassifier
    from sklearn.preprocessing import StandardScaler

    from src.features.extractor import FEATURE_COLUMNS
    from src.models.evaluate import evaluate_predictions
    from src.models.sequence_eval import evaluate_sequence_on_trips
    from src.models.train import load_partition, run_group_cv, subsample_per_trip
    from src.pipeline import LoadedModel

    best_params = json.loads((RESULTS_DIR / "bayesian_optimisation.json").read_text())["best_params"]
    train_df = subsample_per_trip(load_partition("train"), 600)
    student_df = subsample_per_trip(load_partition("student_test"), 600)
    scaler = StandardScaler().fit(train_df[FEATURE_COLUMNS].values)
    X, y, groups = scaler.transform(train_df[FEATURE_COLUMNS].values), train_df["label"].values, train_df["trip_id"].values
    Xs, ys = scaler.transform(student_df[FEATURE_COLUMNS].values), student_df["label"].values
    trip_ids = sorted(student_df["trip_id"].unique().tolist())

    rows = []
    for name, factory in [
        ("Untuned (LightGBM library defaults)", lambda: LGBMClassifier(random_state=42, verbose=-1)),
        ("Bayesian-tuned (Optuna TPE)", lambda: LGBMClassifier(random_state=42, verbose=-1, **best_params)),
    ]:
        cv = run_group_cv(factory(), X, y, groups)
        model = factory().fit(X, y)
        t0 = time.perf_counter()
        proba = model.predict_proba(Xs)
        preds = model.predict(Xs)
        infer_ms = (time.perf_counter() - t0) * 1000.0 / len(Xs)
        row = evaluate_predictions(ys, preds, y_prob=proba)
        seq = evaluate_sequence_on_trips(
            STUDENT_TEST_DIR, trip_ids,
            loaded_model=LoadedModel(model=model, scaler=scaler, feature_columns=FEATURE_COLUMNS, name=name),
        )
        rows.append({
            "Configuration": name,
            "CV_Macro_F1": cv["cv_macro_f1"],
            "StudentTest_Top1_ClassifierOnly": seq["mean_top1_acc_classifier"],
            "StudentTest_Top1_WithHMM": seq["mean_top1_acc_hmm"],
            "StudentTest_Candidate_Macro_F1": row["macro_f1"],
            "StudentTest_LogLoss": row.get("log_loss"),
            "Inference_ms_per_row": infer_ms,
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "tuned_vs_untuned_deployed_pipeline.csv", index=False)
    return df


def missing_value_analysis() -> pd.DataFrame:
    """Counts missing / invalid values per file type across train and student-test."""
    rows = []
    for partition, base in [("train", TRAIN_DIR), ("student_test", STUDENT_TEST_DIR)]:
        totals: dict = {}
        for trip_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            raw = TripDataLoader.load_trip(trip_dir, trip_dir.name)
            for kind in ["track", "nodes", "arcs", "route"]:
                obj = raw[kind]
                t = totals.setdefault(kind, {"files": 0, "missing_files": 0, "rows": 0, "nan_cells": 0})
                t["files"] += 1
                if obj is None:
                    t["missing_files"] += 1
                    continue
                frame = obj if isinstance(obj, pd.DataFrame) else pd.DataFrame({"arc_id": obj})
                t["rows"] += len(frame)
                t["nan_cells"] += int(frame.isna().sum().sum())
            track = raw["track"]
            if track is not None:
                ts = track["timestamp"].values
                t = totals["track"]
                t["non_monotonic_steps"] = t.get("non_monotonic_steps", 0) + int(np.sum(np.diff(ts) < 0))
                t["duplicate_timestamps"] = t.get("duplicate_timestamps", 0) + int(np.sum(np.diff(ts) == 0))
                t["invalid_coords"] = t.get("invalid_coords", 0) + int(
                    ((track["longitude"].abs() > 180) | (track["latitude"].abs() > 90)).sum()
                )
        for kind, t in totals.items():
            rows.append({"partition": partition, "file_type": kind, **t})
    df = pd.DataFrame(rows).fillna(0)
    df.to_csv(RESULTS_DIR / "missing_value_analysis.csv", index=False)
    return df


def plot_raw_trip(trip_id: str) -> None:
    """EDA view of one training trip: raw GPS against the provided road network."""
    raw = TripDataLoader.load_trip(TRAIN_DIR / trip_id, trip_id)
    track, nodes, arcs = clean_track(raw["track"]), raw["nodes"], raw["arcs"]
    proj = LocalProjector.from_coords(track["longitude"].values, track["latitude"].values)
    gps = proj.to_xy(track["longitude"].values, track["latitude"].values)
    nxy = proj.to_xy(nodes["longitude"].values, nodes["latitude"].values)
    pos = {int(n): i for i, n in enumerate(nodes["node_id"].values)}
    margin = 200.0
    x0, x1 = gps[:, 0].min() - margin, gps[:, 0].max() + margin
    y0, y1 = gps[:, 1].min() - margin, gps[:, 1].max() + margin
    fig, ax = plt.subplots(figsize=(7, 7))
    for f, t in zip(arcs["from_node_id"].values, arcs["to_node_id"].values):
        a, b = nxy[pos[int(f)]], nxy[pos[int(t)]]
        if x0 <= a[0] <= x1 and y0 <= a[1] <= y1:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#bbbbbb", lw=0.6, zorder=1)
    ax.scatter(gps[:, 0], gps[:, 1], s=3, c=np.arange(len(gps)), cmap="viridis", zorder=2)
    ax.plot([], [], color="#bbbbbb", lw=1, label="Road network (.nodes/.arcs)")
    ax.scatter([], [], s=10, color="#3b528b", label="GPS fixes (.track), coloured by time")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, local projection)")
    ax.set_ylabel("y (m, local projection)")
    ax.set_title(f"Training trip {trip_id}: raw GPS trajectory and road network")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_train_trip_raw.png", dpi=160)
    plt.close(fig)


def train_vs_cv_fit() -> pd.DataFrame:
    """Training-set vs cross-validated candidate macro-F1 for every model, as
    over/underfitting evidence. Same subsample and scaler as the benchmark."""
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler

    from src.features.extractor import FEATURE_COLUMNS
    from src.models.train import get_candidate_models, load_partition, subsample_per_trip

    train_df = subsample_per_trip(load_partition("train"), 600)
    scaler = StandardScaler().fit(train_df[FEATURE_COLUMNS].values)
    X, y = scaler.transform(train_df[FEATURE_COLUMNS].values), train_df["label"].values
    comparison = pd.read_csv(RESULTS_DIR / "model_comparison.csv").set_index("Model")
    rows = []
    for name, model in get_candidate_models().items():
        model.fit(X, y)
        train_f1 = f1_score(y, model.predict(X), average="macro", zero_division=0)
        rows.append({
            "Model": name,
            "Train_Macro_F1": train_f1,
            "CV_Macro_F1": comparison.loc[name, "CV_Macro_F1"],
            "StudentTest_Macro_F1": comparison.loc[name, "StudentTest_RowMacroF1"],
        })
        if name == "MLP":
            fig, ax1 = plt.subplots(figsize=(7, 4))
            ax1.plot(model.loss_curve_, color="#d62728", label="Training loss (log-loss)")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Training loss")
            ax2 = ax1.twinx()
            ax2.plot(model.validation_scores_, color="#1f77b4", label="Validation accuracy (10% hold-out)")
            ax2.set_ylabel("Validation accuracy")
            lines = ax1.get_lines() + ax2.get_lines()
            ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=8)
            ax1.set_title(f"MLP training history ({model.n_iter_} epochs, early stopping)")
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "mlp_training_history.png", dpi=160)
            plt.close(fig)
    df = pd.DataFrame(rows)
    df["Train_minus_CV"] = df["Train_Macro_F1"] - df["CV_Macro_F1"]
    df.to_csv(RESULTS_DIR / "train_vs_cv_fit.csv", index=False)
    return df


def reverse_direction_errors() -> pd.DataFrame:
    """Share of final-model HMM errors that pick the reverse-direction twin of
    the correct arc, per student-test trip."""
    loaded = load_exported_model()
    rows = []
    for trip_dir in sorted(p for p in STUDENT_TEST_DIR.iterdir() if p.is_dir()):
        tid = trip_dir.name
        r = run_inference(trip_dir, tid, loaded_model=loaded)
        raw = TripDataLoader.load_trip(trip_dir, tid)
        arcs = r["arcs_df"]
        ref = align_route_to_track(r["track_df"], raw["route"], arcs, r["nodes_df"])
        pair = {(int(f), int(t)): int(a) for a, f, t in zip(arcs["arc_id"], arcs["from_node_id"], arcs["to_node_id"])}
        twin = {a: pair[(t, f)] for (f, t), a in pair.items() if (t, f) in pair}
        mask = ref >= 0
        y = ref[mask]
        hmm = np.array([p if p is not None else -1 for p in r["predicted_arcs_hmm"]])[mask]
        clf = np.array([p if p is not None else -1 for p in r["predicted_arcs_classifier"]])[mask]
        wrong = hmm != y
        reverse = np.array([twin.get(int(y[i])) == int(hmm[i]) for i in range(len(y))])
        rows.append({
            "trip_id": tid,
            "top1_classifier_only": float((clf == y).mean()),
            "top1_with_hmm": float((hmm == y).mean()),
            "hmm_errors": int(wrong.sum()),
            "reverse_direction_errors": int((wrong & reverse).sum()),
            "reverse_share_of_errors": float((wrong & reverse).sum() / max(wrong.sum(), 1)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "hmm_reverse_direction_errors.csv", index=False)
    return df


def plot_alignment(df: pd.DataFrame) -> None:
    piv = df.pivot_table(index=["partition", "trip_id"], columns="method", values="median_error_m")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.logspace(0, np.log10(max(piv.values.max(), 10) * 1.2), 40)
    ax.hist(piv["greedy"], bins=bins, alpha=0.6, label="Greedy nearest-midpoint", color="#d62728")
    ax.hist(piv["dp"], bins=bins, alpha=0.6, label="Order-preserving DP", color="#1f77b4")
    ax.set_xscale("log")
    ax.set_xlabel("Per-trip median distance from GPS fix to aligned reference arc (m, log scale)")
    ax.set_ylabel("Number of trips")
    ax.set_title("Route-to-track alignment error (train + student-test, 90 trips)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "alignment_error_greedy_vs_dp.png", dpi=160)
    plt.close(fig)


def plot_route_example(trip_id: str) -> None:
    loaded = load_exported_model()
    r = run_inference(STUDENT_TEST_DIR / trip_id, trip_id, loaded_model=loaded)
    track, nodes, arcs = r["track_df"], r["nodes_df"], r["arcs_df"]
    proj = LocalProjector.from_coords(track["longitude"].values, track["latitude"].values)
    gps = proj.to_xy(track["longitude"].values, track["latitude"].values)
    nxy = proj.to_xy(nodes["longitude"].values, nodes["latitude"].values)
    pos = {int(n): i for i, n in enumerate(nodes["node_id"].values)}
    margin = 150.0
    x0, x1 = gps[:, 0].min() - margin, gps[:, 0].max() + margin
    y0, y1 = gps[:, 1].min() - margin, gps[:, 1].max() + margin

    fig, ax = plt.subplots(figsize=(7, 7))
    for f, t in zip(arcs["from_node_id"].values, arcs["to_node_id"].values):
        a, b = nxy[pos[int(f)]], nxy[pos[int(t)]]
        if x0 <= a[0] <= x1 and y0 <= a[1] <= y1:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#bbbbbb", lw=0.6, zorder=1)
    ai = arcs.set_index("arc_id")
    for arc_id in dict.fromkeys(a for a in r["predicted_arcs_hmm"] if a is not None):
        a = nxy[pos[int(ai.loc[arc_id, "from_node_id"])]]
        b = nxy[pos[int(ai.loc[arc_id, "to_node_id"])]]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#1f77b4", lw=2.4, zorder=3)
    ax.scatter(gps[:, 0], gps[:, 1], s=4, color="#d62728", zorder=2, label="Noisy GPS observations")
    ax.plot([], [], color="#1f77b4", lw=2.4, label="Reconstructed route (classifier + HMM)")
    ax.plot([], [], color="#bbbbbb", lw=1, label="Road network")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, local projection)")
    ax.set_ylabel("y (m, local projection)")
    ax.set_title(f"Student-test trip {trip_id}: reconstruction")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "student_test_route_example.png", dpi=160)
    plt.close(fig)


def main():
    print("Missing values:\n", missing_value_analysis().to_string(index=False))
    plot_raw_trip("00000002")
    print("\nTrain vs CV fit:\n", train_vs_cv_fit().to_string(index=False))
    print("\nReverse-direction errors:\n", reverse_direction_errors().to_string(index=False))
    print("\nTuned vs untuned:\n", tuned_vs_untuned().to_string(index=False))

    align_df = alignment_validation()
    plot_alignment(align_df)
    summary = align_df.groupby("method")[["median_error_m", "p90_error_m", "pct_points_over_50m"]].median()
    print("Alignment validation (median across 90 trips):\n", summary)

    st = final_model_student_test()
    print("\nFinal model on student-test:\n", st.to_string(index=False))
    print("\nMeans:\n", st.drop(columns=["trip_id", "n_points"]).mean())

    best = st.sort_values("top1_with_hmm", ascending=False).iloc[len(st) // 2]["trip_id"]
    plot_route_example(best)
    print(f"\nRoute example figure: student-test trip {best}")


if __name__ == "__main__":
    main()

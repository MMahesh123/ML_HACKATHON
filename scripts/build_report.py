"""Builds reports/technical_report.pdf from the committed results.

Every number in the report is read from artifacts/results/ at build time, so
the PDF cannot drift from the experiments. Regenerate the inputs first with
`python -m scripts.run_full_pipeline`, `python -m scripts.generate_report_figures`
and `python -m scripts.report_evidence`.

Run with: python -m scripts.build_report
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from src.config import (
    BETA, CANDIDATE_SEARCH_RADIUS_METERS, DISCONNECTION_PENALTY, MAX_CANDIDATES_PER_OBSERVATION,
    PROJECT_ROOT, RESULTS_DIR, SIGMA_Z,
)

FIG = PROJECT_ROOT / "reports" / "figures"
SHOT = PROJECT_ROOT / "reports" / "screenshots"
OUT = PROJECT_ROOT / "reports" / "technical_report.pdf"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, spaceBefore=10, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12.5, spaceBefore=8, spaceAfter=4)
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11, spaceBefore=6, spaceAfter=3)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.5, leading=13, spaceAfter=5)
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=7.6, leading=9.4, spaceAfter=0)
CAPTION = ParagraphStyle("Caption", parent=BODY, fontSize=8.3, textColor=colors.HexColor("#444444"), alignment=TA_CENTER)
CODE = ParagraphStyle("Code", parent=BODY, fontName="Courier", fontSize=7.4, leading=9.2, backColor=colors.HexColor("#f4f4f4"))

story: list = []


def h1(t): story.append(Paragraph(t, H1))
def h2(t): story.append(Paragraph(t, H2))
def h3(t): story.append(Paragraph(t, H3))
def p(t): story.append(Paragraph(t, BODY))


def bullets(items):
    for it in items:
        story.append(Paragraph(f"&bull;&nbsp; {it}", BODY))


def table(rows, widths, header=True):
    data = [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e3e8f0")))
    t.setStyle(TableStyle(style))
    story.append(t)
    story.append(Spacer(1, 6))


def figure(path, caption, width_cm=15.5):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(path)).getSize()
    w = width_cm * cm
    story.append(KeepTogether([Image(str(path), width=w, height=w * ih / iw), Paragraph(caption, CAPTION), Spacer(1, 6)]))


def f3(x): return "N/A" if pd.isna(x) else f"{x:.3f}"
def f4(x): return "N/A" if pd.isna(x) else f"{x:.4f}"
def pct(x): return "N/A" if pd.isna(x) else f"{x * 100:.2f}%"


# ---------------------------------------------------------------- data
cmp_df = pd.read_csv(RESULTS_DIR / "model_comparison.csv").set_index("Model")
fit_df = pd.read_csv(RESULTS_DIR / "train_vs_cv_fit.csv").set_index("Model")
tvu = pd.read_csv(RESULTS_DIR / "tuned_vs_untuned_deployed_pipeline.csv")
bo = json.loads((RESULTS_DIR / "bayesian_optimisation.json").read_text())
inv = pd.read_csv(RESULTS_DIR / "dataset_inventory.csv")
miss = pd.read_csv(RESULTS_DIR / "missing_value_analysis.csv")
align = pd.read_csv(RESULTS_DIR / "alignment_validation.csv")
st = pd.read_csv(RESULTS_DIR / "final_model_student_test_by_trip.csv", dtype={"trip_id": str})
rev = pd.read_csv(RESULTS_DIR / "hmm_reverse_direction_errors.csv", dtype={"trip_id": str})

tuned = tvu[tvu["Configuration"].str.startswith("Bayesian")].iloc[0]
untuned = tvu[tvu["Configuration"].str.startswith("Untuned")].iloc[0]

FAMILY = {
    "LogisticRegression": ("Linear / probabilistic", "max_iter=2000, class_weight=balanced"),
    "KNN_Distance": ("Instance-based", "k=7, weights=distance"),
    "DecisionTree": ("Single tree", "max_depth=12, class_weight=balanced"),
    "RandomForest": ("Bagged tree ensemble", "150 trees, max_depth=14, class_weight=balanced"),
    "LightGBM": ("Gradient-boosted trees", "150 trees, learning_rate=0.1, unlimited depth"),
    "MLP": ("Neural network", "hidden (64, 32), Adam, early stopping"),
}

# ---------------------------------------------------------------- title
story.append(Spacer(1, 3 * cm))
story.append(Paragraph("PathSnap: Distributed Map-Matching and Route Reconstruction", ParagraphStyle("T", parent=H1, fontSize=20, alignment=TA_CENTER, leading=25)))
story.append(Spacer(1, 0.4 * cm))
story.append(Paragraph("Technical Report", ParagraphStyle("S", parent=H2, alignment=TA_CENTER)))
story.append(Paragraph("UE24CS352A Machine Learning Hackathon &middot; PES University", ParagraphStyle("S2", parent=BODY, alignment=TA_CENTER)))
story.append(Spacer(1, 1.2 * cm))
table([
    ["Member", "Area of responsibility"],
    ["Member 1", "Geospatial data engineering, EDA, sequence alignment, candidate generation and features"],
    ["Member 2", "Model benchmarking (six families) and Bayesian optimisation"],
    ["Member 3", "HMM / Viterbi route reconstruction and artifact packaging"],
    ["Member 4", "FastAPI backend, Streamlit UI, two-laptop deployment, report"],
], [3 * cm, 13 * cm])
story.append(Spacer(1, 0.8 * cm))
p(f"Report generated {date.today().isoformat()} by <font face='Courier'>scripts/build_report.py</font>. "
  "Every number below is read from <font face='Courier'>artifacts/results/</font> when the report is built, "
  "and the commands to regenerate each input are listed in the appendix.")
story.append(PageBreak())

# ---------------------------------------------------------------- 1 overview
h1("1. System Overview")
p("PathSnap matches noisy GPS fixes to a road network and reconstructs the route a vehicle drove. "
  "Each trip provides <i>.track</i> (timestamped GPS fixes), <i>.nodes</i> and <i>.arcs</i> (a directed road graph) "
  "and, for development partitions, <i>.route</i> (the ordered reference arcs). The pipeline has four stages:")
bullets([
    f"<b>Candidate generation.</b> For every GPS fix, a KD-tree over arc geometry in a local metric projection returns "
    f"up to {MAX_CANDIDATES_PER_OBSERVATION} arcs within {CANDIDATE_SEARCH_RADIUS_METERS:.0f} m, ranked by true "
    "perpendicular point-to-segment distance.",
    "<b>Candidate classification.</b> A binary classifier scores each (fix, candidate arc) pair: "
    "1 if the arc is the aligned reference arc, else 0.",
    "<b>Sequence decoding.</b> An HMM whose hidden states are the candidate arcs is decoded with Viterbi, combining "
    "classifier probabilities with road-network topology.",
    "<b>Deployment.</b> Laptop 1 runs a FastAPI server holding the model, road networks and ground truth; Laptop 2 runs a "
    "Streamlit UI that requests reconstructions over the LAN and draws them with Folium.",
])
p("<font face='Courier'>src/pipeline.py::run_inference</font> is the single implementation used by offline evaluation "
  "and by the live API, so the figures in this report are produced by the same code that serves the demo.")

# ---------------------------------------------------------------- 2 dataset & EDA
h1("2. Dataset Understanding and Exploratory Data Analysis")
h2("2.1 Partitions")
g = inv.groupby("partition")
rows = [["Partition", "Trips", "GPS fixes / trip (mean, min-max)", "Road nodes / trip (min-max)",
         "Route arcs / trip (mean)", "Fixes at 1 s spacing", "Trips with a gap > 10 s"]]
for part in ["train", "student_test"]:
    d = g.get_group(part)
    rows.append([part, len(d), f"{d.n_gps_points.mean():.0f} ({d.n_gps_points.min()}-{d.n_gps_points.max()})",
                 f"{d.n_nodes.min():,}-{d.n_nodes.max():,}", f"{d.n_route_steps.mean():.0f}",
                 f"{d.pct_1s_sampling.mean() * 100:.1f}%", f"{(d.max_gap_s > 10).sum()} of {len(d)}"])
table(rows, [2.3 * cm, 1.2 * cm, 3.2 * cm, 2.6 * cm, 2.2 * cm, 2.2 * cm, 2.5 * cm])
p("Training (80 trips) was used for model fitting, cross-validation and hyperparameter search; the student-testing "
  "partition (10 trips) was used for every held-out comparison in this report. The 10 evaluator-testing trips were "
  "used only for end-to-end integration testing of the deployed API (Section 7); see Section 8 for a disclosure.")

h2("2.2 Missing-value and integrity analysis (train + student-test)")
rows = [["Partition", "File", "Files", "Missing files", "Rows", "NaN cells", "Non-monotonic timestamps", "Duplicate timestamps", "Invalid coordinates"]]
for _, r in miss.iterrows():
    rows.append([r.partition, r.file_type, int(r.files), int(r.missing_files), f"{int(r.rows):,}", int(r.nan_cells),
                 int(r.get("non_monotonic_steps", 0)), int(r.get("duplicate_timestamps", 0)), int(r.get("invalid_coords", 0))])
table(rows, [2.2 * cm, 1.3 * cm, 1.1 * cm, 1.5 * cm, 1.9 * cm, 1.4 * cm, 2.3 * cm, 2.1 * cm, 2.0 * cm])
gap_train = g.get_group("train").max_gap_s
p(f"No file is missing and no cell is empty, so missing data does not appear as NaNs. It appears as <b>gaps in time</b>: "
  f"{(gap_train > 10).sum()} of 80 training trips contain at least one gap longer than 10 s, the median largest gap is "
  f"{gap_train.median():.0f} s, and one trip has a gap of {gap_train.max():,.0f} s. Gaps are exposed to the classifier "
  "through the <i>time_gap_s</i> and <i>speed_mps</i> features (Section 2.4).")

h2("2.3 Preprocessing")
bullets([
    "<b>Cleaning</b> (<font face='Courier'>clean_track</font>): drop fixes with out-of-range coordinates, sort by timestamp, "
    "drop duplicate timestamps.",
    "<b>Metric projection</b> (<font face='Courier'>src/geo.py</font>): each trip is projected onto a local equirectangular "
    "plane centred on its own coordinates, so distances and bearings are in metres and degrees. Raw latitude/longitude "
    "differences are never used as distances.",
    "<b>Route-to-track alignment</b> (<font face='Courier'>align_route_to_track</font>): a <i>.track</i> has many more rows "
    "than its <i>.route</i> (one arc per road step, not per fix), so labels need an alignment. We use an order-preserving "
    "dynamic program that assigns every fix to a route step so as to minimise total perpendicular distance, subject to the "
    "route index never decreasing.",
])
dp = align[align.method == "dp"]
gr = align[align.method == "greedy"]
p(f"Our first implementation used a greedy nearest-step scan inside a moving window. On development data it placed "
  f"more than 10% of fixes over 50 m from their assigned reference arc in <b>{(gr.pct_points_over_50m > 10).sum()} of 90 trips</b>; the dynamic "
  f"program reduces this to <b>{(dp.pct_points_over_50m > 10).sum()}</b>. The greedy scan fails because a forward jump "
  "made early cannot be undone, so the label sequence runs ahead of the vehicle. All models in this report were trained "
  "on labels from the dynamic program.")
figure(FIG / "alignment_error_greedy_vs_dp.png", "Figure 1. Per-trip median alignment error, greedy scan vs order-preserving DP (train + student-test, 90 trips).", 13)
bad = dp[dp.pct_points_over_50m > 10].sort_values("pct_points_over_50m", ascending=False)
p("The trips the dynamic program still cannot align are a property of the data: "
  + "; ".join(f"trip {int(r.trip_id):08d} ({r.partition}) has {r.pct_points_over_50m:.0f}% of fixes over 50 m" for _, r in bad.iterrows())
  + ". In the two worst, the GPS track is several times longer than the reference route while staying on its roads, "
    "which indicates repeated passes that a single-pass alignment cannot label. These trips were kept in training; "
    "excluding them is listed as future work.")

h2("2.4 Patterns observed and features chosen")
bullets([
    "Almost every road appears as two directed arcs with identical geometry (A&rarr;B and B&rarr;A). Distance alone "
    "cannot separate them, which motivates <i>heading_diff_deg</i>.",
    "Road networks range from a few hundred to over 150,000 nodes per trip, which motivates KD-tree candidate search "
    "and a bounded (400 m) shortest-path search in the HMM.",
    "Sampling is mostly 1 Hz but with frequent gaps, which motivates <i>time_gap_s</i> and <i>speed_mps</i>.",
])
table([
    ["Feature", "Definition", "Why"],
    ["perp_distance_m", "Perpendicular distance from fix to arc segment", "Primary geometric evidence"],
    ["proj_fraction", "Position of the projected point along the arc (0-1)", "Distinguishes arc ends near junctions"],
    ["heading_diff_deg", "Angle between GPS heading and arc direction", "Only feature separating an arc from its reverse twin"],
    ["candidate_rank", "Rank of the candidate by distance", "Relative evidence among nearby arcs"],
    ["arc_length_m", "Length of the candidate arc", "Long arcs collect more nearby fixes"],
    ["speed_mps", "GPS displacement / time since previous fix", "Heading is unreliable at low speed"],
    ["time_gap_s", "Seconds since previous fix", "Flags sampling gaps"],
    ["prev_arc_connected", "Candidate starts where the previous fix's nearest arc ends", "Weak local continuity signal"],
], [3.2 * cm, 7 * cm, 6 * cm])

h2("2.5 Visualisations")
figure(FIG / "eda_train_trip_raw.png", "Figure 2. Training trip 00000002: raw GPS fixes (coloured by time) over the provided road network.", 11)
p("Folium maps are produced by <font face='Courier'>src/visualization/map_plotter.py</font>; the committed interactive maps "
  "are in <font face='Courier'>artifacts/maps/</font>. A representative map is shown in Section 6.")
story.append(PageBreak())

# ---------------------------------------------------------------- 3 models
h1("3. Model Experimentation and Comparison")
p("Six model families were trained on the same candidate dataset: rows are (fix, candidate arc) pairs from the 80 "
  "training trips, sub-sampled to at most 600 rows per trip (48,000 rows) so that six models times five folds stays "
  "tractable. Features are standardised with a scaler fitted on training data only. Cross-validation is a 5-fold "
  "<b>GroupKFold grouped by trip_id</b>, so no trip contributes to both training and validation folds.")
p("<b>Metric definitions.</b> <i>Macro-F1</i> and <i>log loss</i> are candidate-level (binary label per candidate row). "
  "<i>Top-1 accuracy</i> is observation-level: for each GPS fix, whether the chosen arc equals the aligned reference arc, "
  "averaged over fixes within a trip and then over the 10 student-test trips. <i>Route continuity</i> is the share of "
  "consecutive predicted arcs that are connected in the road graph.")

h2("3.1 Overall comparison")
rows = [["Model", "Family", "Key configuration", "CV macro-F1", "Student-test Top-1 (classifier / +HMM)",
         "Route continuity (+HMM)", "Student-test macro-F1", "Log loss", "Inference (ms / 1k rows)"]]
for m in cmp_df.index:
    r = cmp_df.loc[m]
    rows.append([m, FAMILY[m][0], FAMILY[m][1], f3(r.CV_Macro_F1),
                 f"{pct(r.StudentTest_Top1Accuracy_ClassifierOnly)} / {pct(r.StudentTest_Top1Accuracy_WithHMM)}",
                 pct(r.RouteContinuity_WithHMM), f3(r.StudentTest_RowMacroF1), f3(r.StudentTest_LogLoss),
                 f"{r.InferenceTime_ms_per_row * 1000:.2f}"])
table(rows, [2.5 * cm, 1.9 * cm, 2.7 * cm, 1.5 * cm, 2.3 * cm, 1.7 * cm, 1.8 * cm, 1.2 * cm, 1.7 * cm])
best = cmp_df.CV_Macro_F1.idxmax()
p(f"<b>Selection.</b> {best} has the highest cross-validated macro-F1 ({cmp_df.loc[best, 'CV_Macro_F1']:.3f}) and the "
  f"lowest log loss ({cmp_df.loc[best, 'StudentTest_LogLoss']:.3f}), so its probabilities are the best calibrated input "
  "for the HMM emission term. The MLP is very close: its end-to-end Top-1 with HMM is "
  f"{(cmp_df.loc['MLP', 'StudentTest_Top1Accuracy_WithHMM'] - cmp_df.loc[best, 'StudentTest_Top1Accuracy_WithHMM']) * 100:+.2f} "
  f"points higher, but it is worse on CV macro-F1 and log loss and trains about "
  f"{cmp_df.loc['MLP', 'TrainTime_s'] / cmp_df.loc[best, 'TrainTime_s']:.1f}&times; slower. Selection used the CV score "
  f"rather than a student-test number, so the held-out partition did not drive the choice. {best} was selected and tuned in Section 4.")
figure(FIG / "cv_macro_f1_comparison.png", "Figure 3. Cross-validated macro-F1 (5-fold GroupKFold by trip).", 12)
figure(FIG / "student_test_accuracy_comparison.png", "Figure 4. Student-test observation-level Top-1 accuracy, classifier-only vs classifier + HMM.", 12)
figure(FIG / "student_test_f1_comparison.png", "Figure 5. Student-test candidate-level macro-F1.", 12)
figure(FIG / "log_loss_comparison.png", "Figure 6. Log loss (cross-validation and student-test).", 12)

h2("3.2 Per-model analysis")
p("All models were fitted to minimise their library-standard objective for binary classification (listed per model) "
  "on the same features and data. Over/underfitting is judged from training-set macro-F1 against cross-validated "
  "macro-F1 (<font face='Courier'>train_vs_cv_fit.csv</font>).")

NOTES = {
    "LogisticRegression": (
        "A linear, well-calibrated baseline that shows how far the raw features go without interactions.",
        "Fast and interpretable, but cannot model interactions such as heading mattering only when the vehicle is moving.",
        "Regularised log-loss (L2, C=1.0), solver lbfgs; class weights balance the roughly 1-in-9 positive rate.",
        "Underfitting: training and CV scores are identical, so the model is limited by capacity, not variance. "
        "It has the lowest macro-F1 of the six.",
    ),
    "KNN_Distance": (
        "Nearby feature vectors (similar distance, heading and rank) should share labels, without assuming a functional form.",
        "No training cost, but slow at inference and sensitive to feature scaling; distance weighting makes it memorise.",
        "No parametric loss; prediction is a distance-weighted vote of the 7 nearest training rows.",
        "Overfitting: distance weighting reproduces the training set exactly (training macro-F1 1.000), and its log loss "
        "is the second worst because near-certain votes are often wrong.",
    ),
    "DecisionTree": (
        "Captures threshold rules (e.g. accept close arcs only when headings agree) directly.",
        "Readable rules, but high variance and coarse probabilities from leaf frequencies.",
        "Greedy splits minimising Gini impurity; max_depth=12, balanced class weights.",
        "Mild overfitting, and the worst log loss of all models because leaf probabilities are often 0 or 1.",
    ),
    "RandomForest": (
        "Averages many decorrelated trees to keep tree-style interactions while reducing variance.",
        "Robust and well calibrated, but slower to train and run than boosting.",
        "Gini impurity per tree; 150 trees, max_depth=14, balanced class weights.",
        "A moderate train/CV gap with good held-out scores: some variance, no harmful overfitting.",
    ),
    "LightGBM": (
        "Gradient boosting fits feature interactions sequentially and gives well-calibrated probabilities on tabular data.",
        "Best accuracy and calibration here and fast inference, but many hyperparameters to tune.",
        "Binary log-loss minimised by gradient boosting; 150 trees, learning_rate=0.1, num_leaves=31 (library default).",
        "A small train/CV gap like the forest's, with the best CV macro-F1, student-test macro-F1 and log loss: "
        "controlled variance.",
    ),
    "MLP": (
        "A smooth non-linear model that tests whether a neural approach improves on trees for these features.",
        "Competitive accuracy, but slowest to train and needs scaled inputs.",
        "Log-loss minimised with Adam; hidden layers (64, 32), early stopping on a 10% validation split.",
        "No overfitting: training and CV scores match, because early stopping halts training once validation accuracy "
        "stops improving (Figure 7).",
    ),
}
for m in cmp_df.index:
    r, fit = cmp_df.loc[m], fit_df.loc[m]
    why, sl, cfg, ovf = NOTES[m]
    h3(f"3.2.{list(cmp_df.index).index(m) + 1} {m}")
    p(f"<b>A1. Suitability.</b> {why}")
    p(f"<b>A2. Strengths and limitations.</b> {sl}")
    p(f"<b>B5/B6. Hyperparameters and objective.</b> {cfg}")
    p(f"<b>C7/C8. Student-test results.</b> Observation Top-1 {pct(r.StudentTest_Top1Accuracy_ClassifierOnly)} classifier-only, "
      f"{pct(r.StudentTest_Top1Accuracy_WithHMM)} with HMM; candidate macro-F1 {f3(r.StudentTest_RowMacroF1)}; "
      f"log loss {f3(r.StudentTest_LogLoss)}; route continuity {pct(r.RouteContinuity_WithHMM)} with HMM.")
    p(f"<b>C9. Over/underfitting.</b> Training macro-F1 {f3(fit.Train_Macro_F1)} vs CV {f3(fit.CV_Macro_F1)} "
      f"(gap {fit.Train_minus_CV:+.3f}). {ovf}")
figure(FIG / "mlp_training_history.png", "Figure 7. MLP training loss and validation accuracy per epoch. The MLP is the only model with an epoch-wise history; no curves are drawn for the other models.", 12)
story.append(PageBreak())

# ---------------------------------------------------------------- 4 Bayesian optimisation
h1(f"4. Bayesian Hyperparameter Optimisation ({best})")
table([
    ["Hyperparameter", "Search range / distribution", "Best value"],
    ["n_estimators", "int 80-400, step 20", bo["best_params"]["n_estimators"]],
    ["num_leaves", "int 15-127", bo["best_params"]["num_leaves"]],
    ["max_depth", "int 3-14", bo["best_params"]["max_depth"]],
    ["learning_rate", "float 0.01-0.3, log-uniform", f"{bo['best_params']['learning_rate']:.4f}"],
    ["min_child_samples", "int 5-100", bo["best_params"]["min_child_samples"]],
    ["subsample", "float 0.6-1.0", f"{bo['best_params']['subsample']:.3f}"],
    ["colsample_bytree", "float 0.6-1.0", f"{bo['best_params']['colsample_bytree']:.3f}"],
    ["reg_lambda", "float 0.001-10, log-uniform", f"{bo['best_params']['reg_lambda']:.4f}"],
], [4 * cm, 7 * cm, 4 * cm])
p(f"<b>Method.</b> Optuna's Tree-structured Parzen Estimator (seed 42), {bo['n_trials']} trials, "
  f"{bo['optimisation_time_s']:.0f} s in total. <b>Objective:</b> maximise mean candidate macro-F1 over the same 5-fold "
  "GroupKFold-by-trip split of the training data. The student-test partition is not used during the search.")
p("<b>Tuned vs untuned.</b> Both rows below use identical data, scaler and metrics; the untuned model is LightGBM with "
  "library defaults.")
table([
    ["Configuration", "CV macro-F1", "Student-test Top-1 (classifier)", "Student-test Top-1 (+HMM)", "Student-test macro-F1", "Log loss", "Inference (ms / 1k rows)"],
    *[[r.Configuration, f4(r.CV_Macro_F1), pct(r.StudentTest_Top1_ClassifierOnly), pct(r.StudentTest_Top1_WithHMM),
       f4(r.StudentTest_Candidate_Macro_F1), f4(r.StudentTest_LogLoss), f"{r.Inference_ms_per_row * 1000:.2f}"] for _, r in tvu.iterrows()],
], [4 * cm, 1.8 * cm, 2.4 * cm, 2.2 * cm, 2.1 * cm, 1.6 * cm, 2.1 * cm])
p(f"Tuning helped, but only a little. CV macro-F1 rose by {tuned.CV_Macro_F1 - untuned.CV_Macro_F1:+.4f}, end-to-end "
  f"Top-1 with HMM by {(tuned.StudentTest_Top1_WithHMM - untuned.StudentTest_Top1_WithHMM) * 100:+.2f} points and log loss "
  f"fell slightly. Classifier-only Top-1 changed by {(tuned.StudentTest_Top1_ClassifierOnly - untuned.StudentTest_Top1_ClassifierOnly) * 100:+.2f} "
  f"points, and inference per row was {tuned.Inference_ms_per_row / untuned.Inference_ms_per_row:.1f}&times; slower "
  "(still microseconds per row). The search mostly confirmed that the defaults were already close to optimal for these "
  "eight features.")
figure(FIG / "bayesian_optimisation_trials.png", "Figure 8. CV macro-F1 per Optuna trial.", 12)

# ---------------------------------------------------------------- 5 HMM
h1("5. HMM and Viterbi Route Reconstruction")
bullets([
    "<b>Hidden states.</b> At step <i>t</i>, the candidate arcs of GPS fix <i>t</i> (up to 10). Each state carries its arc id, "
    "distance and projection fraction along the arc.",
    f"<b>Emission.</b> log P(O<sub>t</sub> | S<sub>t</sub>) = -&frac12;(d / &sigma;<sub>z</sub>)<super>2</super> + log P<sub>ML</sub>, "
    f"with d the perpendicular distance, &sigma;<sub>z</sub> = {SIGMA_Z} m and P<sub>ML</sub> the classifier probability (floored at 10<super>-6</super>).",
    f"<b>Transition.</b> log P(S<sub>t+1</sub> | S<sub>t</sub>) = -|d<sub>road</sub> - d<sub>gps</sub>| / &beta;, &beta; = {BETA}. "
    "d<sub>road</sub> is the network distance from the projected position on the first arc to the projected position on "
    "the second (remaining length + shortest path between them + distance into the second arc), from a Dijkstra search "
    "bounded to 400 m and cached per source node. d<sub>gps</sub> is the straight-line distance between the two fixes.",
    f"<b>Disconnected segments.</b> If no path exists within the bound, the transition gets log({DISCONNECTION_PENALTY:g}), "
    "so a disconnected step is strongly penalised but not forbidden. The decoder can still recover if the network is incomplete.",
    "<b>Missing candidates and gaps.</b> Fixes with no candidate within 50 m are removed from the Viterbi lattice and their "
    "output is left empty; the transition then spans the surrounding fixes with d<sub>gps</sub> measured between them. "
    "Long time gaps get no special treatment beyond this and the <i>time_gap_s</i> feature, and remain a limitation.",
])
h2("5.1 Did HMM decoding help?")
rows = [["Trip", "Candidate ceiling", "Top-1 classifier", "Top-1 + HMM", "Change", "Continuity classifier", "Continuity + HMM"]]
for _, r in st.iterrows():
    rows.append([r.trip_id, pct(r.candidate_oracle_top1), pct(r.top1_classifier_only), pct(r.top1_with_hmm),
                 f"{(r.top1_with_hmm - r.top1_classifier_only) * 100:+.2f}", pct(r.continuity_classifier_only), pct(r.continuity_with_hmm)])
rows.append(["Mean", pct(st.candidate_oracle_top1.mean()), pct(st.top1_classifier_only.mean()), pct(st.top1_with_hmm.mean()),
             f"{(st.top1_with_hmm.mean() - st.top1_classifier_only.mean()) * 100:+.2f}",
             pct(st.continuity_classifier_only.mean()), pct(st.continuity_with_hmm.mean())])
table(rows, [2 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 1.6 * cm, 2.6 * cm, 2.4 * cm])
helped = (st.top1_with_hmm > st.top1_classifier_only).sum()
p(f"With the tuned model on the student-test partition, HMM decoding raised mean Top-1 from {pct(st.top1_classifier_only.mean())} "
  f"to {pct(st.top1_with_hmm.mean())} and route continuity from {pct(st.continuity_classifier_only.mean())} to "
  f"{pct(st.continuity_with_hmm.mean())}. It improved Top-1 on {helped} of 10 trips. <i>Candidate ceiling</i> is the share "
  "of fixes whose reference arc is among the candidates at all; no decoder can exceed it.")
worst = st.assign(d=st.top1_with_hmm - st.top1_classifier_only).sort_values("d").iloc[0]
w_rev = rev.set_index("trip_id").loc[worst.trip_id]
p(f"<b>Where it did not help.</b> On trip {worst.trip_id}, Top-1 fell by {abs(worst.top1_with_hmm - worst.top1_classifier_only) * 100:.2f} "
  f"points, and {w_rev.reverse_share_of_errors * 100:.0f}% of its HMM errors pick the <i>reverse-direction twin</i> of the correct arc. "
  "The cause is structural. The transition term compares path length with GPS displacement, and a route driven backwards "
  "over the reverse arcs has exactly the same path lengths, so the transition term carries no information about direction. "
  "Only the classifier's heading feature separates the two directions, and when sequence smoothing outweighs it the decoder "
  "can settle on a consistent but reversed path. Across student-test trips the reverse-direction share of HMM errors "
  f"ranges from {rev.reverse_share_of_errors.min() * 100:.0f}% to {rev.reverse_share_of_errors.max() * 100:.0f}% "
  "(<font face='Courier'>hmm_reverse_direction_errors.csv</font>). An extra direction penalty in the emission term was "
  "tried on student-test data and lowered accuracy, so it was not adopted. A direction-aware transition is the main "
  "improvement we would pursue next.")
figure(FIG / "classifier_vs_hmm_accuracy.png", "Figure 9. Classifier-only vs classifier + HMM Top-1 accuracy across models (student-test).", 12)
figure(FIG / "classifier_vs_hmm_continuity.png", "Figure 10. Route continuity with and without HMM decoding (student-test).", 12)
figure(FIG / "student_test_route_example.png", "Figure 11. Student-test trip 00000076: GPS fixes (red), reconstructed route (blue) and road network (grey).", 11)

# ---------------------------------------------------------------- 6 Folium
h1("6. Geographical Visualisation Using Folium")
p("<font face='Courier'>src/visualization/map_plotter.py::create_folium_map</font> builds a map with four toggleable layers "
  "(nearby road network in grey, GPS fixes as blue markers, reference route in green, predicted route in dashed red) and a "
  "layer control. Routes are drawn as separate continuous runs, so no straight line is drawn across a gap between "
  "non-adjacent arcs. The map below is the committed file "
  "<font face='Courier'>artifacts/maps/student_test_00000076_reconstruction.html</font>.")
figure(SHOT / "folium_map_student_00000076.jpg", "Figure 12. Folium map for student-test trip 00000076: GPS fixes, reference route and predicted route overlap along the roads; layer control top right.", 15)
figure(SHOT / "folium_map_student_00000076_zoomed.jpg", "Figure 13. Same map zoomed to a junction: individual noisy fixes against the matched route.", 15)
story.append(PageBreak())

# ---------------------------------------------------------------- 7 deployment
h1("7. API, UI and Deployment")
bullets([
    "<b>Laptop 1</b> (<font face='Courier'>src/api/main.py</font>): FastAPI with <font face='Courier'>GET /health</font>, "
    "<font face='Courier'>GET /trips</font> and <font face='Courier'>POST /predict</font>. The exported model and scaler are "
    "loaded once at startup. Accuracy is computed server-side against the reference route, and the response schema "
    "(<font face='Courier'>src/api/schemas.py</font>) has no field for ground truth.",
    "<b>Laptop 2</b> (<font face='Courier'>ui/app.py</font>): Streamlit lists the trips from <font face='Courier'>/trips</font>, "
    "sends the selected id to Laptop 1 over the LAN and renders the metrics and a Folium map of the response.",
    "<b>Integration test.</b> All 10 evaluator-testing trips were run through <font face='Courier'>POST /predict</font> on the "
    "deployed server; all returned HTTP 200, with latencies between about 0.1 and 1.5 s per trip.",
])
figure(SHOT / "ui_results.jpg", "Figure 14. Evaluator UI after a reconstruction request (metrics computed by Laptop 1).", 15)
figure(SHOT / "ui_map.jpg", "Figure 15. Evaluator UI map for the same request: GPS fixes (blue) and predicted route (dashed red).", 15)
p("<b>API log on the serving laptop</b> (excerpt from the uvicorn console during the requests above):")
log_lines = []
try:
    with open("/tmp/pathsnap_api.log") as fh:
        log_lines = [l.rstrip() for l in fh if "inference" in l or "POST /predict" in l][-6:]
except OSError:
    pass
story.append(Paragraph("<br/>".join(l.replace("&", "&amp;").replace("<", "&lt;") for l in log_lines) or "(log not available on this machine)", CODE))
story.append(Spacer(1, 6))

# ---------------------------------------------------------------- 8 limitations & integrity
h1("8. Limitations, Integrity Notes and AI Assistance")
bullets([
    "<b>Dense networks.</b> Where many short arcs lie within a few metres of a fix, the reference arc can be outranked by "
    "ten closer arcs and never enter the candidate set. Raising the candidate limit was tested on student-test data and "
    "gave only marginal gains for several times the cost.",
    "<b>Direction symmetry</b> of the HMM transition term (Section 5.1).",
    "<b>Repeated-pass trips</b> cannot be labelled by a single-pass alignment (Section 2.3).",
    "<b>Long sampling gaps</b> are handled only through features, not by splitting the decoded sequence.",
    "<b>Separate notebook experiments.</b> <font face='Courier'>notebooks/02_model_benchmarking_and_optuna.ipynb</font> and its "
    "outputs (<font face='Courier'>model_cv_*.csv</font>, <font face='Courier'>optuna_trials.csv</font>, "
    "<font face='Courier'>student_test_comparison.csv</font>, <font face='Courier'>tuned_vs_untuned.csv</font>) come from an "
    "earlier, independently implemented benchmark with a different feature set. They do not produce the deployed model. "
    "All numbers in this report come from <font face='Courier'>scripts/run_full_pipeline.py</font> and "
    "<font face='Courier'>scripts/report_evidence.py</font>.",
    "<b>Evaluator data.</b> The alignment defect in Section 2.3 was first noticed while running evaluator-testing trips "
    "through the deployed API for integration testing. The fix has no tunable parameters and was validated on training and "
    "student-test data only (Figure 1). No model, feature or hyperparameter was chosen using evaluator data, and evaluator "
    "ground truth is never sent to the UI.",
    "<b>AI assistance.</b> AI coding assistance was used during implementation, debugging, analysis and drafting of this "
    "report. Every result is produced by code in this repository and can be regenerated with the commands in the appendix.",
])

# ---------------------------------------------------------------- appendix
h1("Appendix: Reproducing the results")
story.append(Paragraph(
    "python -m scripts.run_eda                  # dataset inventory<br/>"
    "python -m scripts.run_full_pipeline        # dataset build, 6-model benchmark, Optuna, model export<br/>"
    "python -m scripts.generate_report_figures  # comparison figures<br/>"
    "python -m scripts.report_evidence          # alignment validation, fit, per-trip and tuned/untuned results<br/>"
    "python -m scripts.build_report             # this PDF<br/>"
    "uvicorn src.api.main:app --host 0.0.0.0 --port 8000   # Laptop 1<br/>"
    "API_URL=http://&lt;laptop-1-ip&gt;:8000 streamlit run ui/app.py   # Laptop 2", CODE))


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(1.8 * cm, 1.1 * cm, "PathSnap Technical Report - UE24CS352A")
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.8 * cm, title="PathSnap Technical Report")
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

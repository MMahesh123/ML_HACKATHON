"""Generates the comparison plots required for the technical report, from the
actual experiment outputs in artifacts/results/ - never hand-drawn or
fabricated numbers.

Run with: python -m scripts.generate_report_figures
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import PROJECT_ROOT, RESULTS_DIR

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def _bar_chart(df: pd.DataFrame, column: str, title: str, ylabel: str, filename: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df["Model"], df[column], color="#2a6f97")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = RESULTS_DIR / "model_comparison.csv"
    if not comparison_path.exists():
        print(f"No {comparison_path} found - run `python -m src.models.train` first.")
        return

    df = pd.read_csv(comparison_path)

    _bar_chart(df, "CV_Macro_F1", "Cross-Validation Macro F1-score by Model", "Macro F1", "cv_macro_f1_comparison.png")
    _bar_chart(df, "StudentTest_RowAccuracy", "Student-Test Row-level Top-1 Accuracy", "Accuracy", "student_test_accuracy_comparison.png")
    _bar_chart(df, "StudentTest_RowMacroF1", "Student-Test Macro F1-score", "Macro F1", "student_test_f1_comparison.png")
    if df["StudentTest_LogLoss"].notna().any():
        _bar_chart(df, "StudentTest_LogLoss", "Student-Test Log Loss (lower is better)", "Log Loss", "log_loss_comparison.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(df))
    ax.bar([i - 0.2 for i in x], df["StudentTest_Top1Accuracy_ClassifierOnly"], width=0.4, label="Classifier only", color="#a8dadc")
    ax.bar([i + 0.2 for i in x], df["StudentTest_Top1Accuracy_WithHMM"], width=0.4, label="Classifier + HMM", color="#1d3557")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["Model"], rotation=20, ha="right")
    ax.set_ylabel("Sequence-level Top-1 Accuracy")
    ax.set_title("Classifier-only vs. Classifier + HMM/Viterbi (student-test trips)")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "classifier_vs_hmm_accuracy.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([i - 0.2 for i in x], df["RouteContinuity_ClassifierOnly"], width=0.4, label="Classifier only", color="#a8dadc")
    ax.bar([i + 0.2 for i in x], df["RouteContinuity_WithHMM"], width=0.4, label="Classifier + HMM", color="#1d3557")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["Model"], rotation=20, ha="right")
    ax.set_ylabel("Route Continuity")
    ax.set_title("Route Continuity: Classifier-only vs. Classifier + HMM/Viterbi")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "classifier_vs_hmm_continuity.png", dpi=150)
    plt.close(fig)

    bo_path = RESULTS_DIR / "bayesian_optimisation.json"
    if bo_path.exists():
        with open(bo_path) as f:
            bo = json.load(f)
        trials_df = pd.DataFrame(bo["trials"])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(trials_df["number"], trials_df["value"], marker="o", markersize=3, color="#e63946")
        ax.axhline(bo["best_value_cv_macro_f1"], color="#1d3557", linestyle="--", label=f"Best: {bo['best_value_cv_macro_f1']:.4f}")
        ax.set_xlabel("Trial")
        ax.set_ylabel("CV Macro F1-score")
        ax.set_title(f"Bayesian Optimisation Trials ({bo['model_family']})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "bayesian_optimisation_trials.png", dpi=150)
        plt.close(fig)

    print(f"Figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()

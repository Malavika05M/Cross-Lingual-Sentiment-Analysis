"""Create a slide-ready summary plot for the final five-seed evaluation."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="research_outputs/regularized_smoke")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run_root = ROOT / args.root
    output = ROOT / args.output if args.output else run_root / "final_evaluation_plots.png"
    runs = [load_json(path) for path in sorted(run_root.glob("seed_*/metrics.json"))]
    if len(runs) != 5:
        raise SystemExit(f"Expected five completed runs; found {len(runs)}.")

    aggregate = load_json(run_root / "aggregate_metrics.json")
    original = load_json(ROOT / "results" / "yosm" / "cnn_bilstm_metrics.json")
    seed42 = next(run for run in runs if run["seed"] == 42)
    history = seed42["history"]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    figure.suptitle("YorubaLiteSent: Final Five-Seed Evaluation", fontsize=15, fontweight="bold")

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    axis = axes[0]
    axis.plot(epochs, history["train_loss"], marker="o", markersize=3, label="Training loss", color="#1f77b4")
    axis.plot(epochs, history["validation_loss"], marker="o", markersize=3, label="Validation loss", color="#ff7f0e")
    axis.axvline(seed42["best_epoch"], color="#555555", linestyle="--", linewidth=1,
                 label=f"Selected epoch ({seed42['best_epoch']})")
    axis.set_title("Seed 42 learning curves")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Binary cross-entropy loss")
    axis.legend(fontsize=8)

    axis = axes[1]
    seeds = [run["seed"] for run in runs]
    f1_values = np.array([run["test"]["f1"] * 100 for run in runs])
    accuracy_values = np.array([run["test"]["accuracy"] * 100 for run in runs])
    positions = np.arange(len(seeds))
    axis.scatter(positions - 0.08, accuracy_values, s=55, color="#1f77b4", label="Accuracy")
    axis.scatter(positions + 0.08, f1_values, s=55, color="#2ca02c", label="F1")
    axis.axhline(aggregate["accuracy"]["mean"] * 100, color="#1f77b4", linestyle="--", linewidth=1)
    axis.axhline(aggregate["f1"]["mean"] * 100, color="#2ca02c", linestyle="--", linewidth=1)
    axis.set_xticks(positions, [str(seed) for seed in seeds])
    axis.set_ylim(75, 90)
    axis.set_title("Held-out test results by seed")
    axis.set_xlabel("Random seed")
    axis.set_ylabel("Score (%)")
    axis.legend(fontsize=8)

    axis = axes[2]
    labels = ["Previous\nCNN–BiLSTM–Attention", "YorubaLiteSent\n(five-seed mean)"]
    accuracies = [original["test"]["accuracy"] * 100, aggregate["accuracy"]["mean"] * 100]
    f1_scores = [original["test"]["f1"] * 100, aggregate["f1"]["mean"] * 100]
    x = np.arange(len(labels))
    width = 0.33
    bars_a = axis.bar(x - width / 2, accuracies, width, label="Accuracy", color="#1f77b4")
    bars_b = axis.bar(x + width / 2, f1_scores, width, label="F1", color="#2ca02c")
    for bar in [*bars_a, *bars_b]:
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.25,
                  f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    axis.set_xticks(x, labels)
    axis.set_ylim(75, 90)
    axis.set_title("Comparison on YOSM test set")
    axis.set_ylabel("Score (%)")
    axis.legend(fontsize=8)

    parameter_count = runs[0]["params"]
    figure.text(0.5, 0.01,
                f"YorubaLiteSent: {parameter_count / 1_000:.0f}K parameters; "
                f"five-seed F1 = {aggregate['f1']['mean'] * 100:.2f}% ± {aggregate['f1']['std'] * 100:.2f}%. "
                "The previous hybrid used approximately 3.52M parameters.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.06, 1, 0.92))
    figure.savefig(output, dpi=220, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()

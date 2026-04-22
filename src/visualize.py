"""
Visualization utilities for adversarial robustness and uncertainty analysis.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_robustness_curves(comparison_results, save_path="results/robustness_curves.png"):
    """Plot accuracy vs. epsilon curves for each model under FGSM and PGD."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for attack_type, ax in zip(["fgsm", "pgd"], axes):
        for model_name, results in comparison_results.items():
            eps_keys = sorted(results[attack_type].keys(), key=float)
            epsilons = [float(e) for e in eps_keys]
            accuracies = [results[attack_type][e]["accuracy"] for e in eps_keys]

            # Prepend clean accuracy at epsilon=0
            epsilons = [0.0] + epsilons
            accuracies = [results["clean"]["accuracy"]] + accuracies

            ax.plot(epsilons, accuracies, marker="o", linewidth=2, label=model_name)

        ax.set_xlabel("Epsilon (perturbation budget)", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title(f"{attack_type.upper()} Attack", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Robustness curves saved to {save_path}")


def plot_per_class_accuracy(results, model_name, attack_type, epsilon, save_path=None):
    """Bar chart of per-class accuracy under a specific attack."""
    if attack_type == "clean":
        data = results["clean"]["per_class_accuracy"]
    else:
        data = results[attack_type][str(epsilon)]["per_class_accuracy"]

    classes = list(data.keys())
    accuracies = list(data.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlGn(np.array(accuracies) / 100)
    bars = ax.bar(classes, accuracies, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    title = f"{model_name} - Per-Class Accuracy"
    if attack_type != "clean":
        title += f" ({attack_type.upper()}, eps={epsilon})"
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_reliability_diagram(calibration_data, title="Reliability Diagram", save_path=None):
    """
    Plot a reliability diagram (calibration plot) with ECE annotation.
    """
    bin_accs = calibration_data["bin_accuracies"]
    bin_confs = calibration_data["bin_confidences"]
    bin_counts = calibration_data["bin_counts"]
    ece = calibration_data["ece"]

    num_bins = len(bin_accs)
    bin_width = 1.0 / num_bins
    positions = np.arange(num_bins) * bin_width + bin_width / 2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Reliability diagram
    ax1.bar(positions, bin_accs, width=bin_width * 0.8, alpha=0.7,
            color="steelblue", edgecolor="black", linewidth=0.5, label="Accuracy")
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")

    # Gap bars
    for pos, acc, conf in zip(positions, bin_accs, bin_confs):
        if conf > 0:
            gap = conf - acc
            color = "salmon" if gap > 0 else "lightgreen"
            ax1.bar(pos, abs(gap), bottom=min(acc, conf), width=bin_width * 0.8,
                    alpha=0.3, color=color, edgecolor="none")

    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title(f"{title}\nECE = {ece:.4f}", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Histogram of predictions per bin
    ax2.bar(positions, bin_counts, width=bin_width * 0.8, color="gray",
            edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Confidence", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Reliability diagram saved to {save_path}")


def plot_uncertainty_histogram(uncertainty_results, save_path="results/uncertainty_hist.png"):
    """Plot entropy distributions for correct vs incorrect predictions."""
    # We need raw data for this - use calibration bins as proxy
    fig, ax = plt.subplots(figsize=(8, 5))

    stats = [
        ("Correct Predictions", uncertainty_results["correct_pred_mean_entropy"], "green"),
        ("Incorrect Predictions", uncertainty_results["incorrect_pred_mean_entropy"], "red"),
    ]

    x = np.arange(len(stats))
    colors = [s[2] for s in stats]
    values = [s[1] for s in stats]
    labels = [s[0] for s in stats]

    bars = ax.bar(x, values, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Mean Predictive Entropy", fontsize=12)
    ax.set_title("Uncertainty: Correct vs Incorrect Predictions", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Uncertainty histogram saved to {save_path}")


def generate_all_plots(comparison_results, uncertainty_results, output_dir="results"):
    """Generate all visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Robustness curves
    plot_robustness_curves(comparison_results, save_path=output_dir / "robustness_curves.png")

    # Per-class accuracy for each model (clean and strongest attack)
    for model_name, results in comparison_results.items():
        safe_name = model_name.replace(" ", "_").lower()
        plot_per_class_accuracy(
            results, model_name, "clean", None,
            save_path=output_dir / f"{safe_name}_per_class_clean.png"
        )
        # Strongest PGD attack
        pgd_epsilons = sorted(results["pgd"].keys(), key=float)
        if pgd_epsilons:
            strongest_eps = pgd_epsilons[-1]
            plot_per_class_accuracy(
                results, model_name, "pgd", float(strongest_eps),
                save_path=output_dir / f"{safe_name}_per_class_pgd_{strongest_eps}.png"
            )

    # Uncertainty plots
    if uncertainty_results:
        if "calibration" in uncertainty_results:
            plot_reliability_diagram(
                uncertainty_results["calibration"],
                title="MC Dropout Calibration",
                save_path=output_dir / "reliability_diagram.png"
            )
        plot_uncertainty_histogram(uncertainty_results, save_path=output_dir / "uncertainty_hist.png")

    print(f"\nAll plots saved to {output_dir}/")

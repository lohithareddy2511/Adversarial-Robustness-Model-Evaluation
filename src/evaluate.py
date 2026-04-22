"""
Evaluation pipeline: systematic comparison of model robustness across attack
scenarios and model iterations.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from .attacks import fgsm_attack, pgd_attack


def evaluate_under_attack(model, loader, device, attack_type, attack_params):
    """
    Evaluate model accuracy under a specific adversarial attack.

    Args:
        model: Target model (set to eval mode internally).
        loader: DataLoader with test data.
        device: torch device.
        attack_type: "clean", "fgsm", or "pgd".
        attack_params: Dict of attack hyperparameters.

    Returns:
        Dict with accuracy, avg_loss, per-class accuracy.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    total_loss = 0.0
    class_correct = [0] * 10
    class_total = [0] * 10

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if attack_type == "fgsm":
            adv_images = fgsm_attack(model, images, labels, **attack_params)
        elif attack_type == "pgd":
            adv_images = pgd_attack(model, images, labels, **attack_params)
        else:
            adv_images = images

        with torch.no_grad():
            outputs = model(adv_images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        for i in range(labels.size(0)):
            label = labels[i].item()
            class_total[label] += 1
            if predicted[i].item() == label:
                class_correct[label] += 1

    cifar10_classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]

    per_class_acc = {}
    for i in range(10):
        if class_total[i] > 0:
            per_class_acc[cifar10_classes[i]] = 100.0 * class_correct[i] / class_total[i]

    return {
        "accuracy": 100.0 * correct / total,
        "avg_loss": total_loss / total,
        "per_class_accuracy": per_class_acc,
    }


def run_robustness_evaluation(model, test_loader, device, epsilon_values=None):
    """
    Run a full robustness evaluation across multiple attack types and epsilon values.

    Returns a structured results dict.
    """
    if epsilon_values is None:
        epsilon_values = [0.01, 0.02, 0.04, 0.08, 0.1]

    results = {}

    # Clean accuracy
    print("Evaluating on clean data...")
    results["clean"] = evaluate_under_attack(model, test_loader, device, "clean", {})
    print(f"  Clean accuracy: {results['clean']['accuracy']:.2f}%")

    # FGSM at various epsilon
    results["fgsm"] = {}
    for eps in epsilon_values:
        print(f"Evaluating FGSM (eps={eps:.3f})...")
        results["fgsm"][str(eps)] = evaluate_under_attack(
            model, test_loader, device, "fgsm", {"epsilon": eps}
        )
        print(f"  FGSM (eps={eps:.3f}) accuracy: {results['fgsm'][str(eps)]['accuracy']:.2f}%")

    # PGD at various epsilon
    results["pgd"] = {}
    for eps in epsilon_values:
        alpha = eps / 4
        print(f"Evaluating PGD (eps={eps:.3f}, alpha={alpha:.4f}, steps=10)...")
        results["pgd"][str(eps)] = evaluate_under_attack(
            model, test_loader, device, "pgd",
            {"epsilon": eps, "alpha": alpha, "num_steps": 10}
        )
        print(f"  PGD (eps={eps:.3f}) accuracy: {results['pgd'][str(eps)]['accuracy']:.2f}%")

    return results


def compare_models(models_dict, test_loader, device, epsilon_values=None):
    """
    Compare multiple models across attack scenarios.

    Args:
        models_dict: Dict mapping model_name -> model instance.
        test_loader: Test DataLoader.
        device: torch device.
        epsilon_values: List of epsilon values to test.

    Returns:
        Comparison results dict: model_name -> robustness_results.
    """
    comparison = {}
    for name, model in models_dict.items():
        print(f"\n{'='*60}")
        print(f"Evaluating model: {name}")
        print(f"{'='*60}")
        comparison[name] = run_robustness_evaluation(model, test_loader, device, epsilon_values)

    return comparison


def generate_comparison_summary(comparison_results):
    """
    Generate a human-readable summary table from comparison results.

    Returns a formatted string.
    """
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("ADVERSARIAL ROBUSTNESS COMPARISON SUMMARY")
    lines.append("=" * 80)

    # Header
    model_names = list(comparison_results.keys())
    header = f"{'Attack':<25}" + "".join(f"{name:>18}" for name in model_names)
    lines.append(header)
    lines.append("-" * len(header))

    # Clean accuracy
    row = f"{'Clean':<25}"
    for name in model_names:
        acc = comparison_results[name]["clean"]["accuracy"]
        row += f"{acc:>17.2f}%"
    lines.append(row)

    # FGSM results
    eps_keys = list(comparison_results[model_names[0]]["fgsm"].keys())
    for eps_key in eps_keys:
        row = f"{'FGSM (eps=' + eps_key + ')':<25}"
        for name in model_names:
            acc = comparison_results[name]["fgsm"][eps_key]["accuracy"]
            row += f"{acc:>17.2f}%"
        lines.append(row)

    # PGD results
    eps_keys = list(comparison_results[model_names[0]]["pgd"].keys())
    for eps_key in eps_keys:
        row = f"{'PGD (eps=' + eps_key + ')':<25}"
        for name in model_names:
            acc = comparison_results[name]["pgd"][eps_key]["accuracy"]
            row += f"{acc:>17.2f}%"
        lines.append(row)

    lines.append("=" * 80)
    return "\n".join(lines)


def save_results(results, path):
    """Save evaluation results to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"Results saved to {path}")

"""
Main orchestration script for the Adversarial Robustness & Model Evaluation project.

Runs the full pipeline:
  1. Train a standard (baseline) model on CIFAR-10
  2. Train an adversarially robust model (PGD adversarial training)
  3. Evaluate both models under FGSM and PGD attacks at various epsilon values
  4. Run uncertainty estimation (MC Dropout + Temperature Scaling)
  5. Generate comparison tables and visualization plots
"""

import argparse
import json
from pathlib import Path

import torch

from src.model import RobustCNN
from src.train import get_dataloaders, train_model
from src.attacks import create_pgd_attack_fn
from src.evaluate import compare_models, generate_comparison_summary, save_results
from src.uncertainty import TemperatureScaling, evaluate_uncertainty
from src.visualize import generate_all_plots


def parse_args():
    parser = argparse.ArgumentParser(description="Adversarial Robustness & Model Evaluation")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--adv-epsilon", type=float, default=0.03, help="Epsilon for adversarial training")
    parser.add_argument("--mc-passes", type=int, default=30, help="MC Dropout forward passes")
    parser.add_argument("--skip-training", action="store_true", help="Skip training and load saved models")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detected if not set)")
    return parser.parse_args()


def get_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")

    # ---- Data ----
    print("\n[1/5] Loading CIFAR-10 data...")
    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)

    # ---- Model Training ----
    baseline_path = models_dir / "baseline.pt"
    advtrained_path = models_dir / "adv_trained.pt"

    if args.skip_training:
        print("\n[2/5] Loading pre-trained models...")
        baseline_model = RobustCNN(mc_dropout=False, dropout_rate=0.1).to(device)
        baseline_model.load_state_dict(torch.load(baseline_path, map_location=device, weights_only=True))

        adv_model = RobustCNN(mc_dropout=False, dropout_rate=0.1).to(device)
        adv_model.load_state_dict(torch.load(advtrained_path, map_location=device, weights_only=True))
    else:
        # Train baseline model
        print("\n[2/5] Training models...")
        print("\n--- Training Baseline Model ---")
        baseline_model = RobustCNN(mc_dropout=False, dropout_rate=0.1)
        baseline_history = train_model(
            baseline_model, train_loader, test_loader, device,
            epochs=args.epochs, lr=args.lr, save_path=str(baseline_path)
        )

        # Train adversarially robust model with PGD
        print("\n--- Training Adversarially Robust Model (PGD) ---")
        adv_model = RobustCNN(mc_dropout=False, dropout_rate=0.1)
        pgd_train_fn = create_pgd_attack_fn(
            epsilon=args.adv_epsilon,
            alpha=args.adv_epsilon / 4,
            num_steps=7,
        )
        adv_history = train_model(
            adv_model, train_loader, test_loader, device,
            epochs=args.epochs, lr=args.lr, attack_fn=pgd_train_fn,
            save_path=str(advtrained_path)
        )

        # Save training histories
        with open(output_dir / "baseline_history.json", "w") as f:
            json.dump(baseline_history, f, indent=2)
        with open(output_dir / "advtrained_history.json", "w") as f:
            json.dump(adv_history, f, indent=2)

    # ---- Robustness Evaluation ----
    print("\n[3/5] Evaluating adversarial robustness...")
    epsilon_values = [0.01, 0.02, 0.04, 0.08, 0.1]

    models_dict = {
        "Baseline": baseline_model,
        "PGD-Trained": adv_model,
    }

    comparison_results = compare_models(models_dict, test_loader, device, epsilon_values)
    summary = generate_comparison_summary(comparison_results)
    print(summary)

    save_results(comparison_results, output_dir / "robustness_results.json")
    with open(output_dir / "robustness_summary.txt", "w") as f:
        f.write(summary)

    # ---- Uncertainty Estimation ----
    print("\n[4/5] Running uncertainty estimation...")

    # Temperature Scaling on baseline model
    print("\n--- Temperature Scaling (Baseline) ---")
    temp_model = TemperatureScaling(baseline_model).to(device)
    temp_model.calibrate(test_loader, device)

    # MC Dropout on baseline model
    print("\n--- MC Dropout Uncertainty (Baseline) ---")
    mc_model = RobustCNN(mc_dropout=True, dropout_rate=0.1).to(device)
    mc_model.load_state_dict(baseline_model.state_dict())
    uncertainty_results = evaluate_uncertainty(mc_model, test_loader, device, num_mc_passes=args.mc_passes)
    save_results(uncertainty_results, output_dir / "uncertainty_results.json")

    # ---- Visualization ----
    print("\n[5/5] Generating visualizations...")
    generate_all_plots(comparison_results, uncertainty_results, output_dir=str(output_dir))

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"All results and plots saved to: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()

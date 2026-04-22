"""
Uncertainty estimation techniques: Temperature Scaling and MC Dropout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader


class TemperatureScaling(nn.Module):
    """
    Post-hoc temperature scaling for confidence calibration (Guo et al., 2017).

    Learns a single scalar temperature parameter on a validation set to
    calibrate the softmax outputs of a pre-trained model.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature

    def calibrate(self, val_loader, device, lr=0.01, max_iter=100):
        """
        Optimize the temperature parameter on a validation set using NLL loss.
        """
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        # Collect all logits and labels
        logits_list = []
        labels_list = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                logits = self.model(images)
                logits_list.append(logits.cpu())
                labels_list.append(labels)

        logits_all = torch.cat(logits_list).to(device)
        labels_all = torch.cat(labels_list).to(device)

        # Optimize temperature
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled_logits = logits_all / self.temperature.to(device)
            loss = criterion(scaled_logits, labels_all)
            loss.backward()
            return loss

        optimizer.step(closure)
        print(f"Optimal temperature: {self.temperature.item():.4f}")
        return self.temperature.item()


def mc_dropout_predict(model, images, num_forward_passes=30):
    """
    Monte Carlo Dropout prediction (Gal & Ghahramani, 2016).

    Performs multiple stochastic forward passes with dropout enabled
    to estimate predictive uncertainty.

    Args:
        model: Model with mc_dropout enabled.
        images: Input tensor (B, C, H, W).
        num_forward_passes: Number of MC samples.

    Returns:
        mean_probs: Mean predicted probabilities (B, num_classes).
        predictive_entropy: Predictive entropy per sample (B,).
        mutual_information: Mutual information (epistemic uncertainty) per sample (B,).
    """
    model.eval()
    model.enable_mc_dropout()

    all_probs = []
    with torch.no_grad():
        for _ in range(num_forward_passes):
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.unsqueeze(0))

    # Stack: (T, B, C) where T = num_forward_passes
    all_probs = torch.cat(all_probs, dim=0)

    # Mean prediction
    mean_probs = all_probs.mean(dim=0)  # (B, C)

    # Predictive entropy: H[y | x, D] = -sum p*log(p)
    predictive_entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1)

    # Expected entropy: E[H[y | x, w]] = -1/T * sum_t sum_c p_tc * log(p_tc)
    expected_entropy = -(all_probs * torch.log(all_probs + 1e-10)).sum(dim=2).mean(dim=0)

    # Mutual information (epistemic uncertainty) = predictive_entropy - expected_entropy
    mutual_information = predictive_entropy - expected_entropy

    model.disable_mc_dropout()

    return mean_probs, predictive_entropy, mutual_information


def compute_calibration_metrics(probs, labels, num_bins=15):
    """
    Compute Expected Calibration Error (ECE) and reliability diagram data.

    Args:
        probs: Predicted probabilities (N, num_classes) as numpy array.
        labels: True labels (N,) as numpy array.
        num_bins: Number of bins for calibration.

    Returns:
        Dict with ECE, bin accuracies, bin confidences, bin counts.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    ece = 0.0
    for i in range(num_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        count = in_bin.sum()
        bin_counts.append(int(count))

        if count > 0:
            avg_acc = accuracies[in_bin].mean()
            avg_conf = confidences[in_bin].mean()
            bin_accs.append(float(avg_acc))
            bin_confs.append(float(avg_conf))
            ece += (count / len(labels)) * abs(avg_acc - avg_conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)

    return {
        "ece": float(ece),
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
        "bin_boundaries": bin_boundaries.tolist(),
    }


def evaluate_uncertainty(model, test_loader, device, num_mc_passes=30):
    """
    Full uncertainty evaluation: MC Dropout predictions + calibration metrics.

    Returns dict with predictions, entropy stats, and calibration metrics.
    """
    model.eval()
    model.enable_mc_dropout()

    all_mean_probs = []
    all_entropies = []
    all_mi = []
    all_labels = []
    all_correct = []

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        mean_probs, entropy, mi = mc_dropout_predict(model, images, num_mc_passes)

        predictions = mean_probs.argmax(dim=1)
        correct = predictions.eq(labels)

        all_mean_probs.append(mean_probs.cpu().numpy())
        all_entropies.append(entropy.cpu().numpy())
        all_mi.append(mi.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_correct.append(correct.cpu().numpy())

    all_mean_probs = np.concatenate(all_mean_probs)
    all_entropies = np.concatenate(all_entropies)
    all_mi = np.concatenate(all_mi)
    all_labels = np.concatenate(all_labels)
    all_correct = np.concatenate(all_correct)

    # Calibration
    calibration = compute_calibration_metrics(all_mean_probs, all_labels)

    # Uncertainty statistics for correct vs incorrect predictions
    correct_entropy = all_entropies[all_correct.astype(bool)]
    incorrect_entropy = all_entropies[~all_correct.astype(bool)]

    model.disable_mc_dropout()

    results = {
        "mc_dropout_accuracy": 100.0 * all_correct.mean(),
        "ece": calibration["ece"],
        "mean_predictive_entropy": float(all_entropies.mean()),
        "mean_mutual_information": float(all_mi.mean()),
        "correct_pred_mean_entropy": float(correct_entropy.mean()) if len(correct_entropy) > 0 else 0,
        "incorrect_pred_mean_entropy": float(incorrect_entropy.mean()) if len(incorrect_entropy) > 0 else 0,
        "calibration": calibration,
    }

    print(f"\nUncertainty Evaluation Results:")
    print(f"  MC Dropout Accuracy: {results['mc_dropout_accuracy']:.2f}%")
    print(f"  ECE: {results['ece']:.4f}")
    print(f"  Mean Predictive Entropy: {results['mean_predictive_entropy']:.4f}")
    print(f"  Mean Mutual Information: {results['mean_mutual_information']:.4f}")
    print(f"  Correct predictions - Mean Entropy: {results['correct_pred_mean_entropy']:.4f}")
    print(f"  Incorrect predictions - Mean Entropy: {results['incorrect_pred_mean_entropy']:.4f}")

    return results

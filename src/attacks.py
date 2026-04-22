"""
Adversarial attack implementations: FGSM and PGD.
"""

import torch
import torch.nn as nn


def fgsm_attack(model, images, labels, epsilon, criterion=None):
    """
    Fast Gradient Sign Method (Goodfellow et al., 2014).

    Generates adversarial examples by adding epsilon-scaled sign of the
    loss gradient to the input images.

    Args:
        model: Target neural network.
        images: Clean input images (B, C, H, W).
        labels: True labels (B,).
        epsilon: Perturbation magnitude.
        criterion: Loss function (defaults to CrossEntropyLoss).

    Returns:
        Adversarial images clamped to valid range.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()

    # Create perturbation
    perturbation = epsilon * images.grad.sign()
    adv_images = images + perturbation

    # Clamp to valid pixel range (assuming normalized input, we use a wide range)
    adv_images = adv_images.detach()
    return adv_images


def pgd_attack(model, images, labels, epsilon, alpha, num_steps, criterion=None, random_start=True):
    """
    Projected Gradient Descent (Madry et al., 2017).

    Iterative adversarial attack that applies FGSM repeatedly with a smaller
    step size and projects back into the epsilon-ball.

    Args:
        model: Target neural network.
        images: Clean input images (B, C, H, W).
        labels: True labels (B,).
        epsilon: Maximum perturbation magnitude (L-inf).
        alpha: Step size per iteration.
        num_steps: Number of PGD iterations.
        criterion: Loss function (defaults to CrossEntropyLoss).
        random_start: Whether to start from a random point in the epsilon-ball.

    Returns:
        Adversarial images clamped to valid range.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    original_images = images.clone().detach()
    adv_images = images.clone().detach()

    if random_start:
        # Start from a random point within the epsilon-ball
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
        adv_images = adv_images.detach()

    for _ in range(num_steps):
        adv_images.requires_grad_(True)
        outputs = model(adv_images)
        loss = criterion(outputs, labels)
        loss.backward()

        # FGSM step
        grad_sign = adv_images.grad.sign()
        adv_images = adv_images.detach() + alpha * grad_sign

        # Project back into epsilon-ball around original images
        delta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        adv_images = (original_images + delta).detach()

    return adv_images


def create_fgsm_attack_fn(epsilon):
    """Create an FGSM attack function with fixed epsilon for use in adversarial training."""
    def attack_fn(model, images, labels):
        return fgsm_attack(model, images, labels, epsilon)
    return attack_fn


def create_pgd_attack_fn(epsilon, alpha, num_steps):
    """Create a PGD attack function with fixed parameters for use in adversarial training."""
    def attack_fn(model, images, labels):
        return pgd_attack(model, images, labels, epsilon, alpha, num_steps)
    return attack_fn

# Adversarial Robustness & Model Evaluation

End-to-end pipeline for evaluating deep learning model robustness under adversarial attacks and estimating prediction uncertainty.

## Project Structure

```
├── main.py                  # Full pipeline orchestration
├── requirements.txt
├── src/
│   ├── model.py             # ResNet-style CNN with MC Dropout support
│   ├── train.py             # Training (standard + adversarial)
│   ├── attacks.py           # FGSM & PGD attack implementations
│   ├── evaluate.py          # Robustness evaluation pipeline
│   ├── uncertainty.py       # Temperature Scaling & MC Dropout
│   └── visualize.py         # Plots: robustness curves, reliability diagrams
└── results/                 # Generated outputs (models, plots, JSON)
```

## Components

### 1. Adversarial Attacks (`src/attacks.py`)
- **FGSM** (Goodfellow et al., 2014) — Single-step gradient sign attack
- **PGD** (Madry et al., 2017) — Iterative projected gradient descent with random start

### 2. Evaluation Pipeline (`src/evaluate.py`)
- Systematic accuracy comparison across attack types and epsilon values
- Per-class accuracy breakdown (all 10 CIFAR-10 classes)
- Multi-model comparison with formatted summary tables
- JSON export of all results

### 3. Uncertainty Estimation (`src/uncertainty.py`)
- **Temperature Scaling** (Guo et al., 2017) — Post-hoc logit calibration
- **MC Dropout** (Gal & Ghahramani, 2016) — Predictive entropy and mutual information
- Expected Calibration Error (ECE) computation
- Reliability diagram data generation

### 4. Visualization (`src/visualize.py`)
- Accuracy vs. epsilon robustness curves
- Per-class accuracy bar charts (color-coded)
- Reliability diagrams with ECE annotation
- Uncertainty histograms (correct vs. incorrect predictions)

## Dataset

This project uses **CIFAR-10**, a widely-used benchmark dataset for image classification:

- **10 classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **60,000 images** (50,000 train / 10,000 test), 32×32 RGB
- **Auto-downloads** on first run via `torchvision.datasets.CIFAR10` — no manual setup needed
- Raw data is stored locally in `data/` (~163MB) and excluded from the repository

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (trains 2 models, evaluates, generates plots)
python main.py --epochs 50

# Quick test with fewer epochs
python main.py --epochs 5

# Skip training (load saved models)
python main.py --skip-training

# Custom settings
python main.py --epochs 30 --batch-size 256 --adv-epsilon 0.03 --mc-passes 50
```

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 50 | Training epochs |
| `--batch-size` | 128 | Batch size |
| `--lr` | 0.1 | Learning rate |
| `--adv-epsilon` | 0.03 | Epsilon for adversarial training |
| `--mc-passes` | 30 | Number of MC Dropout forward passes |
| `--skip-training` | false | Load saved models instead of training |
| `--output-dir` | results | Output directory for results |
| `--device` | auto | Device (cuda/mps/cpu, auto-detected) |

## Pipeline Output

After running, the `results/` directory contains:

- `models/baseline.pt` — Standard trained model
- `models/adv_trained.pt` — PGD adversarially trained model
- `robustness_results.json` — Full evaluation data
- `robustness_summary.txt` — Formatted comparison table
- `uncertainty_results.json` — MC Dropout + calibration metrics
- `robustness_curves.png` — Accuracy vs. epsilon plots
- `reliability_diagram.png` — Calibration plot with ECE
- `uncertainty_hist.png` — Entropy comparison plot
- Per-class accuracy charts per model

## Key References

- Goodfellow et al., *Explaining and Harnessing Adversarial Examples* (2014) — FGSM
- Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks* (2017) — PGD
- Guo et al., *On Calibration of Modern Neural Networks* (2017) — Temperature Scaling
- Gal & Ghahramani, *Dropout as a Bayesian Approximation* (2016) — MC Dropout

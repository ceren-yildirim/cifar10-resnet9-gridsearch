# CS515 — Deep Learning Assignment 1

CIFAR-10 image classification using PyTorch. Covers manual hyperparameter search with ResNet9 and grid search with skorch/scikit-learn using SmallNet.

---

## Project Structure

```
├── models.py       # Network architectures: LeNet, SmallNet, ResNet9
├── data.py         # CIFAR-10 data loading and augmentation
├── utils.py        # Training, evaluation, and weight initialization helpers
├── train.py        # Part I — manual hyperparameter grid search (ResNet9)
└── gridsearch.py   # Part I — GridSearchCV with skorch (SmallNet)
```

---

## Setup

**Requirements**

```
torch
torchvision
scikit-learn
skorch
```

Install with:

```bash
pip install torch torchvision scikit-learn skorch
```

CIFAR-10 data is downloaded automatically to `./data/` on first run.

---

## Usage

### Manual Hyperparameter Search (`train.py`)

Trains ResNet9 across all combinations of learning rate, optimizer, weight initialization, and weight decay. Selects the best configuration by validation accuracy, then retrains and evaluates on the test set.

```bash
python train.py
```

**Search space:**

| Hyperparameter | Values |
|---|---|
| Learning rate | `0.001`, `0.01` |
| Optimizer | `SGD` (momentum=0.9), `Adam` |
| Weight initialization | `xavier`, `he` |
| Weight decay | `0`, `0.1`, `0.01`, `0.001` |

---

### Grid Search with skorch (`gridsearch.py`)

Uses `GridSearchCV` with 3-fold cross-validation to search over optimizer, learning rate, epoch count, and weight initialization for SmallNet.

```bash
python gridsearch.py
```

**Search space:**

| Hyperparameter | Values |
|---|---|
| Optimizer | `SGD`, `Adagrad`, `Adam` |
| Learning rate | `0.001`, `0.01` |
| Max epochs | `100`, `300`, `500` |
| Weight initialization | `uniform`, `normal`, `xavier_normal`, `xavier_uniform`, `kaiming_normal`, `kaiming_uniform` |

---

## Models

**LeNet** — Classic 2-conv + 3-FC architecture for CIFAR-10 (3-channel input).

**SmallNet** — 4-conv block network with configurable weight initialization, designed to work with skorch's `module__weight_init` parameter.

**ResNet9** — Lightweight residual network with 2 layers (`BasicBlock`), suitable for CIFAR-10.

---

## Data

All loaders use the `get_cifar10_loaders()` function from `data.py`.

- **Train / Val split:** 75% / 25% of the training set (reproducible via fixed seed)
- **Augmentation:** random horizontal flip, random rotation (±10°), random crop with padding
- **Normalization:** mean and std of `(0.5, 0.5, 0.5)` per channel

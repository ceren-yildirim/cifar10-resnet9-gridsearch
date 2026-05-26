import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV

from models import SmallNet
from data import get_cifar10_loaders

SEED = 27
torch.manual_seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 200

train_loader, _, test_loader = get_cifar10_loaders(batch_size=BATCH_SIZE)

# skorch expects numpy arrays or tensors, not DataLoader — pull one full batch
train_features, train_labels = next(iter(train_loader))
test_features, test_labels = next(iter(test_loader))

# ── Base model (skorch wrapper) ────────────────────────────────────────────────
model = NeuralNetClassifier(
    module=SmallNet,
    batch_size=8,
    iterator_train__shuffle=True,
    criterion=nn.CrossEntropyLoss,
    train_split=False,
    verbose=False,
)

# ── Hyperparameter grid ────────────────────────────────────────────────────────
param_grid = {
    'optimizer': [optim.SGD, optim.Adagrad, optim.Adam],
    'optimizer__lr': [0.001, 0.01],
    'max_epochs': [100, 300, 500],
    'module__weight_init': [
        init.uniform_,
        init.normal_,
        init.xavier_normal_,
        init.xavier_uniform_,
        init.kaiming_normal_,
        init.kaiming_uniform_,
    ],
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    refit=False,
    cv=3,
    scoring='accuracy',
    verbose=2,
)
grid_result = grid.fit(train_features, train_labels)

print("Best: %.6f using %s" % (grid_result.best_score_, grid_result.best_params_))
best_params = grid_result.best_params_

# ── Retrain best model and evaluate on test set ────────────────────────────────
best_model = NeuralNetClassifier(
    module=SmallNet,
    max_epochs=best_params['max_epochs'],
    optimizer=best_params['optimizer'],
    optimizer__lr=best_params['optimizer__lr'],
    module__weight_init=best_params['module__weight_init'],
    batch_size=8,
    iterator_train__shuffle=True,
    criterion=nn.CrossEntropyLoss,
    train_split=False,
    verbose=False,
)

best_model.fit(train_features, train_labels)
test_accuracy = best_model.score(test_features, test_labels)
print(f"Test accuracy: {test_accuracy:.4f}")

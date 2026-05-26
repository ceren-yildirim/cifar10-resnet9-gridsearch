import itertools
import random
import torch
import torch.optim as optim

from models import ResNet9
from data import get_cifar10_loaders
from utils import train_epoch, evaluate, initialize_weights

SEED = 27
torch.manual_seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
EPOCHS = 20

train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=BATCH_SIZE)

# ── Hyperparameter grid ────────────────────────────────────────────────────────
learning_rates = [0.001, 0.01]
optimizer_classes = [optim.SGD, optim.Adam]
weight_inits = ['xavier', 'he']
weight_decays = [0, 0.1, 0.01, 0.001]

best_val_accuracy = 0.0
best_hyperparameters = {}

for lr, opt_cls, w_init, wd in itertools.product(learning_rates, optimizer_classes, weight_inits, weight_decays):
    print(f"lr={lr}  optimizer={opt_cls.__name__}  weight_init={w_init}  weight_decay={wd}")

    model = ResNet9().to(device)
    initialize_weights(model, w_init)

    if opt_cls == optim.SGD:
        optimizer = opt_cls(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        optimizer = opt_cls(model.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(EPOCHS):
        train_epoch(model, optimizer, train_loader, device, epoch)

    accuracy = evaluate(model, val_loader, device)
    print(f"  val accuracy: {accuracy:.2f}%\n")

    if accuracy > best_val_accuracy:
        best_val_accuracy = accuracy
        best_hyperparameters = {
            'learning_rate': lr,
            'optimizer': opt_cls,
            'weight_init': w_init,
            'weight_decay': wd,
        }

print(f"Best hyperparameters: {best_hyperparameters}")
print(f"Best validation accuracy: {best_val_accuracy:.2f}%\n")

# ── Train final model with best hyperparameters ────────────────────────────────
final_model = ResNet9().to(device)
initialize_weights(final_model, best_hyperparameters['weight_init'])

opt_cls = best_hyperparameters['optimizer']
if opt_cls == optim.SGD:
    optimizer = opt_cls(
        final_model.parameters(),
        lr=best_hyperparameters['learning_rate'],
        momentum=0.9,
        weight_decay=best_hyperparameters['weight_decay'],
    )
else:
    optimizer = opt_cls(
        final_model.parameters(),
        lr=best_hyperparameters['learning_rate'],
        weight_decay=best_hyperparameters['weight_decay'],
    )

for epoch in range(EPOCHS):
    train_epoch(final_model, optimizer, train_loader, device, epoch)

test_accuracy = evaluate(final_model, test_loader, device)
print(f"Test accuracy: {test_accuracy:.2f}%")

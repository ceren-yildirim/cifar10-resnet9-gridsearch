import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def train_epoch(model, optimizer, train_loader, device, epoch=None):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(data), target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 2000 == 1999:
            ep_str = f"epoch {epoch + 1}, " if epoch is not None else ""
            print(f"[{ep_str}batch {batch_idx + 1}] loss: {running_loss / 2000:.3f}")
            running_loss = 0.0


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            _, predicted = torch.max(model(data), 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100.0 * correct / total


def initialize_weights(model, scheme):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if scheme == 'xavier':
                init.xavier_uniform_(m.weight)
            elif scheme == 'he':
                init.kaiming_uniform_(m.weight)
            elif scheme == 'random':
                init.uniform_(m.weight)

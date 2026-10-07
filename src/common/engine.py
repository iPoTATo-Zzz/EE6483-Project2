"""Training and evaluation with fixed backbone state and sample-weighted metrics."""
import torch


def run_epoch(model, loader, device, optimizer=None):
    # eval() prevents running-stat updates in frozen BatchNorm layers.
    model.eval()
    if optimizer is not None:
        model.fc.train()
    total_loss = correct = count = 0
    predictions, targets = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        predicted = logits.argmax(1)
        n = len(labels)
        count += n
        total_loss += loss.item() * n
        correct += (predicted == labels).sum().item()
        predictions.extend(predicted.detach().cpu().tolist())
        targets.extend(labels.cpu().tolist())
    return {"loss": total_loss / count, "accuracy": correct / count, "count": count}, targets, predictions

"""Reload a baseline checkpoint and export validation evidence."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from src.common.data import CatDogDataset
from src.common.engine import run_epoch
from src.common.utils import ROOT, load_checkpoint, save_json
from src.models.resnet18 import baseline_transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    data = CatDogDataset(args.data_root, "val", baseline_transform())
    metrics, targets, predicted = run_epoch(model, DataLoader(data, batch_size=32), device)
    save_json(args.output / "metrics.json", {**metrics, "checkpoint": str(args.checkpoint.resolve()), "epoch": checkpoint["epoch"], "confusion_matrix": confusion_matrix(targets, predicted, labels=[0, 1]).tolist(), "classification_report": classification_report(targets, predicted, labels=[0, 1], target_names=["cat", "dog"], output_dict=True, zero_division=0)})
    save_json(args.output / "predictions.json", [{"path": str(p.relative_to(args.data_root)), "label": y, "prediction": prediction} for (p, _), y, prediction in zip(data.samples, targets, predicted)])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matrix = confusion_matrix(targets, predicted, labels=[0, 1])
    fig, ax = plt.subplots(); ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["cat", "dog"], yticklabels=["cat", "dog"], xlabel="Predicted", ylabel="True")
    for i in range(2):
        for j in range(2): ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.tight_layout(); fig.savefig(args.output / "confusion_matrix.png"); plt.close(fig)
    print(metrics)


if __name__ == "__main__":
    main()

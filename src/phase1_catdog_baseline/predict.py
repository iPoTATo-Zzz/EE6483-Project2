"""Export 500 test predictions in numeric ID order without overwriting files."""
import argparse
import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.common.data import CatDogDataset
from src.common.utils import ROOT, load_checkpoint
from src.models.resnet18 import baseline_transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(args.checkpoint, device); model.eval()
    data = CatDogDataset(args.data_root, "test", baseline_transform())
    rows = []
    with torch.inference_mode():
        for images, ids in DataLoader(data, batch_size=32):
            labels = model(images.to(device)).argmax(1).cpu().tolist()
            rows.extend(zip(ids.tolist(), labels))
    assert [i for i, _ in rows] == list(range(1, 501))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["id", "label"]); writer.writerows(rows)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()

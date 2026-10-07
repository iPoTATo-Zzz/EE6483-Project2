"""Run from repository root: python -m src.phase1_catdog_baseline.train."""
import argparse
import csv
import json
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.common.data import CatDogDataset, LABELS, TRAIN_EXCLUSIONS
from src.common.engine import run_epoch
from src.common.utils import ROOT, save_json, seed_everything
from src.models.resnet18 import build_model, baseline_transform


def main(stage="phase1_catdog_baseline", default_config=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config or ROOT / "configs/phase1_catdog_baseline/baseline.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train-per-class", type=int)
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
        parser.error("run-name must be a single directory name")
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    for key in ["epochs", "train_per_class"]:
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if cfg["epochs"] < 1 or cfg["batch_size"] < 1:
        parser.error("epochs and batch size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use the project GPU environment")
    audit_path = ROOT / "results/phase1_catdog_baseline/data_check.json"
    if not audit_path.exists():
        raise RuntimeError("Run check_data first; a data audit is required")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("passed", False):
        raise RuntimeError("Data audit has unresolved issues; resolve them before training")
    if audit.get("training_exclusions") != sorted(TRAIN_EXCLUSIONS):
        raise RuntimeError("Audit exclusion policy does not match the loader")
    seed_everything(cfg["seed"])
    device = torch.device("cuda")
    output = ROOT / "runs" / stage / args.run_name
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "config.json", cfg)
    save_json(output / "environment.json", {"torch": str(torch.__version__), "torchvision": __import__("torchvision").__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})
    transform = baseline_transform()
    if "augmentation" in cfg:
        from src.common.transforms import augmented_transform
        transform = augmented_transform(cfg["augmentation"])
    train = CatDogDataset(args.data_root, "train", transform, cfg["train_per_class"], cfg["seed"])
    val = CatDogDataset(args.data_root, "val", baseline_transform())
    save_json(output / "data_manifest.json", {"root": str(args.data_root.resolve()), "train": [str(p.relative_to(args.data_root)) for p, _ in train.samples], "val": [str(p.relative_to(args.data_root)) for p, _ in val.samples]})
    generator = torch.Generator().manual_seed(cfg["seed"])
    options = {"batch_size": cfg["batch_size"], "num_workers": cfg["num_workers"], "pin_memory": True}
    train_loader = DataLoader(train, shuffle=True, generator=generator, **options)
    val_loader = DataLoader(val, shuffle=False, **options)
    # Keep downloaded pretrained weights outside Git-tracked directories.
    torch.hub.set_dir(str(ROOT / "runs/pretrained_cache"))
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=cfg["learning_rate"])
    best_key = (-1.0, float("-inf"))
    records = []
    for epoch in range(1, cfg["epochs"] + 1):
        started = time.perf_counter()
        train_metrics, _, _ = run_epoch(model, train_loader, device, optimizer)
        val_metrics, _, _ = run_epoch(model, val_loader, device)
        torch.cuda.synchronize()
        record = {"epoch": epoch, "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"], "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"], "seconds": time.perf_counter() - started}
        records.append(record)
        with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(record)); writer.writeheader(); writer.writerows(records)
        key = (val_metrics["accuracy"], -val_metrics["loss"])
        if key > best_key:
            best_key = key
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "config": cfg, "label_mapping": LABELS, "val_metrics": val_metrics}, output / "best.pth")
            save_json(output / "best_metrics.json", {"epoch": epoch, **val_metrics})
        print(json.dumps(record), flush=True)
    save_json(output / "timing.json", {"epochs": len(records), "total_epoch_seconds": sum(x["seconds"] for x in records), "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20})


if __name__ == "__main__":
    main()

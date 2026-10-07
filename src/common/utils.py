"""Reproducible setup and portable JSON/checkpoint metadata."""
import json
import random
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_checkpoint(path, device):
    from src.models.resnet18 import build_model
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["label_mapping"] != {"cat": 0, "dog": 1}:
        raise ValueError("Unexpected checkpoint label mapping")
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint

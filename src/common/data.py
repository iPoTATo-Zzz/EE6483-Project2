"""Explicit labels and numerical test IDs; deterministic balanced subsets."""
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

LABELS = {"cat": 0, "dog": 1}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_EXCLUSIONS = {"train/cat/cat.2339.jpg"}


class CatDogDataset(Dataset):
    def __init__(self, root, split, transform, per_class=None, seed=42):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        rng = random.Random(seed)
        if split == "test":
            files = [p for p in (self.root / split).iterdir() if p.suffix.lower() in EXTENSIONS]
            ids = [int(p.stem) for p in files]
            if sorted(ids) != list(range(1, 501)):
                raise ValueError("Test filenames must provide unique IDs 1..500")
            self.samples = [(p, int(p.stem)) for p in sorted(files, key=lambda p: int(p.stem))]
        else:
            for name, label in LABELS.items():
                files = sorted(p for p in (self.root / split / name).iterdir() if p.suffix.lower() in EXTENSIONS)
                if split == "train":
                    files = [p for p in files if p.relative_to(self.root).as_posix() not in TRAIN_EXCLUSIONS]
                if per_class is not None:
                    if not 0 < per_class <= len(files):
                        raise ValueError("Invalid per-class subset size")
                    files = sorted(rng.sample(files, per_class))
                self.samples.extend((p, label) for p in files)
        if not self.samples:
            raise ValueError("Empty dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = self.transform(image.convert("RGB"))
        return image, target

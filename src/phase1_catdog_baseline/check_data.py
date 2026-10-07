"""Read every image, hash files and check test IDs without modifying data."""
import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
from src.common.data import EXTENSIONS, TRAIN_EXCLUSIONS
from src.common.utils import ROOT, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--output", type=Path, default=ROOT / "results/phase1_catdog_baseline/data_check.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new filename")
    hashes = defaultdict(list); counts = Counter(); bad = []; ids = []; dimensions = []; modes = Counter()
    for split in ["train", "val", "test"]:
        if not (args.data_root / split).is_dir():
            raise FileNotFoundError(args.data_root / split)
        files = sorted(p for p in (args.data_root / split).rglob("*") if p.suffix.lower() in EXTENSIONS)
        for index, path in enumerate(files):
            relative = path.relative_to(args.data_root).as_posix()
            counts[split + "/" + (path.parent.name if split != "test" else "unlabelled")] += 1
            try:
                with Image.open(path) as image:
                    image.load(); dimensions.append(image.size); modes[image.mode] += 1
                hashes[hashlib.sha256(path.read_bytes()).hexdigest()].append(relative)
                if split == "test": ids.append(int(path.stem))
            except Exception as error:
                bad.append({"path": relative, "error": str(error)})
            if (index + 1) % 1000 == 0:
                print(f"{split}: {index + 1}/{len(files)}", flush=True)
    duplicates = [v for v in hashes.values() if len(v) > 1]
    cross = [v for v in duplicates if len({x.split('/')[0] for x in v}) > 1]
    expected = {"train/cat": 10000, "train/dog": 10000, "val/cat": 2500, "val/dog": 2500, "test/unlabelled": 500}
    report = {"counts": dict(counts), "expected_counts_match": dict(counts) == expected, "bad_images": bad, "duplicate_groups": duplicates, "cross_split_duplicates": cross, "test_ids_complete": sorted(ids) == list(range(1, 501)), "image_modes": dict(modes), "width_range": [min(x[0] for x in dimensions), max(x[0] for x in dimensions)] if dimensions else None, "height_range": [min(x[1] for x in dimensions), max(x[1] for x in dimensions)] if dimensions else None, "limitations": "SHA-256 detects byte-identical files only, not all visually similar images."}
    unresolved = [group for group in cross if len([p for p in group if p not in TRAIN_EXCLUSIONS]) > 1]
    report["training_exclusions"] = sorted(TRAIN_EXCLUSIONS)
    report["unresolved_cross_split_duplicates"] = unresolved
    report["passed"] = report["expected_counts_match"] and not bad and not unresolved and report["test_ids_complete"]
    save_json(args.output, report)
    print(f"Saved {args.output}; passed={report['passed']}", flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""D2 entry point sharing the unchanged D1 optimizer and epoch engine."""
from src.common.utils import ROOT
from src.phase1_catdog_baseline.train import main

if __name__ == "__main__":
    main(stage="phase2_catdog_improvements", default_config=ROOT / "configs/phase2_catdog_improvements/augmentation.json")

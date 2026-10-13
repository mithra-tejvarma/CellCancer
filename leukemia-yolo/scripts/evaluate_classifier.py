from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classifier_config import classifier_config
from src.classifier_evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate best_model.pth on held-out fold_2 only")
    parser.add_argument("--config")
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    print(evaluate(classifier_config(args.config), args.checkpoint))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classifier_gradcam import generate_gradcam


def main():
    parser = argparse.ArgumentParser(description="Generate held-out test Grad-CAM examples")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(generate_gradcam(args.checkpoint, args.predictions, args.output))


if __name__ == "__main__":
    main()

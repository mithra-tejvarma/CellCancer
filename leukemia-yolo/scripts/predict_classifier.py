from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classifier_inference import predict_paths


def main():
    parser = argparse.ArgumentParser(description="Predict a C-NMC image, YOLO WBC crop, or image folder")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(predict_paths(args.checkpoint, args.input, args.output, args.device).to_string(index=False))


if __name__ == "__main__":
    main()

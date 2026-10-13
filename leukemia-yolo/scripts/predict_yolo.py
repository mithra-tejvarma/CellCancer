import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.inference import predict


def main():
    parser = argparse.ArgumentParser(description="Predict and optionally crop WBCs with a completed detector")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--image", type=Path, help="Optional custom microscope image")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--conf", type=float)
    parser.add_argument("--iou", type=float)
    parser.add_argument("--no-crops", action="store_true")
    args = parser.parse_args()
    print(predict(load_config(args.config), run=args.run, custom=args.image, limit=args.limit,
                  conf=args.conf, iou=args.iou, crop=False if args.no_crops else None))


if __name__ == "__main__":
    main()

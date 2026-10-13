import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate a completed YOLOv8n WBC run on test only")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run", type=Path, help="Completed main run; default latest completed run")
    args = parser.parse_args()
    print(evaluate(load_config(args.config), args.run))


if __name__ == "__main__":
    main()

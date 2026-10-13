"""Run as a child process on Windows to satisfy multiprocessing spawn rules."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.training import run_training


def main():
    parser = argparse.ArgumentParser(description="CUDA-only YOLOv8n WBC detection training")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=("smoke", "main"), default="main")
    parser.add_argument("--smoke-run", type=Path, help="Completed 1-epoch CUDA run required for main training")
    parser.add_argument("--resume", type=Path, help="Explicit last.pt from an unfinished main run")
    parser.add_argument("--allow-batch-fallback", action="store_true", help="On AutoBatch/OOM retry batch=4 in a new run")
    args = parser.parse_args()
    run = run_training(load_config(args.config), kind=args.mode, smoke_run=args.smoke_run,
                       resume=args.resume, allow_batch_fallback=args.allow_batch_fallback)
    print(f"RUN_DIR={run}", flush=True)


if __name__ == "__main__":
    main()

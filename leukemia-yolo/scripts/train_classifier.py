"""Windows-safe Stage 2 training entry point."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classifier_config import classifier_config, classifier_output
from src.classifier_training import train
from src.config import run_id


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value); stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    parser = argparse.ArgumentParser(description="Train two-phase CUDA EfficientNet-B3 on saved C-NMC subject splits")
    parser.add_argument("--config", help="Optional config.local.yaml")
    parser.add_argument("--mode", choices=("smoke", "head", "fine", "full"), default="full")
    parser.add_argument("--resume", type=Path, help="Existing run/last_checkpoint.pth")
    parser.add_argument("--batch-size", type=int, choices=(2, 4, 8))
    parser.add_argument("--workers", type=int, help="Use 0 if Windows data-loader workers fail")
    parser.add_argument("--max-minutes", type=float, help="Stop safely after a completed epoch")
    args = parser.parse_args()
    cfg = classifier_config(args.config)
    if args.workers is not None:
        cfg["classifier_workers"] = args.workers
    run_dir = args.resume.resolve().parent if args.resume else classifier_output(cfg) / "runs" / run_id(
        "smoke" if args.mode == "smoke" else "efficientnet_b3")
    batch = args.batch_size or cfg["classifier_batch_size"]
    resume = args.resume
    while True:
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "training_log.txt").open("a", encoding="utf-8") as log:
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = Tee(old_out, log); sys.stderr = Tee(old_err, log)
            try:
                result = train(cfg, args.mode, resume, batch_size=batch,
                               max_minutes=args.max_minutes, run_dir=run_dir)
                print("Classifier run:", result, flush=True)
                return
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or batch <= 2:
                    raise
                next_batch = 4 if batch > 4 else 2
                print(f"CUDA OOM at batch {batch}; retrying at {next_batch}. No other hyperparameters changed.", flush=True)
                __import__("torch").cuda.empty_cache()
                resume = run_dir / "last_checkpoint.pth" if (run_dir / "last_checkpoint.pth").is_file() else None
                if resume is None:
                    # Preserve failed run and its log; the replacement has a new ID.
                    run_dir = classifier_output(cfg) / "runs" / run_id(
                        "smoke_oom_retry" if args.mode == "smoke" else "efficientnet_b3_oom_retry")
                batch = next_batch
            finally:
                sys.stdout, sys.stderr = old_out, old_err


if __name__ == "__main__":
    main()

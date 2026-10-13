"""Hardware and notebook-kernel diagnostics; GPU training has a separate hard gate."""
from __future__ import annotations

import importlib
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import psutil

from .config import ensure_outputs, project_root, write_json


def require_cuda(device: int = 0) -> dict:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"PyTorch import failed: {exc}. Install CUDA PyTorch using the official selector.") from exc
    if torch.version.cuda is None:
        raise RuntimeError(f"Installed PyTorch {torch.__version__} is CPU-only (torch.version.cuda is None). Install a compatible CUDA build from pytorch.org/get-started/locally/.")
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable with torch {torch.__version__}, CUDA wheel {torch.version.cuda}. Check NVIDIA driver, nvidia-smi, and the PyTorch CUDA wheel. GPU training stopped.")
    if device >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device} does not exist; detected {torch.cuda.device_count()} CUDA device(s).")
    free, total = torch.cuda.mem_get_info(device)
    props = torch.cuda.get_device_properties(device)
    return {"gpu_name": props.name, "dedicated_vram_gib": total / 2**30,
            "available_vram_gib": free / 2**30,
            "compute_capability": f"{props.major}.{props.minor}",
            "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
            "device": device}


def nvidia_smi() -> dict:
    command = shutil.which("nvidia-smi")
    if not command:
        return {"available": False, "error": "nvidia-smi not found on PATH"}
    result = subprocess.run([command, "--query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
    if result.returncode:
        return {"available": False, "error": result.stderr.strip() or result.stdout.strip()}
    return {"available": True, "query": result.stdout.strip()}


def windows_media_type(path: Path) -> str:
    """Best-effort physical-media check for the configured dataset drive."""
    drive = Path(path).drive.rstrip(":\\/")
    if os.name != "nt" or len(drive) != 1 or not drive.isalpha():
        return "Unknown"
    shell = shutil.which("powershell") or shutil.which("pwsh")
    if not shell:
        return "Unknown"
    query = (f"$partition = Get-Partition -DriveLetter '{drive}' -ErrorAction Stop; "
             "$disk = $partition | Get-Disk -ErrorAction Stop; "
             "$physical = Get-PhysicalDisk -DeviceNumber $disk.Number -ErrorAction Stop; "
             "$physical.MediaType")
    try:
        result = subprocess.run([shell, "-NoProfile", "-Command", query], capture_output=True,
                                text=True, timeout=15)
        media = result.stdout.strip()
        return media if result.returncode == 0 and media else "Unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "Unknown"


def inspect_environment(config: dict, *, check_model: bool = True) -> dict:
    output = ensure_outputs(config)
    rows = []
    def row(item, value, status, detail=""):
        rows.append({"check": item, "value": str(value), "status": status, "detail": detail})
    row("Python executable", sys.executable, "PASS")
    row("Python version", platform.python_version(), "PASS" if sys.version_info[:2] == (3, 11) else "FAIL")
    env = os.environ.get("CONDA_DEFAULT_ENV", "Not set")
    in_env = env == "leukemia-yolo" and Path(sys.prefix).name.lower() == "leukemia-yolo"
    row("Conda environment and active kernel", env, "PASS" if in_env else "WARNING",
        "Kernel executable must belong to leukemia-yolo; never use base for training.")
    row("Operating system", f"{platform.system()} {platform.release()} {platform.version()}",
        "PASS" if platform.system() == "Windows" else "WARNING")
    ram = psutil.virtual_memory()
    row("System RAM total / available GiB", f"{ram.total/2**30:.2f} / {ram.available/2**30:.2f}", "PASS")
    disk = shutil.disk_usage(config["processed_data_path"].anchor or project_root().anchor)
    row("Dataset-drive free GiB", f"{disk.free/2**30:.2f}", "PASS" if disk.free > 2 * 2**30 else "WARNING",
        "Check in Windows Optimize Drives that this drive is SSD before choosing disk cache.")
    media_type = windows_media_type(config["processed_data_path"])
    row("Dataset-drive media type", media_type, "PASS" if media_type.upper() == "SSD" else "WARNING",
        "If unknown, verify in Windows Optimize Drives. Disk cache should use an SSD with enough free space.")
    try:
        import ctypes
        drive = str(config["processed_data_path"].anchor)
        if os.name == "nt" and drive:
            # DriveType=3 means fixed; Windows does not expose SSD versus HDD here.
            dtype = ctypes.windll.kernel32.GetDriveTypeW(drive)
            row("Storage type", "Fixed drive" if dtype == 3 else f"Drive type {dtype}", "WARNING",
                "Confirm SSD manually in Windows Optimize Drives before disk caching.")
    except Exception:
        pass
    for module, label in (("torch", "PyTorch"), ("torchvision", "Torchvision"),
                          ("ultralytics", "Ultralytics"), ("cv2", "OpenCV"),
                          ("PIL", "Pillow"), ("numpy", "NumPy"), ("yaml", "PyYAML")):
        try:
            version = importlib.metadata.version({"cv2": "opencv-python", "PIL": "Pillow", "yaml": "PyYAML"}.get(module, module))
            importlib.import_module(module)
            row(label, version, "PASS")
        except Exception as exc:
            row(label, "Unavailable", "FAIL", str(exc))
    try:
        gpu = require_cuda(config["device"])
        row("CUDA", "Available", "PASS")
        for key in ("gpu_name", "dedicated_vram_gib", "available_vram_gib", "compute_capability"):
            row(key.replace("_", " ").title(), gpu[key], "PASS")
    except RuntimeError as exc:
        gpu = None
        row("CUDA", "Unavailable", "FAIL", str(exc))
    smi = nvidia_smi()
    row("NVIDIA driver (nvidia-smi)", smi.get("query", "Unavailable"),
        "PASS" if smi["available"] else "WARNING", smi.get("error", ""))
    for folder in (output, output / "checkpoints", output / "figures", output / "predictions", output / "paper_results"):
        try:
            probe = folder / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            row("Writable output", folder, "PASS")
        except OSError as exc:
            row("Writable output", folder, "FAIL", str(exc))
    checkpoint = project_root() / config["model"]
    if check_model:
        try:
            # Child process tests the real supported API and keeps any downloaded
            # detection checkpoint in the ignored project root.
            probe = subprocess.run([sys.executable, "-c",
                "from ultralytics import YOLO; m=YOLO('yolov8n.pt'); assert m.task=='detect'; print('detection checkpoint loaded')"],
                cwd=project_root(), capture_output=True, text=True, timeout=120)
            model_status = "PASS" if probe.returncode == 0 and checkpoint.is_file() else "FAIL"
            detail = ("Detection checkpoint loaded from cache/download. " + probe.stdout.strip()) if model_status == "PASS" else (probe.stderr.strip() or probe.stdout.strip())[-1000:]
        except (OSError, subprocess.TimeoutExpired) as exc:
            model_status, detail = "FAIL", str(exc)
    else:
        model_status, detail = "WARNING", "Model download/cache check skipped for offline fixture."
    row("YOLOv8n weight/cache", checkpoint if checkpoint.exists() else "Unavailable", model_status, detail)
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "checks": rows, "gpu": gpu,
              "nvidia_smi": smi, "cuda_ready": gpu is not None, "kernel_ready": in_env}
    write_json(output / "environment_report.json", report)
    packages = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    (output / "environment_packages.txt").write_text(packages.stdout or packages.stderr, encoding="utf-8")
    return report


def status_table(report: dict):
    import pandas as pd
    from IPython.display import HTML, display
    table = pd.DataFrame(report["checks"])
    display(HTML(table.style.map(lambda value: "color: #16803c; font-weight: bold" if value == "PASS"
                       else "color: #ad6400; font-weight: bold" if value == "WARNING"
                       else "color: #bb2222; font-weight: bold" if value == "FAIL" else "",
                       subset=["status"]).to_html()))
    return table


def main():
    import argparse
    from .config import load_config
    parser = argparse.ArgumentParser(description="Save Windows, package, and CUDA environment diagnostics")
    parser.add_argument("--config")
    parser.add_argument("--skip-model-download", action="store_true")
    args = parser.parse_args()
    report = inspect_environment(load_config(args.config), check_model=not args.skip_model_download)
    for row in report["checks"]:
        print(f"{row['status']:7} {row['check']}: {row['value']} {row['detail']}")


if __name__ == "__main__":
    main()

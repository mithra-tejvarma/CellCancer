"""Read C-NMC training folds in place; never modify source images."""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import re

import pandas as pd
from PIL import Image

from .classifier_config import classifier_output
from .config import write_json

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png"}
PATTERN = re.compile(r"^UID_(H?\d+)_(\d+)_(\d+)_(all|hem)$", re.IGNORECASE)
FOLDS = ("fold_0", "fold_1", "fold_2")
<<<<<<< HEAD
TRAINING_DIRECTORY_NAMES = {"training_data", "c-nmc_training_data"}
CHALLENGE_TEST_DIRECTORY_NAMES = {
    "testing_data", "c-nmc_test_final_phase_data", "c-nmc_test_prelim_phase_data",
}
=======
>>>>>>> f5f3e3e40fcb7dd65b681e3014c49a9f3044dac3


def _child(parent: Path, name: str) -> Path | None:
    found = [p for p in parent.iterdir() if p.is_dir() and p.name.casefold() == name.casefold()]
    if len(found) > 1:
        raise ValueError(f"Ambiguous case-insensitive directory {name} under {parent}")
    return found[0] if found else None


def fold_layout(candidate: Path) -> dict | None:
    if not candidate.is_dir():
        return None
    layout = {}
    for fold in FOLDS:
        folder = _child(candidate, fold)
        if folder is None:
            return None
        classes = {name: _child(folder, name) for name in ("all", "hem")}
        if any(value is None for value in classes.values()):
            return None
        layout[fold] = classes
    return layout


def discover_training_data(root: Path) -> tuple[Path, dict]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"C-NMC path does not exist: {root}. Set cnmc_root in config.local.yaml.")
<<<<<<< HEAD
    if any(part.casefold() in CHALLENGE_TEST_DIRECTORY_NAMES for part in root.parts):
        raise ValueError("cnmc_root points to challenge test data. Select the parent dataset or C-NMC_training_data.")
    candidates = [root]
    pending, visited = [root], set()
    while pending:
        parent = pending.pop()
        resolved = parent.resolve()
        if resolved in visited:
            continue
        visited.add(resolved)
        for child in parent.iterdir():
            if not child.is_dir() or child.is_symlink():
                continue
            name = child.name.casefold()
            if name in CHALLENGE_TEST_DIRECTORY_NAMES or name in FOLDS:
                continue
            if name in TRAINING_DIRECTORY_NAMES:
                candidates.append(child)
            pending.append(child)
    valid = {p.resolve(): layout for p in candidates if (layout := fold_layout(p)) is not None}
    if not valid:
        raise FileNotFoundError(
            f"No verified C-NMC training directory under {root}. Expected C-NMC_training_data "
            "or training_data containing fold_0, fold_1, fold_2, each with all and hem. "
            "Set cnmc_root in config.local.yaml to the dataset parent or the training directory.")
=======
    candidates = [root, *(p for p in root.rglob("*") if p.is_dir() and p.name.casefold() == "training_data")]
    valid = {p.resolve(): layout for p in candidates if (layout := fold_layout(p)) is not None}
    if not valid:
        raise FileNotFoundError(f"No C-NMC training_data with fold_0/1/2 and all/hem under {root}; check cnmc_root.")
>>>>>>> f5f3e3e40fcb7dd65b681e3014c49a9f3044dac3
    if len(valid) != 1:
        raise ValueError(f"Multiple valid C-NMC training roots under {root}: {list(valid)}. Set cnmc_root to one.")
    return next(iter(valid.items()))


def subject_id(filename: str, class_name: str) -> str:
    match = PATTERN.fullmatch(Path(filename).stem)
    if not match:
        raise ValueError(f"Unsupported C-NMC filename {filename!r}; expected UID_[H]subject_image_cell_all|hem")
    subject, image_number, cell_number, suffix = match.groups()
    if suffix.casefold() != class_name.casefold() or subject.upper().startswith("H") != (class_name.casefold() == "hem"):
        raise ValueError(f"Class/subject mismatch in {filename!r}")
    if any(int(n) <= 0 for n in (subject.lstrip("Hh"), image_number, cell_number)):
        raise ValueError(f"Nonpositive subject/image/cell number in {filename!r}")
    return ("HEM:H" if class_name.casefold() == "hem" else "ALL:") + str(int(subject.lstrip("Hh")))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_cnmc(cfg: dict, *, save: bool = True) -> tuple[pd.DataFrame, dict]:
    training, layout = discover_training_data(cfg["cnmc_root"])
    rows, errors, unexpected = [], [], []
    for fold in FOLDS:
        for class_name, folder in layout[fold].items():
            for path in sorted(folder.iterdir()):
                if not path.is_file():
                    continue
                if path.suffix.casefold() not in IMAGE_EXTENSIONS:
                    unexpected.append(str(path))
                    continue
                try:
                    sid = subject_id(path.name, class_name)
                    with Image.open(path) as image:
                        image.load()
                        width, height, bands, image_format = *image.size, len(image.getbands()), image.format
                    rows.append({"absolute_path": str(path.resolve()), "relative_path": str(path.relative_to(training)),
                                 "filename": path.name, "fold": fold, "class_name": class_name.upper(),
                                 "class_id": 1 if class_name == "all" else 0, "subject_id": sid,
                                 "image_width": width, "image_height": height, "image_channels": bands,
                                 "image_format": image_format, "sha256": _sha256(path)})
                except Exception as exc:
                    errors.append({"path": str(path), "error": str(exc)})
    df = pd.DataFrame(rows)
    duplicates = df[df.duplicated("filename", keep=False)]["filename"].unique().tolist() if not df.empty else []
    duplicate_content = df[df.duplicated("sha256", keep=False)]["relative_path"].tolist() if not df.empty else []
    classes = {"ALL": int((df.class_id == 1).sum()), "HEM": int((df.class_id == 0).sum())} if not df.empty else {"ALL": 0, "HEM": 0}
    report = {"training_data": str(training), "total_images": len(df), "class_counts": classes,
              "expected_training_counts": {"ALL": 7272, "HEM": 3389, "total": 10661, "subjects": 73},
              "matches_official_counts": classes == {"ALL": 7272, "HEM": 3389},
              "fold_class_counts": df.groupby(["fold", "class_name"]).size().to_dict() if not df.empty else {},
              "subject_count": int(df.subject_id.nunique()) if not df.empty else 0,
              "duplicate_filenames": duplicates, "duplicate_content": duplicate_content,
              "unexpected_files": unexpected, "errors": errors,
              "image_formats": dict(Counter(df.image_format)) if not df.empty else {},
              "ready": bool(len(df)) and not errors and not duplicates}
    report["fold_class_counts"] = {f"{a}/{b}": int(n) for (a, b), n in report["fold_class_counts"].items()}
    if save:
        output = classifier_output(cfg)
        df.to_csv(output / "cnmc_manifest.csv", index=False)
        write_json(output / "dataset_inspection.json", report)
    if not report["ready"]:
        raise ValueError(f"C-NMC integrity failed: {errors[:3]} duplicate_filenames={duplicates[:3]}")
    return df, report


def sample_figures(df: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 8, figsize=(16, 5))
    for label, row_axes in zip((0, 1), axes):
        subset = df[df.class_id == label].sample(n=min(8, (df.class_id == label).sum()), random_state=42)
        for ax, (_, row) in zip(row_axes, subset.iterrows()):
            with Image.open(row.absolute_path) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_title(f"{'HEM' if label == 0 else 'ALL'} · {row.subject_id}", fontsize=8)
            ax.axis("off")
        for ax in row_axes[len(subset):]:
            ax.axis("off")
    fig.tight_layout(); fig.savefig(output / "cnmc_samples.png", dpi=200); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    df.class_name.value_counts().reindex(["HEM", "ALL"], fill_value=0).plot.bar(ax=ax, color=["#277da1", "#d1495b"])
    ax.set(ylabel="Cell images", title="C-NMC labeled training folds"); fig.tight_layout()
    fig.savefig(output / "class_distribution.png", dpi=200); plt.close(fig)

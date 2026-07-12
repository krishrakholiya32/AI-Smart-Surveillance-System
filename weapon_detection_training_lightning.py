"""Weapon Detection — Fine-Tuning Pipeline (Lightning AI Studio)

Lightning-adapted version of weapon_detection_training.ipynb. Same data
sources, same 2-class (gun/knife) schema, same corrected hyperparameters
(see project_surveillance.md memory note, 2026-07-09) — but restructured
as a single resumable script instead of Kaggle notebook cells, since:

  - Lightning Studio's local disk persists automatically across restarts
    (unlike Kaggle's /kaggle/working, which wipes every new session) — so
    there's no manual "save as Dataset, re-attach next session" dance.
    Just re-run this script after any interruption; it detects an
    existing checkpoint and resumes on its own.
  - Free-tier GPU-hours here run on *interruptible* capacity — the
    process can be killed at any moment with no warning (not a clean
    12-hour session boundary like Kaggle). Run this via a Lightning Job,
    or in a terminal with `nohup python weapon_detection_training_lightning.py &`,
    so a browser-tab close doesn't matter — then just re-run the same
    command if the instance itself gets reclaimed.
  - Lightning Studios ship with a working CUDA/PyTorch environment
    already, so the Kaggle-specific cu118 reinstall hack (needed there
    for an image-specific cuDNN mismatch) is skipped.
  - Secrets: set ROBOFLOW_API_KEY under Studio Settings > Environment
    Variables (Lightning injects it into os.environ directly — no
    special client needed, unlike Kaggle Secrets).

BEFORE RUNNING:
  1. Upload your current models/weapon.pt into this Studio's file browser
     (drag-and-drop into the home directory) — the script expects it at
     ./weapon.pt on first run.
  2. Set ROBOFLOW_API_KEY in Studio Settings > Environment Variables.
  3. `pip install ultralytics fiftyone roboflow pyyaml` (Lightning
     Studios keep pip installs across restarts too).
"""

import os
import shutil
from collections import Counter
from pathlib import Path

import torch
import yaml

# ── Paths — persist automatically on Lightning's local disk ────────────────
ROOT = Path.home() / "weapon_dataset"
WEIGHTS_SRC = Path.home() / "weapon.pt"
RUN_DIR = Path.home() / "weapon_model"
LAST_CKPT = RUN_DIR / "weights" / "last.pt"
DATA_YAML = ROOT / "data.yaml"

CLASS_NAMES = {0: "gun", 1: "knife"}


def _ensure_dirs() -> None:
    for split in ["train", "val", "test"]:
        (ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (ROOT / split / "labels").mkdir(parents=True, exist_ok=True)


# ── Step 1 — Open Images v7 via FiftyOne (skipped if data.yaml already exists) ─
def _download_open_images() -> None:
    import fiftyone as fo  # noqa: F401  (import registers zoo datasets)
    import fiftyone.zoo as foz

    classes = ["Handgun", "Knife"]
    max_samples = {"train": 25000, "validation": 4000}
    out_split_name = {"train": "train", "validation": "val"}

    for fo_split, max_n in max_samples.items():
        out_split = out_split_name[fo_split]
        print(f'\n=== Downloading Open Images "{fo_split}" split (max {max_n:,}) ===')
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split=fo_split,
            label_types=["detections"],
            classes=classes,
            max_samples=max_n,
            shuffle=True,
            seed=51,
        )
        img_dir = ROOT / out_split / "images"
        lbl_dir = ROOT / out_split / "labels"
        copied = 0
        for sample in dataset:
            dets = getattr(sample, "ground_truth", None)
            if dets is None or not dets.detections:
                continue
            lines = []
            for det in dets.detections:
                if det.label not in classes:
                    continue
                cls_id = classes.index(det.label)
                x, y, w, h = det.bounding_box
                cx, cy = x + w / 2.0, y + h / 2.0
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            if not lines:
                continue
            src = Path(sample.filepath)
            shutil.copy(str(src), str(img_dir / src.name))
            with open(lbl_dir / (src.stem + ".txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
            copied += 1
        print(f"{out_split}: {copied} images exported with Handgun/Knife labels.")


# ── Step 2 — merge 3 Roboflow weapon datasets into the same 2-class schema ──
def _download_roboflow() -> None:
    from roboflow import Roboflow

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ROBOFLOW_API_KEY not set. Add it under Studio Settings > "
            "Environment Variables, then restart the Studio."
        )
    rf = Roboflow(api_key=api_key)

    datasets = [
        ("test-7awfy", "weapon-detection-f1lih", 1, Path.home() / "roboflow_dataset_1"),
        ("mahad-ahmed", "gun-and-knife-detection", 1, Path.home() / "roboflow_dataset_2"),
        ("edi-detection", "weapon-yolo8", 1, Path.home() / "roboflow_dataset_3"),
    ]

    gun_keywords = {"gun", "pistol", "handgun", "firearm", "rifle", "weapon", "revolver"}
    knife_keywords = {"knife", "blade", "dagger", "sword", "machete", "cleaver"}

    def remap_label_file(src_path, dst_path, remap):
        lines_out = []
        with open(src_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                new_id = remap.get(int(parts[0]))
                if new_id is None:
                    continue
                parts[0] = str(new_id)
                lines_out.append(" ".join(parts))
        if lines_out:
            with open(dst_path, "w") as f:
                f.write("\n".join(lines_out) + "\n")
            return True
        return False

    for workspace, project_slug, ver, location in datasets:
        print(f"\n=== Dataset: {workspace}/{project_slug} v{ver} ===")
        try:
            project = rf.workspace(workspace).project(project_slug)
            project.version(ver).download("yolov8", location=str(location))
        except Exception as e:
            print(f"  SKIPPED — failed to download ({e}).")
            continue

        rf_yaml = location / "data.yaml"
        if not rf_yaml.exists():
            print(f"  SKIPPED — no data.yaml found at {rf_yaml}")
            continue

        with open(rf_yaml) as f:
            rf_cfg = yaml.safe_load(f)
        rf_names = [n.lower().strip() for n in rf_cfg.get("names", [])]

        remap = {}
        for old_id, name in enumerate(rf_names):
            if any(k in name for k in gun_keywords):
                remap[old_id] = 0
            elif any(k in name for k in knife_keywords):
                remap[old_id] = 1
            else:
                remap[old_id] = None

        split_map = {"train": "train", "valid": "val", "test": "test"}
        copied = {"train": 0, "val": 0, "test": 0}
        for rf_split, our_split in split_map.items():
            img_dir = location / rf_split / "images"
            lbl_dir = location / rf_split / "labels"
            if not img_dir.exists():
                continue
            for img_path in img_dir.glob("*.*"):
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if not lbl_path.exists():
                    continue
                unique_stem = f"{project_slug}_{img_path.stem}"
                dst_lbl = ROOT / our_split / "labels" / f"{unique_stem}.txt"
                if remap_label_file(str(lbl_path), str(dst_lbl), remap):
                    dst_img = ROOT / our_split / "images" / f"{unique_stem}{img_path.suffix}"
                    shutil.copy(str(img_path), str(dst_img))
                    copied[our_split] += 1
        print(f"  Merged: {copied}")


def _summarize_dataset() -> int:
    total_train = 0
    for split in ["train", "val", "test"]:
        lbl_dir = ROOT / split / "labels"
        img_files = list((ROOT / split / "images").glob("*.*"))
        counts = Counter()
        for lf in lbl_dir.glob("*.txt"):
            with open(lf) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        counts[int(line.split()[0])] += 1
        print(f"[{split}] images={len(img_files)}", {CLASS_NAMES[k]: v for k, v in counts.items()})
        if split == "train":
            total_train = len(img_files)
    return total_train


def _write_data_yaml() -> None:
    cfg = {
        "path": str(ROOT),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 2,
        "names": ["gun", "knife"],
    }
    with open(DATA_YAML, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"data.yaml written to {DATA_YAML}")


# ── Step 3 — fine-tune (auto-resumes from LAST_CKPT if this is a re-run) ───
def _train() -> None:
    from ultralytics import YOLO

    assert torch.cuda.is_available(), "No GPU visible — check the Studio's GPU is attached."
    print("GPU:", torch.cuda.get_device_name(0))

    if LAST_CKPT.exists():
        # Resuming a previous (likely interrupted) run — same optimizer
        # state, LR schedule, epoch count. Don't pass training kwargs here;
        # resume=True restores the original run's config from last.pt.
        print(f"Found existing checkpoint at {LAST_CKPT} — resuming.")
        model = YOLO(str(LAST_CKPT))
        model.train(resume=True)
        return

    if not WEIGHTS_SRC.exists():
        raise FileNotFoundError(
            f"{WEIGHTS_SRC} not found. Upload your current models/weapon.pt "
            "into this Studio's home directory before running."
        )

    model = YOLO(str(WEIGHTS_SRC))
    print("Base model classes:", model.names)
    if model.names != CLASS_NAMES:
        raise ValueError(f"Unexpected base class mapping: {model.names} — expected gun/knife at 0/1.")

    # Same dataset-scale-corrected hyperparameters as the Kaggle notebook
    # (Cell 9 / "CELL 12"), corrected 2026-07-09 — the ~35-40k merged
    # dataset is ~10x the original weapon.pt's training set, so this is
    # budgeted like real training, not a light touch-up.
    model.train(
        data=str(DATA_YAML),
        epochs=200,
        imgsz=1280,
        batch=16,
        device=0,  # single GPU on Lightning; set [0, 1] if you provision 2
        name="weapon_model",
        project=str(Path.home()),
        patience=40,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=5,
        weight_decay=0.0005,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        perspective=0.0005,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        erasing=0.4,
        close_mosaic=10,
        save_period=5,  # checkpoint every 5 epochs — required for resuming after an interruption
        plots=True,
    )


def _export_final() -> None:
    from ultralytics import YOLO

    best = RUN_DIR / "weights" / "best.pt"
    output = Path.home() / "weapon.pt.new"
    shutil.copy(str(best), str(output))

    model = YOLO(str(output))
    print("Model classes:", model.names)
    if model.names == CLASS_NAMES:
        print("Class mapping CORRECT.")
    else:
        print("WARNING: Unexpected class mapping — check the merge step.")

    print(f"\nFinal weights: {output}")
    print("Next steps:")
    print("  1. Download this file from the Studio file browser")
    print("  2. Place it in models/weapon.pt in the AI Smart Surveillance System repo")
    print("     (replacing the existing weapon.pt)")
    print("  3. Delete/regenerate models/weapon.onnx so the backend re-exports ONNX")
    print("  4. Trained at imgsz=1280 — consider raising the pipeline's inference")
    print("     resolution in backend/app/services/detection/pipeline.py")


if __name__ == "__main__":
    _ensure_dirs()
    if not DATA_YAML.exists():
        # Only download/merge once — on a resumed run after an interruption,
        # the dataset is already on disk (Lightning persists it), skip straight to training.
        _download_open_images()
        _download_roboflow()
        total_train = _summarize_dataset()
        if total_train < 500:
            raise RuntimeError("Low image count — check the Roboflow downloads above.")
        _write_data_yaml()
    else:
        print(f"{DATA_YAML} already exists — skipping dataset download/merge.")

    _train()
    _export_final()

#!/usr/bin/env python3
"""
Direct PCB Training Script (MPS-accelerated)
---------------------------------------------
Trains YOLO11n on the already-present pcb-defects dataset using
Apple Silicon MPS GPU acceleration and saves model to models/pcb_model.pt

Usage:
    python3 run_training.py
"""
import sys
import shutil
from pathlib import Path

# Make sure repo root on path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

DATA_DIR   = REPO_ROOT / "data" / "pcb-defects"
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Use already-present yolo11n.pt (nano = much faster on CPU/MPS)
MODEL_FILE = str(REPO_ROOT / "yolo11n.pt")


def main():
    from ultralytics import YOLO
    from src.data_ingestion import DataIngestion
    from src.config import Config

    # ── Prepare dataset ──────────────────────────────────────────────────
    print("=" * 60)
    print("  PCB Defect Detection — YOLO11n (MPS-accelerated)")
    print("=" * 60)

    config = Config.create(epochs=50)

    data = DataIngestion(data_path=DATA_DIR)
    if not data.find_data_structure():
        print("❌ Dataset not found!")
        sys.exit(1)

    data.collect_images()
    stats = data.get_stats()
    print(f"\n📊 Dataset: {stats['total_images']} images ({stats['with_xml']} annotated)")

    train_count, val_count = data.create_yolo_dataset()
    print(f"📂 Train: {train_count} | Val: {val_count}")

    yaml_path = data.get_yaml_path()
    print(f"✅ Dataset YAML: {yaml_path}")

    # ── Train ────────────────────────────────────────────────────────────
    print(f"\n🚀 Starting YOLO11n training with MPS acceleration...")
    model = YOLO(MODEL_FILE)

    output_path = Config.get_output_path()
    results = model.train(
        data=str(yaml_path),
        epochs=50,
        imgsz=640,
        batch=8,           # smaller batch = less memory pressure on MPS
        device="mps",      # Apple Silicon GPU
        project=str(output_path),
        name="pcb_yolo",
        exist_ok=True,
        patience=15,
        optimizer="AdamW",
        lr0=0.001,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        flipud=0.5,
        fliplr=0.5,
        degrees=10.0,
        translate=0.2,
        scale=0.9,
        workers=4,
        cache=False,       # avoid RAM cache issues
        verbose=True,
    )

    # ── Copy best model ───────────────────────────────────────────────────
    best = output_path / "pcb_yolo" / "weights" / "best.pt"
    if best.exists():
        dest = MODELS_DIR / "pcb_model.pt"
        shutil.copy(best, dest)
        print(f"\n✅ Model saved → {dest}")
        print("   Restart the webapp: python3 webapp/backend/app.py")
    else:
        print("⚠️  best.pt not found — check training logs")


if __name__ == "__main__":
    main()

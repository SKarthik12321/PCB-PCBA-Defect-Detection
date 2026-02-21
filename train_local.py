#!/usr/bin/env python3
"""
PCB Defect Detection — Local Training Script
=============================================
Downloads the akhatova/pcb-defects Kaggle dataset and trains a YOLO11 model locally.

REQUIREMENTS:
  1. Kaggle API credentials in ~/.kaggle/kaggle.json
     (Get from https://www.kaggle.com/settings → API → "Create New Token")
  2. Python packages: pip install -r requirements.txt

USAGE:
  python3 train_local.py

The trained model will be placed in models/pcb_model.pt automatically.
Then restart the webapp server to use it:
  python3 webapp/backend/app.py
"""

import subprocess
import sys
import shutil
import zipfile
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
MODELS_DIR = REPO_ROOT / "models"
DATA_DIR   = REPO_ROOT / "data" / "pcb-defects"
KAGGLE_DATASET = "akhatova/pcb-defects"


def check_kaggle_credentials():
    cred = Path.home() / ".kaggle" / "kaggle.json"
    if not cred.exists():
        print("❌ Kaggle credentials not found!")
        print()
        print("Steps to fix:")
        print("  1. Go to: https://www.kaggle.com/settings")
        print("  2. Scroll to API section → click 'Create New Token'")
        print("  3. Save the downloaded kaggle.json to: ~/.kaggle/kaggle.json")
        print("  4. Run:  chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)
    try:
        with open(cred) as f:
            data = json.load(f)
        assert "username" in data and "key" in data
        print(f"✅ Kaggle credentials found for: {data['username']}")
    except Exception as e:
        print(f"❌ Invalid kaggle.json: {e}")
        sys.exit(1)


def download_dataset():
    if DATA_DIR.exists() and any(DATA_DIR.rglob("*.xml")):
        print(f"✅ Dataset already present at: {DATA_DIR}")
        return
    print(f"📥 Downloading dataset: {KAGGLE_DATASET} ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
         "-p", str(DATA_DIR), "--unzip"],
        check=True
    )
    print("✅ Dataset downloaded!")


def run_training():
    from src.trainer import TrainingManager
    from src.config import Config

    config = Config.create(epochs=50)

    trainer = TrainingManager(data_path=DATA_DIR, config=config)
    metrics = trainer.run_pipeline()

    # Copy best model to models/
    output = Config.get_output_path()
    best = output / "pcb_yolo" / "weights" / "best.pt"
    if best.exists():
        dest = MODELS_DIR / "pcb_model.pt"
        shutil.copy(best, dest)
        print(f"\n✅ Trained model saved → {dest}")
        print("   Now restart the webapp: python3 webapp/backend/app.py")
    else:
        print("⚠️  best.pt not found — check training logs above.")

    return metrics


def main():
    print("=" * 60)
    print("  PCB Defect Detection — Local Training")
    print("=" * 60)

    check_kaggle_credentials()
    download_dataset()
    run_training()


if __name__ == "__main__":
    main()

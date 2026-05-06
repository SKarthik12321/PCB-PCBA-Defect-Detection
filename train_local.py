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

# Standard libraries for system commands, path building, and JSON parsing
import subprocess
import sys
import shutil
import zipfile
import json
from pathlib import Path

# Determine absolute path to the repository root to ensure scripts run from anywhere
REPO_ROOT = Path(__file__).resolve().parent
# Define dedicated directories for output model weights and input raw data
MODELS_DIR = REPO_ROOT / "models"
DATA_DIR   = REPO_ROOT / "data" / "pcb-defects"
# Target Kaggle dataset identifier for download
KAGGLE_DATASET = "akhatova/pcb-defects"


def check_kaggle_credentials():
    # Construct path to check if Kaggle API key is installed locally
    cred = Path.home() / ".kaggle" / "kaggle.json"
    if not cred.exists():
        print(" Kaggle credentials not found!")
        print()
        print("Steps to fix:")
        print("  1. Go to: https://www.kaggle.com/settings")
        print("  2. Scroll to API section → click 'Create New Token'")
        print("  3. Save the downloaded kaggle.json to: ~/.kaggle/kaggle.json")
        print("  4. Run:  chmod 600 ~/.kaggle/kaggle.json")
        # sys.exit(1) forcefully terminates the script execution because it cannot proceed without an API key (1 = error code)
        sys.exit(1)
    try:
        # Load JSON config to verify valid Kaggle username and secret key layout
        with open(cred) as f:
            data = json.load(f)
        assert "username" in data and "key" in data
        print(f" Kaggle credentials found for: {data['username']}")
    except Exception as e:
        print(f" Invalid kaggle.json: {e}")
        # sys.exit(1) forcefully terminates the script execution because it cannot proceed without an API key (1 = error code)
        sys.exit(1)


def download_dataset():
    # Skips massive internet download if local data mapping exists and is populated
    if DATA_DIR.exists() and any(DATA_DIR.rglob("*.xml")):
        print(f" Dataset already present at: {DATA_DIR}")
        return
    print(f"📥 Downloading dataset: {KAGGLE_DATASET} ...")
    # Creates necessary subdirectories and downloads compressed raw data
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # subprocess.run is used to securely execute shell terminal commands directly from Python 
    # check=True ensures Python throws an error if the shell command fails
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
         "-p", str(DATA_DIR), "--unzip"],
        check=True
    )
    print(" Dataset downloaded!")


def run_training():
    from src.trainer import TrainingManager
    from src.config import Config

    # Initialize YOLO config with 50 epochs for fast convergence on PCB dataset
    config = Config.create(epochs=50)

    # Initialize the TrainingManager which orchestrates the entire YOLO pipeline
    trainer = TrainingManager(data_path=DATA_DIR, config=config)
    
    # Trigger full pipeline: train/val split -> active training -> metric evaluation
    metrics = trainer.run_pipeline()

    # Copy generated deep-learning weights file systematically for production serving
    output = Config.get_output_path()
    best = output / "pcb_yolo" / "weights" / "best.pt"
    if best.exists():
        dest = MODELS_DIR / "pcb_model.pt"
        shutil.copy(best, dest)
        print(f"\n Trained model saved → {dest}")
        print("   Now restart the webapp: python3 webapp/backend/app.py")
    else:
        print("  best.pt not found — check training logs above.")

    return metrics


def main():
    # Print console header for user visibility
    print("=" * 60)
    print("  PCB Defect Detection — Local Training")
    print("=" * 60)

    # Sequentially execute the pipeline: auth -> ingest -> train -> test
    check_kaggle_credentials()
    download_dataset()
    run_training()


# This protects the script so that main() only triggers if the file is executed directly (python3 train_local.py)
# and NOT if it is imported into another script
if __name__ == "__main__":
    main()

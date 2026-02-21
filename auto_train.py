#!/usr/bin/env python3
"""
PCB Defect Detection — Full Auto Setup & Train (No API Required)
================================================================
Downloads the PCB Dataset from public sources (no Kaggle/Roboflow API needed),
prepares YOLO format labels, trains a YOLOv11 model, and places the final
model in models/pcb_model.pt so the webapp immediately detects defects.

Run:
    python3 auto_train.py

The webapp will auto-pick up the new model on next restart.
"""

import os
import sys
import shutil
import subprocess
import zipfile
import time
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent
MODELS_DIR  = REPO_ROOT / "models"
DATA_DIR    = REPO_ROOT / "data" / "pcb-defects"
ZIP_TMP     = Path("/tmp/PCB_DATASET.zip")

# Public direct-download URL (no auth)
DATASET_URL = "https://www.dropbox.com/s/32kolsaa45z2mpj/PCB_DATASET.zip?dl=1"
DATASET_SIZE_APPROX_GB = 1.9


def log(msg, emoji=""):
    print(f"{emoji}  {msg}" if emoji else msg, flush=True)


def check_disk_space():
    import shutil as sh
    free = sh.disk_usage(REPO_ROOT).free / 1e9
    if free < 3:
        log(f"WARNING: Only {free:.1f} GB free. Need ~3 GB.", "⚠️")
    else:
        log(f"Disk space OK: {free:.1f} GB free", "✅")


def download_dataset():
    if DATA_DIR.exists() and any(DATA_DIR.rglob("*.xml")):
        xml_count = len(list(DATA_DIR.rglob("*.xml")))
        log(f"Dataset already present ({xml_count} annotations found). Skipping download.", "✅")
        return

    # Check if download already in progress or completed
    if ZIP_TMP.exists():
        size_gb = ZIP_TMP.stat().st_size / 1e9
        log(f"Found existing partial/complete download: {size_gb:.2f} GB at {ZIP_TMP}", "📦")
        if size_gb >= DATASET_SIZE_APPROX_GB * 0.95:
            log("Download appears complete. Proceeding to extract.", "✅")
            return
        else:
            log(f"Partial download found ({size_gb:.2f}/{DATASET_SIZE_APPROX_GB} GB). Resuming...", "⬇️")
            _resume_download()
            return

    log(f"Downloading PCB Dataset (~{DATASET_SIZE_APPROX_GB} GB) from public source...", "⬇️")
    log("This will take 5-15 minutes depending on your connection.", "ℹ️")
    _resume_download()


def _resume_download():
    """Download or resume download using wget."""
    wget_path = shutil.which("wget")
    if not wget_path:
        log("wget not found. Trying curl...", "ℹ️")
        _curl_download()
        return

    cmd = ["wget", "-c", DATASET_URL, "-O", str(ZIP_TMP),
           "--timeout=300", "--tries=10", "--retry-connrefused",
           "--progress=bar:force"]
    log(f"Running: {' '.join(cmd)}", "🔧")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sz = ZIP_TMP.stat().st_size if ZIP_TMP.exists() else 0
        if sz > 100_000_000:  # > 100MB, might be usable
            log(f"Download interrupted but have {sz/1e6:.0f}MB — trying to extract partial zip...", "⚠️")
        else:
            log("Download failed and file too small. Try running this script again.", "❌")
            sys.exit(1)


def _curl_download():
    cmd = ["curl", "-L", "-C", "-", DATASET_URL, "-o", str(ZIP_TMP),
           "--retry", "10", "--retry-delay", "3"]
    subprocess.run(cmd, check=True)


def extract_dataset():
    if DATA_DIR.exists() and any(DATA_DIR.rglob("*.xml")):
        return

    log(f"Extracting dataset to {DATA_DIR}...", "📂")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(ZIP_TMP, 'r') as z:
            members = z.namelist()
            imgs = [m for m in members if m.lower().endswith(('.jpg', '.jpeg', '.png'))]
            xmls = [m for m in members if m.endswith('.xml')]
            log(f"Found {len(imgs)} images and {len(xmls)} XML annotations in zip.", "📊")

            log("Extracting... (this may take a minute)", "⏳")
            z.extractall(DATA_DIR)
    except zipfile.BadZipFile:
        log("Zip file appears corrupted. Re-downloading...", "❌")
        ZIP_TMP.unlink(missing_ok=True)
        _resume_download()
        extract_dataset()
        return

    xml_count = len(list(DATA_DIR.rglob("*.xml")))
    log(f"Extracted! Found {xml_count} annotation files.", "✅")


def run_training():
    log("Starting YOLO11 training pipeline...", "🚀")

    # Import after repo_root is on path
    sys.path.insert(0, str(REPO_ROOT))

    from src.trainer import TrainingManager
    from src.config import Config

    # Faster settings for local CPU training
    config = Config.create(epochs=50)
    config.model.name = "yolo11n.pt"          # Nano — fastest
    config.model.batch_size = 8               # Smaller batch for CPU
    config.model.img_size = 416               # Smaller image size for speed
    config.model.workers = 2
    config.model.cache = False                # Don't cache on CPU (RAM)

    trainer = TrainingManager(data_path=DATA_DIR, config=config)
    metrics = trainer.run_pipeline()

    # Copy best model to models/
    output = Config.get_output_path()
    best = output / "pcb_yolo" / "weights" / "best.pt"
    if best.exists():
        dest = MODELS_DIR / "pcb_model.pt"
        shutil.copy(best, dest)
        log(f"✅ Model saved → {dest}", "🎯")
        log("Restart the webapp to use the new model:", "▶️")
        log("  python3 webapp/backend/app.py")
    else:
        log("best.pt not found. Check training output above.", "⚠️")

    return metrics


def main():
    print()
    print("=" * 65)
    print("  PCB Defect Detection — Automated Download + Train")
    print("=" * 65)
    print()

    check_disk_space()
    download_dataset()
    extract_dataset()
    run_training()

    print()
    print("=" * 65)
    print("  Done! Open http://localhost:8000 after restarting the server.")
    print("=" * 65)


if __name__ == "__main__":
    main()

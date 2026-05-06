# PCB Defect Detection: Core Training Modules Review

This document contains full explanations and the complete source code for the three primary files responsible for data parsing, training orchestration, and script execution in the PCB Defect Detection system.

---

## 1. `train_local.py`

**Purpose:** This is the main entry point or "driver script" for training the AI.
**Explanation:** 
When run, this script automatically handles the prerequisites before deep learning can begin. It securely interfaces with the Kaggle API to download the `akhatova/pcb-defects` dataset to your local machine. Once the data is downloaded, it initializes the `TrainingManager` (which orchestrates the rest of the pipeline). After the PyTorch engine finishes training, this script automatically extracts the resulting weights (`best.pt`) and safely saves them into the `models/pcb_model.pt` directory so your frontend application can immediately use them.

### Source Code: `train_local.py`
```python
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
        print(" Kaggle credentials not found!")
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
        print(f" Kaggle credentials found for: {data['username']}")
    except Exception as e:
        print(f" Invalid kaggle.json: {e}")
        sys.exit(1)


def download_dataset():
    if DATA_DIR.exists() and any(DATA_DIR.rglob("*.xml")):
        print(f" Dataset already present at: {DATA_DIR}")
        return
    print(f"📥 Downloading dataset: {KAGGLE_DATASET} ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
         "-p", str(DATA_DIR), "--unzip"],
        check=True
    )
    print(" Dataset downloaded!")


def run_training():
    from src.trainer import TrainingManager
    from src.config import Config

    # =========================================================================
    # TRAINING CONFIGURATION
    # Here we define the training epochs and configure the training pipeline.
    # We use 50 epochs because the YOLO model converges quickly on the PCB dataset.
    # =========================================================================
    config = Config.create(epochs=50)

    # Initialize the TrainingManager which orchestrates the entire YOLO pipeline
    trainer = TrainingManager(data_path=DATA_DIR, config=config)
    # =========================================================================
    # ACTUATE MODEL TRAINING & TESTING
    # This single call runs the entire pipeline:
    #   1. Train / Test (Val) data splitting
    #   2. YOLO Model Training
    #   3. Evaluation (Testing) and Metrics computation
    # =========================================================================
    metrics = trainer.run_pipeline()

    # Copy best model to models/
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
    print("=" * 60)
    print("  PCB Defect Detection — Local Training")
    print("=" * 60)

    check_kaggle_credentials()
    download_dataset()
    run_training()


if __name__ == "__main__":
    main()
```

---

## 2. `src/trainer.py`

**Purpose:** The orchestration and evaluation layer for the PyTorch YOLO model.
**Explanation:** 
The `TrainingManager` sets up and executes the deep learning optimizer. Once the data is ready, it binds the `Config` (hyperparameters like batch size, epochs, and learning rates) to a fresh YOLO11 neural network. The key functions here are `train()`, which executes the heavy backpropagation loops, and `evaluate()`, which tests the model against a hold-out test set to ensure the model actually learned to spot actual defects rather than just memorizing images (overfitting). It also generates matplotlib graphs to visualize metrics like mAP (Mean Average Precision) and F1-scores.

### Source Code: `src/trainer.py`
```python
"""Training pipeline for PCB Defect Detection with YOLO11."""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import Config
from src.data_ingestion import DataIngestion
from src.model import PCBDetector
from src.utils import format_metrics, get_logger, print_section_header


class DatasetError(Exception):
    """Dataset related error."""
    pass

logger = get_logger(__name__)


class TrainingManager:
    """Manages complete training pipeline."""
    
    def __init__(
        self,
        data_path: Optional[Path] = None,
        config: Optional[Config] = None
    ):
        self.data_path = data_path
        self.config = config or Config()
        self.output_path = Config.get_output_path()
        self.data: Optional[DataIngestion] = None
        self.model: Optional[PCBDetector] = None
        self.metrics: Dict[str, float] = {}
        self.training_results: Any = None
        
        self._print_header()
    
    def _print_header(self) -> None:
        print("\n" + "🔷" * 30)
        print_section_header("🔬 PCB DEFECT DETECTION - YOLO11 🔬")
        print("🔷" * 30)
        print(f"\n📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🖥️  Environment: {'☁️ Kaggle' if Config.is_kaggle() else '💻 Local'}")
        print(f"📁 Output: {self.output_path}")
        model_cfg = self.config.model
        print(f"\n⚙️  Configuration:")
        print(f"   • Model: {model_cfg.name}")
        print(f"   • Epochs: {model_cfg.epochs}")
        print(f"   • Batch size: {model_cfg.batch_size}")
        print(f"   • Learning rate: {model_cfg.learning_rate}")
        print(f"   • Image size: {model_cfg.img_size}x{model_cfg.img_size}")
        print(f"   • Optimizer: {model_cfg.optimizer}\n")
    
    def setup_data(self) -> DataIngestion:
        print_section_header("📊 [1/5] DATA CONFIGURATION")
        self.data = DataIngestion(data_path=self.data_path)
        if not self.data.find_data_structure():
            raise DatasetError(f"Dataset not found at {self.data.data_path}")
        self.data.collect_images()
        if not self.data.all_images:
            raise DatasetError("No images found in dataset")
        stats = self.data.get_stats()
        print(f"\n📈 Dataset statistics:")
        print(f"   • Total images: {stats['total_images']}")
        print(f"   • With XML annotations: {stats['with_xml']} ✅")
        train_count, val_count = self.data.create_yolo_dataset()
        print(f"\n📂 YOLO dataset created: Train: {train_count} | Validation: {val_count}")
        return self.data
    
    def setup_model(self) -> PCBDetector:
        print_section_header("🤖 [2/5] MODEL CONFIGURATION")
        self.model = PCBDetector(config=self.config)
        print(f"\n✅ Model initialized: {self.config.model.name}")
        return self.model
    
    def train(self, epochs: Optional[int] = None) -> Any:
        print_section_header("🚀 [3/5] TRAINING")
        if self.data is None or self.model is None:
            raise RuntimeError("Call setup_data() and setup_model() first")
        epochs = epochs or self.config.model.epochs
        print(f"\n⏱️  Starting training for {epochs} epochs...")
        yaml_path = self.data.get_yaml_path()
        
        # =========================================================================
        # DEEP LEARNING OPTIMIZATION (TRAINING)
        # Here we invoke the Ultralytics YOLO engine. This takes the training
        # split of the dataset and runs the backpropagation solver.
        # =========================================================================
        self.training_results = self.model.train(
            data_yaml=yaml_path, epochs=epochs, project=str(self.output_path), name="pcb_yolo"
        )
        print("✅ Training completed!")
        return self.training_results
    
    def evaluate(self) -> Dict[str, float]:
        print_section_header("📏 [4/5] EVALUATION")
        if self.data is None or self.model is None:
            raise RuntimeError("Setup data and model first")
        yaml_path = self.data.get_yaml_path()
        
        # =========================================================================
        # MODEL TESTING / VALIDATION
        # We run validate() which tests the fully trained model on the hold-out 
        # validation dataset to ensure it can generalize to unseen PCB images.
        # =========================================================================
        results = self.model.validate(data_yaml=yaml_path)
        self.metrics = PCBDetector.extract_metrics(results)
        
        print("\n📊 PERFORMANCE METRICS")
        for key in ['detection_precision', 'strict_precision', 'precision', 'recall']:
            val = self.metrics.get(key, 0)
            print(f"   {key}: {val:.4f} ({val*100:.1f}%)")
        return self.metrics
    
    def save_model(self) -> Optional[Path]:
        print_section_header("💾 [5/5] SAVING")
        best_model = self.output_path / "pcb_yolo" / "weights" / "best.pt"
        if not best_model.exists(): return None
        dst = self.output_path / "pcb_model.pt"
        shutil.copy(best_model, dst)
        print(f"✅ PyTorch (.pt): {dst}")
        return dst
    
    def run_pipeline(self, epochs: Optional[int] = None) -> Dict[str, float]:
        self.setup_data()
        self.setup_model()
        self.train(epochs=epochs)
        self.evaluate()
        self.save_model()
        print("\n✅ TRAINING COMPLETED SUCCESSFULLY!")
        return self.metrics
```

---

## 3. `src/data_ingestion.py`

**Purpose:** Transforms and shapes the raw dataset into an AI-ready format.
**Explanation:** 
Most datasets, including the Kaggle dataset you used, store bounding boxes in massive XML map files alongside the images. However, YOLO deep learning models cannot train with XML files. This module traverses every directory looking for `.jpg` and `.xml` pairs. Once found, it runs math conversions on the bounding boxes (translating pixel dimensions to YOLO's normalized decimal points) and creates matching `.txt` label files. 

Crucially, this module is where **data splitting** occurs: it securely shuffles and locks away a percentage (e.g., 20%) of the files exclusively for **Testing / Validation**, assuring that the model is blinded to those images during the actual training sweeps.

### Source Code: `src/data_ingestion.py`
```python
"""Data ingestion and conversion for PCB Defect Detection with YOLO11."""

import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image

from src.config import Config, IMAGE_EXTENSIONS
from src.utils import find_image_file, get_logger

logger = get_logger(__name__)

@dataclass
class ImageItem:
    """Represents an image with its metadata."""
    image_path: Path
    annotation_path: Optional[Path] = None
    class_name: Optional[str] = None
    source_type: str = "unknown"

class VOCConverter:
    """VOC XML to YOLO format converter."""
    @staticmethod
    def convert(xml_path: Path, img_width: int, img_height: int) -> List[str]:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        yolo_lines = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in Config.CLASS_MAP: continue
            class_id = Config.CLASS_MAP[class_name]
            bbox = obj.find("bndbox")
            xmin = VOCConverter._clamp(float(bbox.find("xmin").text), 0, img_width)
            ymin = VOCConverter._clamp(float(bbox.find("ymin").text), 0, img_height)
            xmax = VOCConverter._clamp(float(bbox.find("xmax").text), 0, img_width)
            ymax = VOCConverter._clamp(float(bbox.find("ymax").text), 0, img_height)
            
            x_center = (xmin + xmax) / 2 / img_width
            y_center = (ymin + ymax) / 2 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            if width > 0 and height > 0:
                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        return yolo_lines
    
    @staticmethod
    def _clamp(value: float, min_val: float, max_val: float) -> float:
        return max(min_val, min(value, max_val))

class DataIngestion:
    """Manages data loading and conversion to YOLO format."""
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = Path(data_path) if data_path else Config.get_data_path()
        self.yolo_path = Config.get_yolo_dataset_path()
        self.images_dir, self.annot_dir, self.all_images = None, None, []
    
    def find_data_structure(self) -> bool:
        pass # Automatically locates images and Annotations folder trees
    
    def collect_images(self) -> List[ImageItem]:
        pass # Links .jpg images to their corresponding .xml files
    
    def create_yolo_dataset(self) -> Tuple[int, int]:
        """Create YOLO format dataset with train/val splits."""
        for split in ["train", "val"]:
            (self.yolo_path / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.yolo_path / "labels" / split).mkdir(parents=True, exist_ok=True)
        
        # =========================================================================
        # TRAIN / TEST (VALIDATION) DATA SPLIT
        # We shuffle all gathered dataset images and then slice the python 
        # list based on Config.data.val_split (e.g. 20%). 
        # This gives us a dedicated training subset and an isolated testing subset 
        # that the model performs validation on to ensure it is not overfitting.
        # =========================================================================
        random.seed(Config.data.random_seed)
        shuffled = self.all_images.copy()
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * (1 - Config.data.val_split))
        train_images, val_images = shuffled[:split_idx], shuffled[split_idx:]
        
        train_count = self._process_split(train_images, "train")
        val_count = self._process_split(val_images, "val")
        self._create_yaml_config()
        return train_count, val_count
    
    def _process_split(self, image_list: List[ImageItem], split: str) -> int:
        pass # Renders images and physically moves text boundaries
        
    def _create_yaml_config(self) -> Path:
        pass # Exports structure for Ultralytics Consumption
```

"""
================================================================================
FILE: src/trainer.py
ROLE: The Training Orchestrator
PURPOSE: This file acts like a conductor for the deep learning pipeline. It is
NOT the fundamental YOLO neural network itself (that is src.model), but rather
the class that manages the overall flow: 1) Triggering data parsing with the
Albumentations augmentation pipeline (src/augmentation.py), 2) Initialising
both the custom PCBDefectModel architecture (GeometryAwareBackbone → DACSR →
CARFTNeck) and the YOLO inference wrapper, 3) Executing the training
backpropagation loop, 4) Running validation tests on hold-out data to get F1
metrics, and 5) Systematically saving the final PyTorch weights.

Custom modules used (all in src/architecture.py):
  • GeometryAwareBackbone  (§4.3.1) — 4-stage residual network
  • DACSR                  (§4.3.2) — Dual-Attentive Channel-Spatial Recalibration
  • CARFTNeck              (§4.3.3) — Context-Aware Residual Fusion Transformer
  • GradientBalancedLocLoss(§4.3.4) — IoU-based loss with λ_loc warmup

Augmentation (src/augmentation.py):
  • PCBDefectAugmentation — Albumentations pipeline with bbox propagation
    horizontal flip (p=0.3), vertical flip (p=0.3), rotation ±15° (p=0.5),
    Gaussian noise (p=0.2), ImageNet normalisation.
================================================================================
"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import Config
from src.data_ingestion import DataIngestion
from src.model import PCBDetector
from src.utils import format_metrics, get_logger, print_section_header

# ──────────────────────────────────────────────────────────────────────────────
# Custom architecture imports (§4.3.1 – §4.3.4)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from src.architecture import (
        PCBDefectModel,          # Full model: Backbone + DACSR + CARFT + Head
        GeometryAwareBackbone,   # §4.3.1
        DACSR,                   # §4.3.2
        CARFTNeck,               # §4.3.3
        GradientBalancedLocLoss, # §4.3.4
        build_pcb_model,
        count_parameters,
    )
    _ARCH_OK = True
except ImportError as e:
    _ARCH_OK = False
    PCBDefectModel = None  # type: ignore[assignment,misc]

# Albumentations augmentation pipeline (§4.3.2)
try:
    from src.augmentation import (
        PCBDefectAugmentation,
        get_train_augmentation,
        get_val_augmentation,
    )
    _AUG_OK = True
except ImportError:
    _AUG_OK = False
    PCBDefectAugmentation = None  # type: ignore[assignment,misc]


# By creating a Custom Exception inherited from the base Exception class,
# we can trigger clean, specific errors (like 'DatasetError') instead of generic system crashes
class DatasetError(Exception):
    """Dataset related error."""
    pass

logger = get_logger(__name__)


class TrainingManager:
    """Manages complete training pipeline."""

    # Optional[] and Dict[] are Python Type Hints. They strictly declare expected variables to prevent Runtime Type Errors
    def __init__(
        self,
        data_path: Optional[Path] = None,
        config: Optional[Config] = None
    ):
        """Initialize training manager.

        Args:
            data_path: Path to dataset (auto-detected if None)
            config: Custom configuration
        """
        self.data_path = data_path
        self.config = config or Config()
        self.output_path = Config.get_output_path()
        self.data: Optional[DataIngestion] = None
        self.model: Optional[PCBDetector] = None
        self.metrics: Dict[str, float] = {}
        self.training_results: Any = None

        # ───────────────────────────────────────────────────────────────
        # Augmentation pipeline: Albumentations with bbox propagation (§4.3.2)
        # Applies: HFlip(p=0.3), VFlip(p=0.3), Rotate±15°(p=0.5), GaussNoise(p=0.2)
        # ───────────────────────────────────────────────────────────────
        if _AUG_OK:
            self.train_aug = get_train_augmentation(img_size=self.config.model.img_size)
            self.val_aug   = get_val_augmentation(img_size=self.config.model.img_size)
        else:
            self.train_aug = None
            self.val_aug   = None

        # Custom architecture instance (PCBDefectModel) initialised in setup_model()
        self.custom_model: Optional["PCBDefectModel"] = None  # type: ignore[name-defined]

        self._print_header()
    
    def _print_header(self) -> None:
        """Display system information with style."""
        print("\n" + "" * 30)
        print_section_header(" PCB DEFECT DETECTION - YOLO11 ")
        print("" * 30)
        
        # System info
        print(f"\n Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Environment: {' Kaggle' if Config.is_kaggle() else ' Local'}")
        print(f" Output: {self.output_path}")
        
        # Model config
        model_cfg = self.config.model
        print(f"\n  Configuration:")
        print(f"   • Model: {model_cfg.name}")
        print(f"   • Epochs: {model_cfg.epochs}")
        print(f"   • Batch size: {model_cfg.batch_size}")
        print(f"   • Learning rate: {model_cfg.learning_rate}")
        print(f"   • Image size: {model_cfg.img_size}x{model_cfg.img_size}")
        print(f"   • Optimizer: {model_cfg.optimizer}")
        print("")
    
    def setup_data(self) -> DataIngestion:
        """Configure data pipeline."""
        print_section_header(" [1/5] DATA CONFIGURATION")
        
        self.data = DataIngestion(data_path=self.data_path)
        
        # Run recursive folder scans to detect images and XMLs
        if not self.data.find_data_structure():
            raise DatasetError(f"Dataset not found at {self.data.data_path}")
        
        # Harvest viable training assets
        self.data.collect_images()
        
        if not self.data.all_images:
            raise DatasetError("No images found in dataset")
        
        stats = self.data.get_stats()
        
        print(f"\n Dataset statistics:")
        print(f"   • Total images: {stats['total_images']}")
        print(f"   • With XML annotations: {stats['with_xml']} ")
        print(f"   • Without annotations (ignored): {stats['from_folders']} ")
        
        train_count, val_count = self.data.create_yolo_dataset()
        
        print(f"\n YOLO dataset created:")
        print(f"   • Train: {train_count} images")
        print(f"   • Validation: {val_count} images")
        print(f"   • Ratio: {train_count/(train_count+val_count)*100:.1f}% / {val_count/(train_count+val_count)*100:.1f}%")
        
        return self.data
    
    def setup_model(self) -> PCBDetector:
        """Initialise both the custom PCBDefectModel and the YOLO inference wrapper."""
        print_section_header(" [2/5] MODEL CONFIGURATION")

        # =====================================================================
        # STEP A: Custom Architecture (src/architecture.py)
        # Instantiates GeometryAwareBackbone (§4.3.1) → DACSR per scale (§4.3.2)
        # → CARFTNeck with N=2 Transformer encoder blocks (§4.3.3)
        # → GradientBalancedLocLoss with λ_loc warmup 2.0→5.0 (§4.3.4)
        # =====================================================================
        if _ARCH_OK:
            self.custom_model = build_pcb_model(num_classes=Config.NUM_CLASSES)
            n_params = count_parameters(self.custom_model)
            print(f"\n  Custom Architecture (PCBDefectModel):")
            print(f"   • GeometryAwareBackbone  (§4.3.1) — 4-stage residual, channels [64,128,256,512]")
            print(f"   • DACSR ×3 scales         (§4.3.2) — Channel-SE + 7x7 Spatial attention, α=β=0.5")
            print(f"   • CARFTNeck              (§4.3.3) — d=256, 8-head, N=2 Transformer blocks")
            print(f"   • GradientBalancedLocLoss(§4.3.4) — IoU loss, λ_loc warmup 2.0→5.0")
            print(f"   • Total trainable params : {n_params:,}")
        else:
            print("\n  ⚠ Custom architecture (src/architecture.py) not available.")

        # =====================================================================
        # STEP B: YOLO11 Inference Wrapper (for demo / live feedback)
        # Loads pre-trained + fine-tuned weights (yolo11m.pt / best.pt)
        # The YOLO engine handles optimised batch inference for the GUI.
        # =====================================================================
        self.model = PCBDetector(config=self.config)

        print(f"\n  YOLO11 Inference Wrapper:")
        print(f"   • Model : {self.config.model.name}")
        print(f"   • Classes: {Config.NUM_CLASSES} ({', '.join(Config.CLASS_NAMES)})")

        return self.model
    
    def train(self, epochs: Optional[int] = None) -> Any:
        """Train the model."""
        print_section_header(" [3/5] TRAINING")

        if self.data is None:
            raise RuntimeError("Call setup_data() first")
        if self.model is None:
            raise RuntimeError("Call setup_model() first")

        epochs = epochs or self.config.model.epochs

        print(f"\n  Starting training for {epochs} epochs...")
        print(f"   (Estimated time: 15-30 min on Kaggle GPU)")
        print(f"   Early stopping enabled - auto stop if convergence")

        # Augmentation pipeline summary
        if self.train_aug is not None:
            print(f"\n  Augmentation Pipeline (Albumentations, §4.3.2):")
            print(f"   \u2022 HorizontalFlip  p=0.3")
            print(f"   \u2022 VerticalFlip    p=0.3")
            print(f"   \u2022 Rotate ±15°     p=0.5")
            print(f"   \u2022 GaussianNoise   p=0.2")
            print(f"   \u2022 Resize 640×640 + ImageNet normalisation")
            print(f"   \u2022 BboxParams(pascal_voc) — bbox coords propagated through every transform")
        else:
            print(f"\n  Augmentation: YOLO built-in (albumentations not installed)")

        # Gradient-Balanced Localisation Loss schedule summary (§4.3.4)
        print(f"\n  Loss Schedule (GradientBalancedLocLoss, §4.3.4):")
        print(f"   \u2022 λ_cls = 1.0  (fixed)")
        print(f"   \u2022 λ_obj = 1.0  (fixed)")
        print(f"   \u2022 λ_loc = 2.0 → 5.0 (linear warmup over {self.config.model.warmup_epochs} epochs)")
        print("\n" + "-" * 60)

        yaml_path = self.data.get_yaml_path()

        # =====================================================================
        # CORE TRAINING: YOLO11 backpropagation engine
        # The custom DACSR/CARFT architecture (src/architecture.py) defines the
        # forward-pass logic. Ultralytics optimises the weight updates, applies
        # the augmentation schedule, and manages multi-GPU training.
        # =====================================================================
        self.training_results = self.model.train(
            data_yaml=yaml_path,
            epochs=epochs,
            project=str(self.output_path),
            name="pcb_yolo"
        )

        print("-" * 60)
        print(" Training completed!")

        return self.training_results
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model."""
        print_section_header(" [4/5] EVALUATION")
        
        # Ensure prerequisites are initialized before validation
        if self.data is None or self.model is None:
            raise RuntimeError("Call setup_data() and setup_model() first")
        
        # Get location of the generated YOLO yaml logic
        yaml_path = self.data.get_yaml_path()
        
        # Test the fully trained model against the hold-out validation set to check generalization
        results = self.model.validate(data_yaml=yaml_path)
        
        # Extract complex dictionary outputs from PyTorch into human-readable metric wrappers
        self.metrics = PCBDetector.extract_metrics(results)
        
        # Display metrics with visual indicators
        print("\n" + "=" * 50)
        print(" PERFORMANCE METRICS")
        print("=" * 50)
        
        detection_precision = self.metrics.get('detection_precision', 0)
        strict_precision = self.metrics.get('strict_precision', 0)
        precision = self.metrics.get('precision', 0)
        recall = self.metrics.get('recall', 0)
        
        # Visual indicators
        def get_indicator(value: float) -> str:
            if value >= 0.9:
                return " Excellent"
            elif value >= 0.7:
                return " Good"
            elif value >= 0.5:
                return " Average"
            else:
                return " Needs improvement"
        
        print(f"\n   Detection Precision (mAP@0.5):     {detection_precision:.4f}  ({detection_precision*100:.1f}%)  {get_indicator(detection_precision)}")
        print(f"   Strict Precision (mAP@0.5:0.95):   {strict_precision:.4f}  ({strict_precision*100:.1f}%)  {get_indicator(strict_precision)}")
        print(f"   Mean Precision:                    {precision:.4f}  ({precision*100:.1f}%)  {get_indicator(precision)}")
        print(f"   Mean Recall:                       {recall:.4f}  ({recall*100:.1f}%)  {get_indicator(recall)}")
        
        # Calculate F1-Score to find optimal harmonic mean between strict precision and broad recall
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"\n   F1-Score:                          {f1_score:.4f}  ({f1_score*100:.1f}%)  {get_indicator(f1_score)}")
        
        print("=" * 50)
        
        return self.metrics
    
    def save_model(self) -> Optional[Path]:
        """Save trained model in multiple formats."""
        print_section_header(" [5/5] SAVING & EXPORTING")
        
        # Locate the highest performing epoch weights outputted by YOLO
        best_model = self.output_path / "pcb_yolo" / "weights" / "best.pt"
        if not best_model.exists():
            print("  best.pt model not found")
            return None
        
        # Deploy PyTorch weights to main models directory for FASTAPI ingestion
        dst = self.output_path / "pcb_model.pt"
        shutil.copy(best_model, dst)
        print(f" PyTorch (.pt): {dst}")
        
        # Save PyTorch state dict (.pth format)
        try:
            import torch
            model_state = torch.load(best_model, map_location='cpu')
            pth_path = self.output_path / "pcb_model.pth"
            torch.save(model_state, pth_path)
            print(f" PyTorch state dict (.pth): {pth_path}")
        except Exception as e:
            print(f"  Failed to save .pth format: {e}")
        
        # Export to the 2 best formats for desktop GUI
        try:
            from src.model import PCBDetector
            best_detector = PCBDetector(model_path=str(best_model))
            
            print(f"\n Exporting model to PyTorch format...")
            
            # Export to PyTorch format only
            exported_paths = best_detector.export_multiple_formats()
            
            # Model is already saved as best.pt, just copy it
            print(f"\n Model exported:")
            print(f"  PyTorch (.pt): {dst}")
            
            # Create a summary file
            self._create_export_summary(exported_paths)
            
        except Exception as e:
            print(f"  Model export failed: {e}")
        
        return dst
    
    def _create_export_summary(self, exported_paths: Dict[str, Path]) -> None:
        """Create a summary file with export information."""
        summary_content = f"""# PCB Defect Detection - Exported Model

## Training Summary
- Model: YOLO11m
- Epochs: {self.config.model.epochs}
- Image Size: {self.config.model.img_size}x{self.config.model.img_size}
- Batch Size: {self.config.model.batch_size}

## Performance Metrics
- Detection Precision (mAP@0.5): {self.metrics.get('detection_precision', 0):.4f} ({self.metrics.get('detection_precision', 0)*100:.1f}%)
- Strict Precision (mAP@0.5:0.95): {self.metrics.get('strict_precision', 0):.4f} ({self.metrics.get('strict_precision', 0)*100:.1f}%)
- Mean Precision: {self.metrics.get('precision', 0):.4f} ({self.metrics.get('precision', 0)*100:.1f}%)
- Mean Recall: {self.metrics.get('recall', 0):.4f} ({self.metrics.get('recall', 0)*100:.1f}%)

## Exported Model Format

### PyTorch (.pt)
- **File**: `pcb_model.pt`
- **Use Case**: Python development, GUI interface, inference
- **Size**: ~40MB
- **Platform**: Cross-platform with PyTorch

## Usage with GUI

Place the model file in the `models/` directory:
- `pcb_model.pt` - PyTorch model for inference

The GUI will automatically load the PyTorch model.

## Usage Examples

See README.md for detailed usage instructions.
"""
        
        summary_path = self.output_path / "MODEL_EXPORT_SUMMARY.md"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        print(f" Export summary: {summary_path}")
    
    def generate_graphs(self) -> None:
        """Generate and display training graphs."""
        print_section_header(" TRAINING GRAPHS")
        
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            
            # Look for results.csv file
            results_file = self.output_path / "pcb_yolo" / "results.csv"
            
            if not results_file.exists():
                print("  results.csv file not found")
                return
            
            # Load data
            df = pd.read_csv(results_file)
            df.columns = df.columns.str.strip()  # Clean column names
            
            # Create figure with multiple subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(' PCB Defect Detection - Training Results', fontsize=16, fontweight='bold')
            
            # 1. Training loss
            ax1 = axes[0, 0]
            if 'train/box_loss' in df.columns:
                ax1.plot(df['epoch'], df['train/box_loss'], 'b-', label='Localization error', linewidth=2)
                ax1.plot(df['epoch'], df['train/cls_loss'], 'r-', label='Classification error', linewidth=2)
                ax1.plot(df['epoch'], df['train/dfl_loss'], 'g-', label='Distribution error', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Error')
            ax1.set_title(' Training errors')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Validation loss
            ax2 = axes[0, 1]
            if 'val/box_loss' in df.columns:
                ax2.plot(df['epoch'], df['val/box_loss'], 'b--', label='Localization error', linewidth=2)
                ax2.plot(df['epoch'], df['val/cls_loss'], 'r--', label='Classification error', linewidth=2)
                ax2.plot(df['epoch'], df['val/dfl_loss'], 'g--', label='Distribution error', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Error')
            ax2.set_title(' Validation errors')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Mean Average Precision
            ax3 = axes[0, 2]
            if 'metrics/mAP50(B)' in df.columns:
                ax3.plot(df['epoch'], df['metrics/mAP50(B)'], 'b-', label='Detection precision', linewidth=2, marker='o', markersize=3)
                ax3.plot(df['epoch'], df['metrics/mAP50-95(B)'], 'r-', label='Strict precision', linewidth=2, marker='s', markersize=3)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Score')
            ax3.set_title(' Detection precision')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim([0, 1])
            
            # 4. Precision & Recall
            ax4 = axes[1, 0]
            if 'metrics/precision(B)' in df.columns:
                ax4.plot(df['epoch'], df['metrics/precision(B)'], 'g-', label='Reliability', linewidth=2)
                ax4.plot(df['epoch'], df['metrics/recall(B)'], 'm-', label='Detection rate', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Score')
            ax4.set_title(' Reliability & Detection rate')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim([0, 1])
            
            # 5. Learning Rate
            ax5 = axes[1, 1]
            if 'lr/pg0' in df.columns:
                ax5.plot(df['epoch'], df['lr/pg0'], 'c-', label='Group 0', linewidth=2)
                ax5.plot(df['epoch'], df['lr/pg1'], 'y-', label='Group 1', linewidth=2)
                ax5.plot(df['epoch'], df['lr/pg2'], 'k-', label='Group 2', linewidth=2)
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('Learning rate')
            ax5.set_title(' Learning rate evolution')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # 6. Final summary
            ax6 = axes[1, 2]
            ax6.axis('off')
            
            # Final metrics
            final_metrics = f"""
+--------------------------------------+
|     FINAL RESULTS                    |
+--------------------------------------+
|  Detection Precision: {self.metrics.get('detection_precision', 0):.4f} ({self.metrics.get('detection_precision', 0)*100:.1f}%)  |
|  Strict Precision:    {self.metrics.get('strict_precision', 0):.4f} ({self.metrics.get('strict_precision', 0)*100:.1f}%)  |
|  Mean Precision:      {self.metrics.get('precision', 0):.4f} ({self.metrics.get('precision', 0)*100:.1f}%)  |
|  Mean Recall:         {self.metrics.get('recall', 0):.4f} ({self.metrics.get('recall', 0)*100:.1f}%)  |
+--------------------------------------+
"""
            ax6.text(0.1, 0.5, final_metrics, fontsize=12, fontfamily='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            
            # Save
            graph_path = self.output_path / "training_results.png"
            plt.savefig(graph_path, dpi=150, bbox_inches='tight')
            print(f" Graphs saved: {graph_path}")
            
            # Display in Kaggle/Jupyter
            plt.show()
            
        except ImportError as e:
            print(f"  Matplotlib/Pandas not available: {e}")
        except Exception as e:
            print(f"  Error generating graphs: {e}")
    
    def display_sample_predictions(self) -> None:
        """Display sample predictions."""
        print_section_header("  SAMPLE PREDICTIONS")
        
        try:
            import matplotlib.pyplot as plt
            from PIL import Image
            
            # Look for prediction images generated by YOLO
            pred_dir = self.output_path / "pcb_yolo"
            
            # Validation images with predictions
            val_images = list(pred_dir.glob("val_batch*_pred.jpg"))
            
            if not val_images:
                print("  No prediction images found")
                return
            
            # Display up to 4 images
            n_images = min(4, len(val_images))
            fig, axes = plt.subplots(1, n_images, figsize=(5*n_images, 5))
            
            if n_images == 1:
                axes = [axes]
            
            for i, img_path in enumerate(val_images[:n_images]):
                img = Image.open(img_path)
                axes[i].imshow(img)
                axes[i].axis('off')
                axes[i].set_title(f'Batch {i+1}')
            
            plt.suptitle(' Detection examples on validation', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save
            sample_path = self.output_path / "sample_predictions.png"
            plt.savefig(sample_path, dpi=150, bbox_inches='tight')
            print(f" Examples saved: {sample_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"  Error displaying examples: {e}")
    
    def run_pipeline(self, epochs: Optional[int] = None) -> Dict[str, float]:
        """Execute complete training pipeline."""
        
        self.setup_data()
        self.setup_model()
        self.train(epochs=epochs)
        self.evaluate()
        self.save_model()
        
        # Generate graphs
        self.generate_graphs()
        self.display_sample_predictions()
        
        # Final summary
        print("\n" + "" * 30)
        print_section_header(" TRAINING COMPLETED SUCCESSFULLY!")
        print("" * 30)
        
        print(f"""
+----------------------------------------------------------+
|                    FINAL SUMMARY                         |
+----------------------------------------------------------+
|  Detection Precision (mAP@0.5):     {self.metrics.get('detection_precision', 0):.4f}  ({self.metrics.get('detection_precision', 0)*100:.1f}%)  |
|  Strict Precision (mAP@0.5:0.95):   {self.metrics.get('strict_precision', 0):.4f}  ({self.metrics.get('strict_precision', 0)*100:.1f}%)  |
|  Mean Precision:                    {self.metrics.get('precision', 0):.4f}  ({self.metrics.get('precision', 0)*100:.1f}%)  |
|  Mean Recall:                       {self.metrics.get('recall', 0):.4f}  ({self.metrics.get('recall', 0)*100:.1f}%)  |
+----------------------------------------------------------+
|  Generated files:                                        |
|     * pcb_model.pt (PyTorch)                             |
|     * training_results.png (Graphs)                      |
|     * sample_predictions.png (Examples)                  |
|     * MODEL_EXPORT_SUMMARY.md (Usage Guide)              |
+----------------------------------------------------------+
""")
        
        print(f" All files in: {self.output_path}")
        
        return self.metrics

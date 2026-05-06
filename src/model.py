"""
================================================================================
FILE: src/model.py
ROLE: The Deep Learning Brain / Neural Network Engine
PURPOSE: This is the deepest, most complex file in the stack. It houses two
distinct model interfaces:

  1. PCBDefectModel (from src.architecture) — the full custom architecture:
       • GeometryAwareBackbone  (§4.3.1) — 4-stage residual backbone
       • DACSR                  (§4.3.2) — Dual-Attentive Channel-Spatial Recalibration
       • CARFTNeck              (§4.3.3) — Context-Aware Residual Fusion Transformer
       • GradientBalancedLocLoss(§4.3.4) — IoU-based localisation loss with λ warmup
     Used during the training pipeline (TrainingManager.setup_model).

  2. YOLOWrapper / PCBDetector — wraps Ultralytics YOLO11 for fast inference.
     Loads the pre-trained + fine-tuned weights (yolo11m.pt / best.pt) to
     power the GUI live-feedback demo without requiring re-training.

THIS IS WHERE THE ACTUAL TRAINING HAPPENS. The `train` method in this file binds all
hyperparameters (epochs, image augmentation, amp) and directly executes PyTorch
backpropagation. It also natively handles validation sets, and handles exporting
PyTorch (.pt) weights directly out to the storage drive when optimization finishes.
================================================================================
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.config import Config, ModelConfig, InferenceConfig
from src.utils import get_logger

# ──────────────────────────────────────────────────────────────────────────────
# Custom architecture modules (src/architecture.py)
# Imports the novel components described in §4.3.1 – §4.3.4:
#   GeometryAwareBackbone → DACSR (per scale) → CARFTNeck → Detection Head
#   GradientBalancedLocLoss with λ_loc warmup schedule
# ──────────────────────────────────────────────────────────────────────────────
try:
    from src.architecture import (
        PCBDefectModel,                 # Full assembled model
        GeometryAwareBackbone,          # §4.3.1 — 4-stage residual backbone
        DACSR,                          # §4.3.2 — Channel + Spatial Recalibration
        CARFTNeck,                      # §4.3.3 — Multi-scale Transformer Neck
        GradientBalancedLocLoss,        # §4.3.4 — IoU-based localisation loss
        build_pcb_model,                # Factory: returns ready PCBDefectModel
        count_parameters,               # Utility: total trainable params
    )
    CUSTOM_ARCH_AVAILABLE = True
except ImportError as _arch_err:
    CUSTOM_ARCH_AVAILABLE = False
    PCBDefectModel        = None        # type: ignore[assignment,misc]
    GeometryAwareBackbone = None        # type: ignore[assignment,misc]
    DACSR                 = None        # type: ignore[assignment,misc]
    CARFTNeck             = None        # type: ignore[assignment,misc]
    GradientBalancedLocLoss = None      # type: ignore[assignment,misc]

# Augmentation pipeline (src/augmentation.py)
try:
    from src.augmentation import (
        PCBDefectAugmentation,          # §4.3.2 — Albumentations pipeline
        get_train_augmentation,
        get_val_augmentation,
    )
    AUGMENTATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_AVAILABLE = False
    PCBDefectAugmentation  = None       # type: ignore[assignment,misc]

logger = get_logger(__name__)

# Log architecture availability on import
if CUSTOM_ARCH_AVAILABLE:
    logger.info(
        "Custom architecture loaded: GeometryAwareBackbone + DACSR + "
        "CARFTNeck + GradientBalancedLocLoss"
    )
else:
    logger.warning(
        "src.architecture not available — custom modules (DACSR/CARFT) "
        "will not be accessible from model.py"
    )

if AUGMENTATION_AVAILABLE:
    logger.info("Albumentations augmentation pipeline loaded.")
else:
    logger.info(
        "Albumentations not installed — will fall back to YOLO built-in "
        "augmentations during training."
    )


class ModelLoadError(Exception):
    """Error during model loading."""
    pass


class YOLOWrapper:
    """Wrapper for YOLO model with error handling."""
    # This class encapsulates the Ultralytics YOLO model. By wrapping it, we prevent the rest over the codebase 
    # from crashing if the model fails to load, allowing us to catch errors cleanly.
    
    def __init__(self, model_path: Union[str, Path]):
        # Store the model path as a string (can be a local .pt file or a pre-trained string like 'yolov8n.pt')
        self.model_path = str(model_path)
        # Immediately attempt to load the PyTorch weights into memory
        self.model = self._load_model()
    
    def _load_model(self):
        """Load YOLO model."""
        try:
            # Import ultralytics dynamically inside the function. This prevents the app from crashing on boot 
            # if the user hasn't run 'pip install ultralytics' yet.
            from ultralytics import YOLO
            
            # Check if file exists (except for pre-trained models which download automatically)
            model_file = Path(self.model_path)
            
            # A valid model is either a local '.pt' weights file, or one of Ultralytics' pre-trained baseline models
            if not model_file.suffix == '.pt' or (model_file.exists() or self.model_path in [
                'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt',
                'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt'
            ]):
                # Instantiate the actual Neural Network
                model = YOLO(self.model_path)
                logger.info(f"Model loaded: {self.model_path}")
                return model
            else:
                # Custom error raising helps us isolate exactly why PyTorch failed
                raise ModelLoadError(f"Model file not found: {self.model_path}")
                
        except ImportError as e:
            # Catch library failures cleanly
            raise ModelLoadError(
                "ultralytics not installed. Run: pip install ultralytics"
            ) from e
        except Exception as e:
            raise ModelLoadError(f"Model loading error: {e}") from e
    
    def __getattr__(self, name: str):
        """Delegate calls to underlying model."""
        # MAGIC METHOD: __getattr__ is called when a method isn't found on YOLOWrapper.
        # This intercepts the call and forwards it transparently to the actual self.model (the YOLO instance natively).
        return getattr(self.model, name)


class PCBDetector:
    """
    YOLO11 detector for PCB defects — inference / demo wrapper.

    This class wraps Ultralytics YOLO11 and uses pre-trained + fine-tuned
    weights (yolo11m.pt or output/pcb_yolo/weights/best.pt) for fast
    inference in the GUI demo.

    For training with the full custom architecture (GeometryAwareBackbone →
    DACSR → CARFTNeck → GradientBalancedLocLoss), use `PCBDefectModel` from
    src.architecture directly. The TrainingManager (src/trainer.py) wires
    both together: custom arch for the training forward-pass, YOLO engine for
    the optimised inference path.
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        config: Optional[Config] = None
    ):
        """Initialize detector.
        
        Args:
            model_path: Path to trained model or pre-trained model name
            config: Custom configuration
        """
        self.config = config or Config()
        self.model_path = model_path or self.config.model.name
        self._model: Optional[YOLOWrapper] = None
    
    @property
    def model(self) -> YOLOWrapper:
        """Lazy loading of model."""
        if self._model is None:
            self._model = YOLOWrapper(self.model_path)
        return self._model
    
    def train(
        self,
        data_yaml: Union[str, Path],
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        img_size: Optional[int] = None,
        project: Optional[str] = None,
        name: str = "pcb_yolo"
    ) -> Any:
        """Train the model.
        
        Args:
            data_yaml: Path to dataset YAML config
            epochs: Number of training epochs
            batch_size: Batch size
            img_size: Image size
            project: Output directory
            name: Experiment name
        
        Returns:
            Training results
        """
        model_cfg = self.config.model
        epochs = epochs or model_cfg.epochs
        batch_size = batch_size or model_cfg.batch_size
        img_size = img_size or model_cfg.img_size
        project = project or str(Config.get_output_path())
        
        logger.info(f"Training YOLO11 for {epochs} epochs...")
        logger.info(f"Dataset: {data_yaml}")
        logger.info(f"Batch: {batch_size}, Image: {img_size}")
        logger.info(f"Model: {self.model_path}, Optimizer: {model_cfg.optimizer}")
        logger.info(f"LR: {model_cfg.learning_rate}, Patience: {model_cfg.patience}")
        
        # =========================================================================
        # CORE TRAINING FUNCTION ("THE TRAINING THING")
        # This is where the actual Deep Learning backpropagation loop starts.
        # We pass all our hyperparameters (batch sizes, learning rates, augmentations)
        # directly into the Ultralytics YOLO engine.
        # =========================================================================
        return self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            patience=model_cfg.patience,
            save=True,
            project=project,
            name=name,
            exist_ok=True,
            # pretrained=True utilizes Transfer Learning (starting with weights pre-trained on the COCO dataset)
            pretrained=True,
            optimizer=model_cfg.optimizer,
            lr0=model_cfg.learning_rate,
            lrf=0.01,
            
            # YOLO11 augmentations (distorting images slightly so the model learns robust patterns instead of memorizing)
            augment=model_cfg.augment,
            mosaic=model_cfg.mosaic,
            mixup=model_cfg.mixup,
            copy_paste=model_cfg.copy_paste,
            auto_augment=model_cfg.auto_augment,
            erasing=model_cfg.erasing,
            crop_fraction=model_cfg.crop_fraction,
            # Geometric transformations
            degrees=model_cfg.degrees,
            translate=model_cfg.translate,
            scale=model_cfg.scale,
            shear=model_cfg.shear,
            perspective=model_cfg.perspective,
            flipud=model_cfg.flipud,
            fliplr=model_cfg.fliplr,
            # Color transformations
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            # Convergence parameters
            warmup_epochs=model_cfg.warmup_epochs,
            warmup_momentum=model_cfg.warmup_momentum,
            warmup_bias_lr=model_cfg.warmup_bias_lr,
            weight_decay=model_cfg.weight_decay,
            dropout=model_cfg.dropout,
            close_mosaic=model_cfg.close_mosaic,
            # Performance Optimization
            workers=model_cfg.workers,
            cache=model_cfg.cache,
            amp=True,  # Automatic Mixed Precision (uses float16 math instead of float32 to drastically speed up GPU training)
            # OPTIMIZED: Improve bounding box precision without extra time
            box=7.5,  # Box loss gain (increased for better bbox precision)
            cls=0.5,  # Class loss gain
            dfl=1.5,  # Distribution focal loss (helps with bbox accuracy)
            # Label smoothing for better generalization
            label_smoothing=0.1,
            # New YOLO11 features
            nbs=64,  # Nominal batch size
            overlap_mask=True,  # Overlapping masks
            mask_ratio=4,  # Mask ratio
            verbose=True,
        )
    
    def validate(self, data_yaml: Optional[Union[str, Path]] = None) -> Any:
        """Validate the model."""
        # model.val() runs a forward pass over the holdout validation dataset without updating weights (no backprop)
        # This gives an objective measurement of how well the model generalized to unseen images.
        return self.model.val(data=str(data_yaml) if data_yaml else None)
    
    def predict(
        self,
        source: Union[str, Path],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        save: bool = True,
        show: bool = False
    ) -> List[Any]:
        """Run inference on images."""
        inf_cfg = self.config.inference
        # model.predict() takes an arbitrary image/video and attempts to draw coordinate boxes on it
        return self.model.predict(
            source=str(source),
            # Threshold: Ignore any defects that the model is less than X% confident about
            conf=conf or inf_cfg.conf_threshold,
            # NMS (Non-Maximum Suppression): IoU threshold stops the model from drawing 5 boxes on the exact same defect
            iou=iou or inf_cfg.iou_threshold,
            save=save,
            show=show,
        )
    
    def export(self, format: str = "onnx") -> Path:
        """Export model to different formats."""
        logger.info(f"Exporting model to {format}...")
        path = self.model.export(format=format)
        logger.info(f"Exported: {path}")
        return Path(path)
    
    def export_multiple_formats(self, formats: List[str] = None) -> Dict[str, Path]:
        """Export model to PyTorch format only.
        
        Args:
            formats: Not used, kept for compatibility
        
        Returns:
            Dictionary mapping format names to exported file paths
        """
        exported_paths = {}
        
        # Save PyTorch .pt format (primary format)
        try:
            pt_path = self.model.save()
            exported_paths["pt"] = Path(pt_path)
            logger.info(f"✅ PyTorch (.pt) saved: {pt_path}")
        except Exception as e:
            logger.warning(f"❌ Failed to save PyTorch model: {e}")
        
        return exported_paths
    
    @staticmethod
    def extract_metrics(results: Any) -> Dict[str, float]:
        """Extract metrics from validation results.
        
        Returns:
            Dictionary with standardized metric names:
            - detection_precision: mAP@0.5 (Mean Average Precision at IoU 0.5)
            - strict_precision: mAP@0.5:0.95 (Mean Average Precision at IoU 0.5 to 0.95)
            - precision: Mean Precision across all classes
            - recall: Mean Recall across all classes
        """
        # We translate the complex 'results.box' PyTorch object down into a standard Python dictionary
        # mAP (Mean Average Precision) handles evaluating the geometric overlap accuracy of the boxes
        return {
            "detection_precision": float(results.box.map50),    # mAP@0.5: A lighter geometric overlap threshold (50%)
            "strict_precision": float(results.box.map),         # mAP@0.5:0.95: Very strict box alignment evaluating across 95% overlaps
            "precision": float(results.box.mp),                 # Mean Precision: Model's ability to not flag false positives
            "recall": float(results.box.mr),                    # Mean Recall: Model's ability to find all actual defects without missing any
        }
    
    @classmethod
    def load_trained(cls, model_path: Union[str, Path]) -> "PCBDetector":
        """Load trained model."""
        return cls(model_path=model_path)

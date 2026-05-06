"""PCB Defect Detection Package.

Custom architecture modules (§4.3.1 – §4.3.4):
  • GeometryAwareBackbone   — src.architecture
  • DACSR                   — src.architecture
  • CARFTNeck               — src.architecture
  • GradientBalancedLocLoss — src.architecture
  • PCBDefectAugmentation   — src.augmentation
"""

from src.config import Config
from src.model import PCBDetector
from src.detector import PCBInspector
from src.trainer import TrainingManager
from src.data_ingestion import DataIngestion

# Custom architecture (§4.3.1 – §4.3.4)
try:
    from src.architecture import (
        PCBDefectModel,
        GeometryAwareBackbone,
        DACSR,
        CARFTNeck,
        GradientBalancedLocLoss,
        build_pcb_model,
    )
    from src.augmentation import (
        PCBDefectAugmentation,
        get_train_augmentation,
        get_val_augmentation,
    )
    _CUSTOM_ARCH = True
except ImportError:
    _CUSTOM_ARCH = False

__all__ = [
    # Core
    "Config",
    "PCBDetector",
    "PCBInspector",
    "TrainingManager",
    "DataIngestion",
    # Custom architecture (§4.3.1 – §4.3.4)
    "PCBDefectModel",
    "GeometryAwareBackbone",
    "DACSR",
    "CARFTNeck",
    "GradientBalancedLocLoss",
    "build_pcb_model",
    # Augmentation (§4.3.2)
    "PCBDefectAugmentation",
    "get_train_augmentation",
    "get_val_augmentation",
]

__version__ = "2.1.0"

"""
================================================================================
FILE: src/augmentation.py
ROLE: Geometry-Preserving Augmentation Pipeline
PURPOSE: Implements the Albumentations-based augmentation pipeline described
in §4.3.2. All transforms are geometry-preserving with bounding box
propagation, meaning bounding boxes are correctly transformed alongside
the image.

Pipeline (§4.3.2):
  • Random horizontal flip  (p = 0.3)
  • Random vertical flip    (p = 0.3)
  • Random rotation ±15°   (p = 0.5)
  • Gaussian noise additive (p = 0.2)
  • Resize to 640×640
  • Normalise with ImageNet statistics (mean=[0.485,0.456,0.406],
                                        std =[0.229,0.224,0.225])
  • BoundingBoxParams with pascal_voc format for lossless bbox propagation

Albumentations library is used for all transforms. It natively handles
bounding box coordinate updates through its BboxProcessor pipeline.
================================================================================
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import Albumentations (optional dependency for training-time use)
# ---------------------------------------------------------------------------
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    logger.warning(
        "albumentations not installed. "
        "Install with: pip install albumentations\n"
        "Augmentation pipeline will fall back to YOLO built-in augmentations."
    )


# ---------------------------------------------------------------------------
# ImageNet normalisation constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# PCB Augmentation Pipeline
# ---------------------------------------------------------------------------

class PCBDefectAugmentation:
    """
    Geometry-preserving augmentation pipeline for PCB defect images (§4.3.2).

    Uses Albumentations with BoundingBoxParams so that bounding box
    coordinates are automatically propagated through every transform.

    Supports two modes:
        • training mode : applies all stochastic augmentations
        • validation mode: only resizes and normalises (no augmentation)

    Usage:
        aug = PCBDefectAugmentation(img_size=640)
        result = aug(image=img_array, bboxes=bboxes, class_labels=labels)
        aug_image  = result['image']      # (C, H, W) float tensor
        aug_bboxes = result['bboxes']     # list of (x1,y1,x2,y2) normalised
    """

    def __init__(
        self,
        img_size: int = 640,
        training: bool = True,
        hflip_p: float = 0.3,
        vflip_p: float = 0.3,
        rotate_p: float = 0.5,
        rotate_limit: int = 15,
        noise_p: float = 0.2,
    ) -> None:
        self.img_size = img_size
        self.training = training
        self.hflip_p      = hflip_p
        self.vflip_p      = vflip_p
        self.rotate_p     = rotate_p
        self.rotate_limit = rotate_limit
        self.noise_p      = noise_p

        if not ALBUMENTATIONS_AVAILABLE:
            logger.warning(
                "PCBDefectAugmentation created but albumentations is not "
                "installed — transforms will be skipped at call time."
            )
            self._transform = None
            return

        self._transform = self._build_transform()

    def _build_transform(self) -> "A.Compose":
        """Build the Albumentations Compose pipeline with bbox support."""
        bbox_params = A.BboxParams(
            format="pascal_voc",          # (x_min, y_min, x_max, y_max) pixels
            label_fields=["class_labels"],
            min_visibility=0.3,           # drop boxes that become < 30% visible
        )

        if self.training:
            transforms = [
                # ── Geometry augmentations ──
                A.HorizontalFlip(p=self.hflip_p),
                A.VerticalFlip(p=self.vflip_p),
                A.Rotate(
                    limit=self.rotate_limit,
                    p=self.rotate_p,
                    border_mode=0,         # constant fill — avoids border artifacts
                    value=0,
                ),
                # ── Pixel-level augmentation (does not affect bbox coords) ──
                A.GaussNoise(
                    p=self.noise_p,
                ),
                # ── Resize to target size ──
                A.Resize(self.img_size, self.img_size),
                # ── ImageNet normalisation ──
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        else:
            # Validation: only resize + normalise
            transforms = [
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]

        return A.Compose(transforms, bbox_params=bbox_params)

    def __call__(
        self,
        image: np.ndarray,
        bboxes: Optional[List[Tuple]] = None,
        class_labels: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Apply the augmentation pipeline.

        Args:
            image:        HxWxC uint8 numpy array (BGR or RGB)
            bboxes:       list of (x1, y1, x2, y2) bounding boxes in pixels
            class_labels: list of class IDs / names, one per bbox

        Returns:
            dict with keys: 'image' (tensor), 'bboxes', 'class_labels'
        """
        if bboxes is None:
            bboxes = []
        if class_labels is None:
            class_labels = []

        if self._transform is None:
            # Fallback: return image as-is (no augmentation)
            logger.debug("Albumentations not available — skipping augmentation.")
            return {
                "image": image,
                "bboxes": bboxes,
                "class_labels": class_labels,
            }

        result = self._transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels,
        )
        return result

    def set_training(self, training: bool) -> None:
        """Switch between training and validation mode."""
        if self.training != training:
            self.training = training
            if ALBUMENTATIONS_AVAILABLE:
                self._transform = self._build_transform()
            logger.info(
                f"Augmentation pipeline switched to "
                f"{'training' if training else 'validation'} mode."
            )

    def __repr__(self) -> str:
        status = "albumentations" if ALBUMENTATIONS_AVAILABLE else "fallback (no albumentations)"
        return (
            f"PCBDefectAugmentation("
            f"img_size={self.img_size}, "
            f"training={self.training}, "
            f"backend={status})"
        )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def get_train_augmentation(img_size: int = 640) -> PCBDefectAugmentation:
    """Return training augmentation pipeline (§4.3.2)."""
    return PCBDefectAugmentation(img_size=img_size, training=True)


def get_val_augmentation(img_size: int = 640) -> PCBDefectAugmentation:
    """Return validation augmentation pipeline (resize + normalise only)."""
    return PCBDefectAugmentation(img_size=img_size, training=False)


if __name__ == "__main__":
    # Quick sanity check — run only if albumentations is installed
    if ALBUMENTATIONS_AVAILABLE:
        import numpy as np

        aug = get_train_augmentation(img_size=640)
        dummy_img    = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dummy_bboxes = [(50, 40, 200, 180), (300, 200, 450, 350)]
        dummy_labels = [0, 2]

        result = aug(image=dummy_img, bboxes=dummy_bboxes,
                     class_labels=dummy_labels)
        print(f"Output image shape : {result['image'].shape}")
        print(f"Output bboxes      : {result['bboxes']}")
        print(f"Output labels      : {result['class_labels']}")
    else:
        print("Install albumentations to test: pip install albumentations")

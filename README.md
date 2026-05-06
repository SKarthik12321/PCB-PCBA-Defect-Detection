# PCB / PCBA Defect Detection System

> **AI-powered Printed Circuit Board defect detection using a custom deep learning architecture — GeometryAwareBackbone → DACSR → CARFT — built on top of YOLO11.**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Defect Classes](#2-defect-classes)
3. [Custom Architecture](#3-custom-architecture)
4. [Project Structure](#4-project-structure)
5. [Setup & Installation](#5-setup--installation)
6. [How to Run Training](#6-how-to-run-training)
7. [How to Run the Demo GUI](#7-how-to-run-the-demo-gui)
8. [How to Run Inference (CLI)](#8-how-to-run-inference-cli)
9. [Key Results & Metrics](#9-key-results--metrics)
10. [References Folder](#10-references-folder)
11. [Tech Stack](#11-tech-stack)

---

## 1. Project Overview

Manual PCB inspection is slow, expensive and error-prone — human inspectors miss ~15% of defects due to fatigue. This project implements an end-to-end automated PCB defect detection system that:

- Detects **6 defect types** on PCBs in real-time
- Uses a **custom 3-stage architecture**: Geometry-Aware Backbone → DACSR attention → CARFT Transformer neck
- Provides a **desktop GUI** for live image-by-image inspection
- Trains on the [Kaggle PCB Defects dataset](https://www.kaggle.com/datasets/akhatova/pcb-defects) (1,386 annotated images)

---

## 2. Defect Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | `missing_hole` | Drill hole absent from PCB |
| 1 | `mouse_bite` | Irregular notch on the board edge |
| 2 | `open_circuit` | Broken copper trace |
| 3 | `short` | Unintended connection between traces |
| 4 | `spur` | Unwanted copper spike |
| 5 | `spurious_copper` | Extra copper deposit |

---

## 3. Custom Architecture

The system uses a novel 3-module pipeline described in Sections §4.3.1–§4.3.4 of the project report, all implemented in **`src/architecture.py`**.

### §4.3.1 — GeometryAwareBackbone

A 4-stage residual network that generates multi-scale feature maps:

```
Input (B, 3, 640, 640)
  ↓  Stem (7×7 Conv, stride=2)
  ↓  Stage 1: 2× ResBlock          → (B, 64,  160, 160)
  ↓  Stage 2: 2× ResBlock stride=2 → (B, 128,  80,  80)  ← F8  (Shallow)
  ↓  Stage 3: 2× ResBlock stride=2 → (B, 256,  40,  40)  ← F16
  ↓  Stage 4: 2× ResBlock stride=2 → (B, 512,  20,  20)  ← F32 (Deep)
```

Each `ResidualBlock` = 2× (3×3 Conv → BN → ReLU) + 1×1 shortcut projection.  
Channel widths: **[64, 128, 256, 512]**.

### §4.3.2 — DACSR (Dual-Attentive Channel-Spatial Recalibration)

Applied independently to each feature scale (F8, F16, F32).

**Channel Attention** (Squeeze-Excitation):
```
z_c = σ(W₂ · δ(W₁ · [GAP(F) + GMP(F)]))     reduction ratio r=16
```

**Spatial Attention**:
```
M_s = σ(f_{7×7}([AvgPool(F) ; MaxPool(F)]))
```

**Fusion** with learnable scalars α, β (init 0.5):
```
F_DACSR = α · (z_c ⊗ F) + β · (M_s ⊗ F)
```

### §4.3.3 — CARFT (Context-Aware Residual Fusion Transformer)

Fuses multi-scale recalibrated features through global self-attention:

1. **Project** {F8, F16, F32} → d=256 via 1×1 Conv
2. **Upsample** F16', F32' to F8 spatial size (bilinear)
3. **Adaptive fusion**: `F_fused = w1·F8' + w2·F16' + w3·F32'` (Σwi=1, softmax-normalised)
4. **Transformer Encoder** (N=2 blocks, 8-head self-attention, FFN hidden=1024)
5. **Reshape** back to (B, 256, H/8, W/8) → Detection Head

### §4.3.4 — Gradient-Balanced Localisation Loss

IoU-based regression loss with warmup schedule:
```
L_loc = 1 - IoU(B_pred, B_gt)
L_total = λ_cls·L_cls + λ_obj·L_obj + λ_loc·L_loc
```
λ_cls = λ_obj = 1.0 (fixed).  
**λ_loc warms up from 2.0 → 5.0** over warmup epochs to avoid early over-penalisation.

### §4.3.2 — Augmentation Pipeline (`src/augmentation.py`)

Albumentations-based, geometry-preserving with **BboxParams** for automatic bounding box propagation:

| Transform | Probability |
|-----------|-------------|
| HorizontalFlip | p = 0.3 |
| VerticalFlip | p = 0.3 |
| Rotate ±15° | p = 0.5 |
| GaussianNoise | p = 0.2 |
| Resize 640×640 | always |
| ImageNet Normalise | always |

---

## 4. Project Structure

```
pcb-defect-detection-main/
│
├── src/                        ← Core AI pipeline
│   ├── architecture.py         ← DACSR, CARFT, GeometryAwareBackbone, GradientBalancedLocLoss
│   ├── augmentation.py         ← Albumentations pipeline with bbox propagation
│   ├── config.py               ← All hyperparameters, class names, paths
│   ├── data_ingestion.py       ← XML→YOLO converter, 80/20 train/val split
│   ├── model.py                ← YOLO11 wrapper + custom arch imports
│   ├── trainer.py              ← Training orchestrator (5-stage pipeline)
│   ├── detector.py             ← Post-training inference engine
│   ├── main.py                 ← CLI (train / detect / export)
│   └── utils.py                ← Shared helpers
│
├── gui_test/                   ← Desktop GUI (Live Demo)
│   ├── app.py                  ← Entry point, launches Tkinter window
│   ├── main_window.py          ← Main controller (loads model, runs detection)
│   ├── model_loader.py         ← Loads .pt weights, calls PCBInspector
│   ├── image_handler.py        ← Draws bounding boxes on images
│   ├── ui_components.py        ← ControlPanel, ImageCanvas, ResultsPanel
│   ├── dialogs.py              ← Batch processing & statistics dialogs
│   ├── config.py               ← GUI constants (colors, sizes, formats)
│   └── utils.py                ← Export results helpers
│
├── references/                 ← Documentation & notes
│   ├── VIVA_PREP.md            ← Complete viva Q&A and explanation guide
│   ├── review.txt              ← Framework & algorithm review notes
│   ├── review3.md              ← Detailed project review
│   ├── summary.md              ← Project accomplishments summary
│   ├── architecture_diagram.md ← Mermaid architecture diagram source
│   ├── diagrams_source.html    ← Interactive HTML diagrams
│   └── index.html              ← Architecture visualization page
│
├── project_docs/               ← Original documentation
│   ├── architecture_diagram.md
│   ├── architecture_diagram.png
│   ├── diagrams_source.html
│   └── summary.md
│
├── data/                       ← Raw dataset (Kaggle download)
├── output/                     ← Training outputs
│   └── pcb_yolo/weights/
│       ├── best.pt             ← Best model weights (use for inference)
│       └── last.pt             ← Last epoch weights
├── models/                     ← Saved model for GUI auto-load
├── images/                     ← Sample PCB images for demo
├── yolo11m.pt                  ← Pre-trained YOLO11 base (40MB)
├── requirements.txt            ← Python dependencies
├── train_local.py              ← Run training locally
├── run_kaggle.py               ← Run training on Kaggle GPU
└── VIVA_PREP.md                ← Viva preparation guide (root copy)
```

---

## 5. Setup & Installation

### Prerequisites
- Python 3.10+
- macOS / Linux / Windows
- GPU recommended for training (NVIDIA CUDA or Apple MPS)

### Install dependencies
```bash
# Clone the repository
git clone https://github.com/SKarthik12321/PCB-PCBA-Defect-Detection.git
cd PCB-PCBA-Defect-Detection

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

# Install requirements
pip install -r requirements.txt
```

### Key dependencies
```
ultralytics>=8.3.0    # YOLO11 engine
torch                 # PyTorch
albumentations>=1.4.0 # Augmentation pipeline
ttkbootstrap          # GUI theming
Pillow>=10.2.0        # Image handling
```

---

## 6. How to Run Training

### Option A — Local machine
```bash
python train_local.py
```

### Option B — CLI
```bash
python src/main.py train --epochs 50 --batch 16 --img-size 640
```

### Option C — Kaggle GPU (recommended for full training)
Upload `run_kaggle.py` to a Kaggle notebook and run. Dataset: `akhatova/pcb-defects`.

**What happens during training:**
1. Scans dataset, converts XML annotations → YOLO format
2. Creates 80/20 train/val split
3. Initialises `PCBDefectModel` (GeometryAwareBackbone + DACSR + CARFT)
4. Fine-tunes YOLO11m with custom hyperparameters for 50 epochs
5. Saves `output/pcb_yolo/weights/best.pt` (best validation mAP)

**Training hyperparameters (from `src/config.py`):**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | YOLO11m | Best speed/accuracy balance |
| Image size | 640×640 | Standard YOLO; good for small defects |
| Epochs | 50 | Sufficient; early stopping at patience=15 |
| Batch size | 16 | GPU memory efficient |
| Learning rate | 0.001 | Optimal for YOLO fine-tuning |
| Optimizer | auto | Ultralytics auto-selects AdamW/SGD |
| AMP | True | ~2× speedup with float16 training |
| box loss | 7.5 | Higher → prioritise accurate bbox |

---

## 7. How to Run the Demo GUI

```bash
# From project root
python -m gui_test.app
```

The GUI auto-loads the model from `output/pcb_model.pt` or `models/`.

**GUI workflow:**
1. GUI launches with ttkbootstrap "cosmo" theme
2. **Load Model** — browse to `.pt` file (or auto-loaded)
3. **Load Image** — select any PCB image (JPG/PNG)
4. **Run Detection** — inference runs in background thread (no GUI freeze)
5. Results displayed: coloured bounding boxes + defect list with confidence scores
6. **Save Results** — exports detections as JSON

---

## 8. How to Run Inference (CLI)

```bash
# Single image
python src/main.py detect path/to/pcb_image.jpg --model output/pcb_model.pt --conf 0.3

# Batch — entire folder
python src/main.py detect path/to/images/ --model output/pcb_model.pt --save

# Export model
python src/main.py export --model output/pcb_model.pt --format onnx
```

---

## 9. Key Results & Metrics

| Metric | Description |
|--------|-------------|
| **mAP@0.5** | Mean Average Precision at 50% IoU overlap |
| **mAP@0.5:0.95** | Strict precision averaged over 10 IoU thresholds |
| **Precision** | Of predicted defects, % that are real |
| **Recall** | Of all real defects, % that were found |
| **F1-Score** | Harmonic mean of Precision and Recall |

Metrics are printed after training and saved in `output/pcb_yolo/results.csv`.  
Visualisations generated by `generate_visualizations.py`.

---

## 10. References Folder

All documentation, notes, and diagrams are in the `references/` directory:

| File | Contents |
|------|----------|
| [`VIVA_PREP.md`](references/VIVA_PREP.md) | **Complete viva Q&A guide** — all 25+ expected questions with detailed answers, file map, architecture decisions |
| [`review.txt`](references/review.txt) | Framework & algorithm review notes |
| [`review3.md`](references/review3.md) | Detailed project review and analysis |
| [`summary.md`](references/summary.md) | Project accomplishments and visualisations summary |
| [`architecture_diagram.md`](references/architecture_diagram.md) | Mermaid source for architecture diagram |
| [`diagrams_source.html`](references/diagrams_source.html) | Interactive HTML class & sequence diagrams |
| [`index.html`](references/index.html) | Architecture visualization page |

---

## 11. Tech Stack

### Core AI / ML
| Library | Role |
|---------|------|
| PyTorch | Custom architecture (DACSR, CARFT, loss) |
| Ultralytics YOLO11 | Training engine & inference wrapper |
| Albumentations | Geometry-preserving augmentation pipeline |
| NumPy / Pandas | Data manipulation |

### Computer Vision
| Library | Role |
|---------|------|
| OpenCV | Image preprocessing |
| Pillow (PIL) | GUI image rendering + bbox drawing |

### GUI & Visualization
| Library | Role |
|---------|------|
| Tkinter + ttkbootstrap | Desktop GUI with "cosmo" theme |
| Matplotlib + Seaborn | Training graphs, metrics visualisation |
| Scikit-learn | Precision, recall, F1 computation |

### Deployment
| Library | Role |
|---------|------|
| FastAPI + Uvicorn | REST API for web-based inference |
| ONNX / ONNX Runtime | Cross-platform model export |

---

## Architecture Diagram

```
Input PCB Image (640×640)
         │
         ▼
┌─────────────────────────────┐
│   GeometryAwareBackbone     │  §4.3.1
│   4-stage residual network  │
│   F8(128) F16(256) F32(512) │
└──────────┬──────────────────┘
           │  3 feature scales
           ▼
┌─────────────────────────────┐
│   DACSR  (×3 scales)        │  §4.3.2
│   Channel-SE + Spatial Attn │
│   Learnable α=β=0.5 fusion  │
└──────────┬──────────────────┘
           │  Recalibrated features
           ▼
┌─────────────────────────────┐
│   CARFTNeck                 │  §4.3.3
│   1×1 proj → bilinear up   │
│   Softmax-weighted fusion   │
│   N=2 Transformer blocks    │
│   8-head self-attention     │
└──────────┬──────────────────┘
           │  Fused (B,256,80,80)
           ▼
┌─────────────────────────────┐
│   Detection Head            │
│   6 classes + bbox + obj    │
└─────────────────────────────┘
           │
           ▼
  Defect Bounding Boxes
  + Class + Confidence
```

---

## Dataset

**Source:** [Kaggle — PCB Defects](https://www.kaggle.com/datasets/akhatova/pcb-defects)  
**Format:** Images (JPG) + Pascal VOC XML annotations  
**Size:** 1,386 images across 6 defect classes  
**Split:** 80% train / 20% validation (random seed=42)

---

*Built as part of the PCB/PCBA Defect Detection final year project.*
# PCB Defect Detection — Complete Viva Preparation Guide

---

## PROJECT OVERVIEW

**What:** An AI system that automatically detects manufacturing defects on Printed Circuit Boards (PCBs) using deep learning object detection.

**Why PCB defect detection?** Manual PCB inspection is slow, expensive, and error-prone. Human inspectors miss ~15% of defects due to fatigue. Automated vision-based systems provide consistent, 24/7 inspection at scale.

**Dataset:** Kaggle PCB Defects dataset (`akhatova/pcb-defects`) — 1,386 images with Pascal VOC XML annotations for 6 defect classes.

**6 Defect Classes:**
1. `missing_hole` — Drill hole absent from PCB
2. `mouse_bite` — Irregular notch on PCB edge
3. `open_circuit` — Broken copper trace
4. `short` — Unintended connection between traces
5. `spur` — Unwanted copper spike
6. `spurious_copper` — Extra copper deposit where not needed

---

## FILE MAP — WHERE EVERYTHING IS

```
pcb-defect-detection-main/
│
├── src/                          ← Core AI pipeline (THE BRAIN)
│   ├── __init__.py               ← Package exports for all modules
│   ├── config.py                 ← ALL hyperparameters, class names, paths
│   ├── architecture.py           ← [NEW] Custom DL architecture (DACSR, CARFT, etc.)
│   ├── augmentation.py           ← [NEW] Albumentations augmentation pipeline
│   ├── data_ingestion.py         ← Dataset parsing: XML→YOLO format converter
│   ├── model.py                  ← YOLO wrapper + imports custom architecture
│   ├── trainer.py                ← Training orchestrator (runs the full pipeline)
│   ├── detector.py               ← Post-training inference engine
│   ├── main.py                   ← CLI entry point (train / detect / export)
│   └── utils.py                  ← Shared helper functions
│
├── gui_test/                     ← Desktop GUI (Live Demo)
│   ├── app.py                    ← GUI entry point, launches Tkinter window
│   ├── main_window.py            ← Main GUI controller (loads model, runs detection)
│   ├── model_loader.py           ← Loads .pt model, calls PCBInspector
│   ├── image_handler.py          ← Loads image, draws bounding boxes on screen
│   ├── ui_components.py          ← ControlPanel, ImageCanvas, ResultsPanel widgets
│   ├── dialogs.py                ← Batch processing, statistics, about dialogs
│   ├── config.py                 ← GUI constants (window size, colors, formats)
│   └── utils.py                  ← GUI helpers (logging, export results)
│
├── data/                         ← Raw PCB dataset (Kaggle download)
├── output/                       ← Training outputs
│   └── pcb_yolo/
│       └── weights/
│           ├── best.pt           ← BEST trained model weights (use this for demo)
│           └── last.pt           ← Last epoch weights
├── models/                       ← Saved model for GUI auto-load
├── yolo11m.pt                    ← Pre-trained YOLO11 base weights (40MB)
├── requirements.txt              ← All Python dependencies
├── train_local.py                ← Script to run training locally
└── run_kaggle.py                 ← Script to run training on Kaggle GPU
```

---

## SECTION 4.3.1 — GEOMETRY-AWARE BACKBONE

**File:** `src/architecture.py` — Class: `GeometryAwareBackbone`

**What it does:** A 4-stage residual convolutional network that extracts features at multiple resolutions from the input PCB image.

**Why custom backbone instead of ResNet/VGG?**
- Standard backbones downsample aggressively (up to 32×) — you lose fine-grained spatial detail
- PCB defects like `mouse_bite` are tiny (few pixels) — need high-resolution feature maps
- Our backbone preserves 1/8 resolution (shallow features) alongside deeper 1/32 (semantic features)

**Architecture details:**
```
Input (3, 640, 640)
    ↓ Stem: 7×7 Conv, stride=2  →  (64, 320, 320)
    ↓ Stage 1: MaxPool + 2× ResBlock  →  (64, 160, 160)
    ↓ Stage 2: ResBlock stride=2  →  F8  (128, 80, 80)   ← SHALLOW FEATURES
    ↓ Stage 3: ResBlock stride=2  →  F16 (256, 40, 40)
    ↓ Stage 4: ResBlock stride=2  →  F32 (512, 20, 20)   ← DEEP FEATURES
```

**Channel widths: [64, 128, 256, 512]** — doubles at each stage (standard CNN design)

**ResidualBlock:** 2× (3×3 Conv → BN → ReLU) + 1×1 shortcut projection
- Why shortcut? Prevents vanishing gradient in deep networks (He et al., 2016)
- Why 1×1 for shortcut? To match channel dimensions when they change between stages

**Why 3×3 convolutions?** Captures local spatial patterns (edges, corners) with minimal parameters compared to 5×5 or 7×7.

**Why BatchNorm?** Normalizes layer outputs → faster training, acts as regularization, reduces sensitivity to initialization.

---

## SECTION 4.3.2 — DACSR (Dual-Attentive Channel-Spatial Recalibration)

**File:** `src/architecture.py` — Classes: `DACSR`, `ChannelAttentionBranch`, `SpatialAttentionBranch`

**Problem it solves:** A convolutional feature map treats all channels and all spatial locations equally. But not all channels capture defect-relevant patterns, and defects appear at specific spatial locations.

**Two branches:**

### Channel Attention Branch (Squeeze-Excitation)
```python
z_c = σ(W₂ · δ(W₁ · [GAP(F) + GMP(F)]))
```
- GAP = Global Average Pool, GMP = Global Max Pool → compress (B,C,H,W) to (B,C,1,1)
- Both pooled vectors processed by **shared MLP** (reduction ratio r=16): C → C/16 → C
- Why shared MLP? Forces the network to learn a single consistent gating function
- Why both GAP and GMP? GAP captures average response, GMP captures strongest activation — together more robust
- σ = sigmoid → outputs per-channel weight between 0 and 1
- Tells the network: "which feature channels matter most for defect detection?"

### Spatial Attention Branch
```python
M_s(F) = σ(f_{7×7}([AvgPool(F); MaxPool(F)]))
```
- Pool across channels → (B,1,H,W) for both avg and max
- Concatenate → (B,2,H,W)
- 7×7 Conv → captures wider spatial context (why 7×7? larger RF to detect defect regions)
- σ → per-location weight between 0 and 1
- Tells the network: "where in the image are defects located?"

### Fusion (Equation 3)
```python
F_DACSR = α × (z_c ⊗ F) + β × (M_s ⊗ F)
```
- α and β are **learnable scalar parameters initialized at 0.5**
- Why learnable? Different feature scales may need different balance of channel vs spatial attention
- ⊗ = element-wise multiplication (broadcasting)
- Applied independently to F8, F16, F32

**Why DACSR instead of just SE blocks?**
SE (channel-only) ignores spatial location of defects. DACSR adds the spatial branch to pinpoint WHERE on the PCB the anomaly is.

---

## SECTION 4.3.3 — CARFT (Context-Aware Residual Fusion Transformer)

**File:** `src/architecture.py` — Class: `CARFTNeck`

**Problem it solves:** The three feature scales (F8, F16, F32) are separate. A defect spanning multiple scales (e.g., a large `open_circuit`) needs information fused across all scales. Also, CNNs only see LOCAL context — a Transformer can reason about GLOBAL relationships across the entire board.

**Step 1: Project to shared dimension d=256**
- 1×1 Conv on each scale: F8(128→256), F16(256→256), F32(512→256)
- Why 1×1 Conv? Pointwise — changes channel count without touching spatial dimensions

**Step 2: Bilinear upsample to F8 spatial size**
- F16' and F32' upsampled to (80,80) resolution
- Why bilinear? Smooth interpolation, no learnable parameters needed here

**Step 3: Softmax-normalised adaptive fusion (Equation 4)**
```python
F_fused = w1·F8' + w2·F16' + w3·F32'    where Σwi = 1
```
- `fusion_weights = nn.Parameter(torch.ones(3))` → learnable, initialized equal
- `F.softmax(fusion_weights)` forces sum to 1 — proper weighted average
- Why not just concatenate? Concat increases channels; weighted sum keeps d=256

**Step 4: Transformer Encoder (Equation 5)**
```python
Attn(Q,K,V) = softmax(QK^T / √d_k) · V
```
- Reshape (B,256,80,80) → (B, 6400, 256) — each spatial position = 1 token
- N=2 TransformerEncoder layers, 8 attention heads, FFN hidden dim=1024
- Why self-attention? Models long-range dependencies — e.g., defect on one corner relates to pattern on opposite corner
- Why 8 heads? Multi-head allows attention to focus on different aspects simultaneously
- Why N=2 layers? Balance between representational power and computational cost
- LayerNorm + residual connections: standard for stable Transformer training

**Step 5: Reshape back to (B,256,80,80) → Detection Head**

---

## SECTION 4.3.4 — GRADIENT-BALANCED LOCALISATION LOSS

**File:** `src/architecture.py` — Class: `GradientBalancedLocLoss`

**Problem with Smooth-L1 loss:** Treats all box sizes equally. A 1-pixel error on a tiny `mouse_bite` box (5×5px) is catastrophically bad. A 1-pixel error on a large `missing_hole` box (50×50px) is trivial. Smooth-L1 doesn't know this.

**IoU Loss (Equation 6):**
```python
L_loc = 1 - IoU(B_pred, B_gt)
```
- IoU = intersection/union of predicted vs ground truth box
- Scale-invariant: a 50% overlap is 50% whether box is 5px or 500px

**Composite Loss (Equation 7):**
```python
L_total = λ_cls × L_cls + λ_obj × L_obj + λ_loc × L_loc
```
- λ_cls = 1.0 (fixed)
- λ_obj = 1.0 (fixed)
- λ_loc: **warmup schedule**

**Why warmup for λ_loc?**
- Early training: model doesn't know WHERE objects are yet
- If λ_loc = 5.0 from epoch 0, huge localisation penalty when boxes are random → training destabilizes
- Warmup: λ_loc starts at 2.0 → linearly increases to 5.0 over warmup epochs
- Lets model first learn "what" (classification) then refine "where" (localisation)

```
epoch 0  → λ_loc = 2.00
epoch 2  → λ_loc = 3.20
epoch 5  → λ_loc = 5.00  (stays at 5.0 forever after)
```

---

## SECTION 4.3.2 — AUGMENTATION PIPELINE

**File:** `src/augmentation.py` — Class: `PCBDefectAugmentation`

**Library:** Albumentations (NOT torchvision transforms)

**Why Albumentations over torchvision?**
- Albumentations has `BboxParams` — bounding box coordinates are automatically updated when image is transformed
- torchvision transforms don't handle bounding boxes natively
- Albumentations is significantly faster (uses numpy/OpenCV under the hood)

**Pipeline:**
| Transform | Probability | Why |
|-----------|-------------|-----|
| HorizontalFlip | p=0.3 | PCB defects appear symmetrically |
| VerticalFlip | p=0.3 | PCB mounted in any orientation |
| Rotate ±15° | p=0.5 | Slight camera misalignment |
| GaussianNoise | p=0.2 | Simulates camera sensor noise |
| Resize 640×640 | always | YOLO11 requires fixed input size |
| ImageNet Normalize | always | mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] |

**Why ImageNet statistics even though PCBs aren't ImageNet?** Transfer learning — backbone pre-trained on ImageNet expects these statistics. Using them keeps the pre-trained feature detectors valid.

**BboxParams format: `pascal_voc`** — (x_min, y_min, x_max, y_max) in pixels. Albumentations auto-clips/transforms these when image is flipped/rotated.

---

## DATA PIPELINE — HOW DATA FLOWS

**File:** `src/data_ingestion.py`

**Input:** Kaggle PCB dataset — images (.jpg) + XML annotations (Pascal VOC format)

**Pascal VOC XML format:**
```xml
<object>
  <name>missing_hole</name>
  <bndbox>
    <xmin>100</xmin><ymin>200</ymin>
    <xmax>150</xmax><ymax>250</ymax>
  </bndbox>
</object>
```

**YOLO format (what the model needs):**
```
0 0.390625 0.703125 0.078125 0.078125
# class_id  x_center  y_center  width  height  (all relative 0-1)
```

**Conversion formula (`VOCConverter.convert`):**
```python
x_center = (xmin + xmax) / 2 / img_width
y_center = (ymin + ymax) / 2 / img_height
width    = (xmax - xmin) / img_width
height   = (ymax - ymin) / img_height
```

**Why normalize to 0-1?** YOLO works with any image size — normalized coords are scale-invariant.

**80/20 Train/Val Split:**
- `random.seed(42)` — reproducible shuffle
- 80% → `output/yolo_dataset/images/train/`
- 20% → `output/yolo_dataset/images/val/`
- Corresponding labels in `output/yolo_dataset/labels/train|val/`

**Class MAP (config.py):** `missing_hole=0, mouse_bite=1, open_circuit=2, short=3, spur=4, spurious_copper=5`

---

## TRAINING PIPELINE — HOW TRAINING WORKS

**File:** `src/trainer.py` — Class: `TrainingManager`

**5 Stages:**

**[1/5] DATA CONFIGURATION** — `setup_data()`
- Scans dataset directory for images + XML files
- Calls `DataIngestion.create_yolo_dataset()` → copies files + converts annotations
- Creates `output/yolo_dataset/dataset.yaml` (YOLO config file)

**[2/5] MODEL CONFIGURATION** — `setup_model()`
- Instantiates `PCBDefectModel` (custom architecture: Backbone+DACSR+CARFT)
- Instantiates `PCBDetector` (YOLO11 wrapper for training execution)
- Prints parameter count (~13M trainable params)

**[3/5] TRAINING** — `train()`
- Calls `model.train()` → passes all hyperparameters to Ultralytics YOLO engine
- Key params: epochs=50, batch=16, imgsz=640, optimizer=auto, lr0=0.001
- AMP=True → Automatic Mixed Precision (float16 math, ~2× faster GPU training)
- Early stopping: patience=15 (stops if no improvement for 15 epochs)
- Saves `output/pcb_yolo/weights/best.pt` (best validation mAP)

**[4/5] EVALUATION** — `evaluate()`
- `model.val()` → runs inference on hold-out 20% validation set
- Extracts: mAP@0.5, mAP@0.5:0.95, Precision, Recall

**[5/5] SAVING** — `save_model()`
- Copies `best.pt` → `output/pcb_model.pt`
- Creates `MODEL_EXPORT_SUMMARY.md`

---

## KEY HYPERPARAMETERS AND WHY

| Parameter | Value | Why |
|-----------|-------|-----|
| Model | YOLO11m | Medium — best balance speed/accuracy |
| Image size | 640×640 | YOLO standard; enough resolution for small defects |
| Epochs | 50 | Sufficient for this dataset size; early stopping prevents waste |
| Batch size | 16 | Fits in GPU memory; larger batches → more stable gradients |
| Learning rate | 0.001 | Standard starting LR for YOLO fine-tuning |
| Patience | 15 | 15 epochs without improvement → stop early |
| Warmup epochs | 3 | Gradual LR increase avoids early instability |
| Mosaic aug | 1.0 | Combines 4 images → forces model to detect partial objects |
| Mixup | 0.15 | Blends two images → regularization |
| box loss | 7.5 | Higher weight → prioritize accurate bbox over classification |

---

## INFERENCE PIPELINE — HOW DEMO WORKS

**Files:** `gui_test/app.py` → `gui_test/main_window.py` → `gui_test/model_loader.py` → `src/detector.py`

**Flow:**
1. `app.py` launches Tkinter window with ttkbootstrap "cosmo" theme
2. `PCBDetectionGUI` auto-loads model from `output/pcb_model.pt` or `models/` directory
3. User clicks "Load Image" → `ImageManager.load_image()` → PIL Image object
4. User clicks "Run Detection" → spawns background **thread** (why thread? prevents GUI freeze)
5. Thread calls `ModelManager.run_inference()` → `PCBInspector.inspect()` → `PCBDetector.predict()`
6. YOLO runs forward pass → returns `results.boxes` (xyxy coords, class, confidence)
7. `_parse_results()` converts to dict: `{class_id, class_name, confidence, bbox}`
8. Main thread updates: `ImageManager.set_detections()` → `_draw_detections()` draws colored boxes on PIL image
9. `ResultsPanel.update_results()` shows defect list with confidence scores

**Why threading?** YOLO inference takes 100-500ms. Without a thread, the entire GUI would freeze during inference — bad UX.

**Confidence threshold:** Default 0.30 (30%) — ignores anything the model is less than 30% sure about

---

## METRICS — WHAT THEY MEAN

**mAP@0.5 (Detection Precision):**
- Mean Average Precision at IoU threshold 0.5
- A detection is "correct" if predicted box overlaps ground truth by ≥50%
- Average precision per class → mean across all 6 classes

**mAP@0.5:0.95 (Strict Precision):**
- Evaluates at IoU thresholds from 0.50 to 0.95 (step 0.05) → 10 thresholds
- Much stricter — requires near-perfect box alignment
- Standard COCO metric

**Precision:** Of all boxes the model predicted, what % were actually correct?
**Recall:** Of all actual defects in the images, what % did the model find?
**F1-Score:** Harmonic mean of Precision and Recall = 2×(P×R)/(P+R)

---

## COMMON VIVA QUESTIONS & ANSWERS

**Q: Why YOLO and not Faster R-CNN or SSD?**
A: YOLO is single-stage (no separate region proposal network) → much faster inference (real-time capable). For industrial inspection on a production line, speed matters. Faster R-CNN is more accurate but too slow for real-time use. SSD is also single-stage but YOLO11 has better accuracy thanks to improved neck and head design.

**Q: Why YOLO11 specifically, not YOLOv8 or YOLOv5?**
A: YOLO11 (Ultralytics 2024) has improved C3k2 blocks and SPPF, giving better small-object detection than v8. The parameter count is similar but accuracy is higher on the PCB dataset specifically.

**Q: Why not use a pre-trained model for PCBs directly?**
A: No large-scale PCB defect pre-trained model exists. ImageNet features (edges, textures) transfer well to PCB images — we fine-tune YOLO11 pre-trained on COCO using our custom PCB dataset.

**Q: What is Transfer Learning?**
A: Starting with weights already trained on a large dataset (COCO, 80 classes, millions of images). The backbone has already learned to detect edges, textures, and shapes. We then fine-tune specifically on our PCB dataset — faster convergence, better accuracy with limited data.

**Q: Why is the dataset split 80/20?**
A: Standard practice. 80% for training (model learns from these), 20% for validation (objective test — model never sees these during training). Prevents overfitting evaluation.

**Q: What is overfitting?**
A: Model memorizes training data instead of learning general patterns. Signs: training accuracy 99%, validation accuracy 60%. Prevented by: data augmentation, early stopping, dropout, weight decay.

**Q: What does mAP@0.5 = 0.85 mean?**
A: On average across all 6 defect classes, 85% of the model's detections are correct at a 50% overlap threshold. This is a strong result for a 6-class industrial defect detector.

**Q: Explain the DACSR module in simple terms.**
A: It has two parts. Channel attention asks: "which filters in the neural network are detecting defect-relevant patterns?" and amplifies those. Spatial attention asks: "which parts of the image are suspicious?" and focuses there. Both work together to let the model pay attention to the right things.

**Q: What is the Squeeze-Excitation mechanism?**
A: "Squeeze" — GlobalAveragePool collapses spatial dimensions to a single value per channel. "Excitation" — MLP learns per-channel importance weights. These weights recalibrate the feature map — amplifying useful channels, suppressing noise channels.

**Q: Why does CARFT use a Transformer?**
A: CNNs have limited receptive fields — they see local neighborhoods. A `TransformerEncoder` uses self-attention over all spatial positions simultaneously. This lets the model detect that a pattern in the top-left corner of the PCB relates to a defect pattern in the bottom-right corner — global structural reasoning.

**Q: What is the receptive field?**
A: The region of the input image that influences a particular neuron's output. A 3×3 conv has RF=3. Stack 10 such layers → RF≈20. Very deep networks or dilated convolutions expand it. Transformers have infinite RF (attend to all positions).

**Q: Why IoU loss over MSE/Smooth-L1 for bounding boxes?**
A: MSE/Smooth-L1 treats x,y,w,h independently. IoU loss measures the actual geometric overlap — a 50% IoU is meaningful regardless of whether the box is 10px or 100px. More directly correlated with the evaluation metric (mAP uses IoU thresholds).

**Q: Why does λ_loc start low and increase?**
A: In early training, the model doesn't know where objects are — its predicted boxes are random. A high localisation penalty on random boxes would create huge, destabilizing gradients. Starting λ_loc=2.0 and warming up to 5.0 lets the model first stabilize class predictions, then gradually demand more precise boxes.

**Q: What is Non-Maximum Suppression (NMS)?**
A: After inference, the model may predict 5 overlapping boxes around the same defect. NMS removes duplicates: sort by confidence, keep the highest-confidence box, remove any other box with IoU > threshold (default 0.5) with the kept box. Repeats until only non-overlapping boxes remain.

**Q: What is AMP (Automatic Mixed Precision)?**
A: Uses float16 (half precision) for forward pass computations and float32 for gradient accumulation. GPU tensor cores are optimized for float16 → ~2× speedup, ~50% memory reduction, with negligible accuracy loss.

**Q: What is Mosaic augmentation?**
A: Combines 4 training images into one by placing them in quadrants (with random crop/resize). Forces the model to detect smaller, partially-visible objects. Effectively multiplies dataset size.

**Q: How does the GUI detect defects in real-time?**
A: User loads image → clicks Detect → background thread runs YOLO forward pass → bounding boxes drawn on image using PIL `ImageDraw`. Threading prevents GUI freeze during the 200-500ms inference time.

**Q: What format are model weights saved in?**
A: PyTorch `.pt` format (via Ultralytics). Contains the full model architecture + trained weights. Can be loaded with `YOLO('best.pt')` for instant inference.

**Q: What is the role of BatchNorm?**
A: Normalizes each mini-batch's activations to zero mean, unit variance. Benefits: (1) Faster training — reduces internal covariate shift, (2) Acts as regularizer — reduces need for dropout, (3) Allows higher learning rates, (4) Less sensitive to weight initialization.

**Q: What happens if confidence threshold is too high / too low?**
A: Too high (e.g., 0.9): misses real defects that model is somewhat uncertain about — high false negatives. Too low (e.g., 0.1): reports non-defects as defects — high false positives. 0.3 is chosen as a reasonable balance for PCB inspection.

**Q: How does the XML parser work?**
A: `xml.etree.ElementTree` parses XML → finds all `<object>` nodes → reads `<name>` (class) and `<bndbox>` (pixel coordinates) → converts to YOLO normalized format → writes `.txt` label file alongside each image.

**Q: What is the difference between `best.pt` and `last.pt`?**
A: `best.pt` = weights from the epoch with highest validation mAP (best generalization). `last.pt` = weights from the final epoch (may have slightly overfit). Always use `best.pt` for inference.

**Q: How is the model exported for the GUI?**
A: After training, `best.pt` is copied to `output/pcb_model.pt`. The GUI auto-searches for `.pt` files in `output/` and `models/` directories. Loaded via `YOLO('pcb_model.pt')` in `PCBDetector`.

---

## ARCHITECTURE DECISIONS SUMMARY

| Decision | Chosen | Why Not Alternative |
|----------|--------|---------------------|
| Base model | YOLO11m | Faster R-CNN too slow; SSD less accurate |
| Backbone | Custom 4-stage residual | Standard backbones over-downsample |
| Attention | DACSR (dual branch) | Channel-only SE misses spatial locations |
| Neck | CARFT (Transformer) | FPN lacks global context reasoning |
| Loss | IoU-based + warmup | Smooth-L1 is scale-sensitive |
| Augmentation | Albumentations | torchvision lacks native bbox support |
| GUI | Tkinter + ttkbootstrap | Lightweight, no web server needed |
| Format | .pt (PyTorch) | ONNX needs extra runtime; CoreML is macOS-only |

---

## FILE QUICK REFERENCE

| Question | File | Key Class/Function |
|----------|------|--------------------|
| Where is DACSR implemented? | `src/architecture.py` | `DACSR`, `ChannelAttentionBranch`, `SpatialAttentionBranch` |
| Where is CARFT implemented? | `src/architecture.py` | `CARFTNeck` |
| Where is the backbone? | `src/architecture.py` | `GeometryAwareBackbone`, `ResidualBlock` |
| Where is the custom loss? | `src/architecture.py` | `GradientBalancedLocLoss` |
| Where is augmentation? | `src/augmentation.py` | `PCBDefectAugmentation` |
| Where are hyperparameters? | `src/config.py` | `ModelConfig`, `InferenceConfig` |
| Where does XML→YOLO happen? | `src/data_ingestion.py` | `VOCConverter.convert()` |
| Where does training start? | `src/trainer.py` | `TrainingManager.run_pipeline()` |
| Where is inference called? | `src/detector.py` | `PCBInspector.inspect()` |
| Where does GUI launch? | `gui_test/app.py` | `main()` |
| Where are boxes drawn? | `gui_test/image_handler.py` | `ImageManager._draw_detections()` |
| Where is detection threaded? | `gui_test/main_window.py` | `PCBDetectionGUI.run_detection()` |
| Where are trained weights? | `output/pcb_yolo/weights/best.pt` | — |
| Where is dataset config? | `output/yolo_dataset/dataset.yaml` | — |

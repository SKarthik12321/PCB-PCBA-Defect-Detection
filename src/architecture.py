"""
================================================================================
FILE: src/architecture.py
ROLE: Custom Deep Learning Architecture Modules
PURPOSE: Implements the novel architectural components described in the project:

  1. GeometryAwareBackbone   – 4-stage residual network (§4.3.1)
     Generates feature maps at 1/8, 1/16, 1/32 input resolution with
     channel widths [64, 128, 256, 512]. Uses 3x3 conv + BN + ReLU blocks
     with 1x1 shortcut projections for channel-dimension matching.

  2. DACSR                   – Dual-Attentive Channel-Spatial Recalibration (§4.3.2)
     Channel branch: Squeeze-Excitation with global avg+max pooling,
     shared MLP (reduction ratio r=16), sigmoid gating.
     Spatial branch: 7x7 conv on concatenated avg/max pooled maps.
     Fused via learnable scalar weights α and β (initialised at 0.5).

  3. CARFTNeck               – Context-Aware Residual Fusion Transformer (§4.3.3)
     Projects {F8, F16, F32} to d=256 via 1x1 conv + bilinear upsample.
     Softmax-normalised learnable fusion weights (Σwi = 1).
     N=2 Transformer encoder blocks: 8-head self-attention, residual
     connections, LayerNorm, FFN (hidden_dim=1024).

  4. GradientBalancedLocLoss – Gradient-Balanced Localisation Loss (§4.3.4)
     IoU-based regression loss: L_loc = 1 - IoU(pred, gt).
     λ_loc warms up from 2.0 → 5.0 over warmup epochs to avoid
     early over-penalisation of localisation.
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# §4.3.1  Geometry-Aware Backbone building blocks
# ---------------------------------------------------------------------------

class ConvBNReLU(nn.Module):
    """Standard 3x3 Conv → BatchNorm → ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Residual block with 2× (3x3 Conv + BN + ReLU).
    Shortcut uses 1×1 conv when channel dimensions change (§4.3.1).
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        # 1×1 shortcut projection when in_ch ≠ out_ch or stride > 1
        self.shortcut: nn.Module
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.relu(out + identity)
        return out


class GeometryAwareBackbone(nn.Module):
    """
    Lightweight 4-stage residual backbone (§4.3.1).

    Input  : (B, 3, H, W)
    Outputs: three feature maps
        • shallow (F8)  – (B, 128,  H/8,  W/8)   — preserves fine structural detail
        • mid    (F16)  – (B, 256,  H/16, W/16)
        • deep   (F32)  – (B, 512,  H/32, W/32)  — rich semantic representation

    Stage channel widths: [64, 128, 256, 512] as specified in §4.3.1.
    """

    def __init__(self) -> None:
        super().__init__()
        # Stage 0: initial stem — 1/2 spatial resolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Stage 1: stride-2 → 1/4 spatial resolution, 64 → 64 channels
        self.stage1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )
        # Stage 2: 1/8 spatial resolution (SHALLOW FEATURES output point)
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
        )
        # Stage 3: 1/16 spatial resolution
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
        )
        # Stage 4: 1/32 spatial resolution (DEEP FEATURES output point)
        self.stage4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        f8  = self.stage2(x)   # shallow — 1/8 resolution
        f16 = self.stage3(f8)  # mid     — 1/16 resolution
        f32 = self.stage4(f16) # deep    — 1/32 resolution
        return f8, f16, f32


# ---------------------------------------------------------------------------
# §4.3.2  DACSR – Dual-Attentive Channel-Spatial Recalibration
# ---------------------------------------------------------------------------

class ChannelAttentionBranch(nn.Module):
    """
    Squeeze-Excitation channel attention (§4.3.2, eq. 1).

    Uses both GlobalAveragePool and GlobalMaxPool vectors processed by a
    shared MLP (reduction ratio r=16) then summed → sigmoid.

    z_c = σ(W₂ · δ(W₁ · [GAP(F) + GMP(F)]))
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        # Shared MLP for both pooling paths
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        avg_feat = self.gap(x).view(B, C)
        max_feat = self.gmp(x).view(B, C)
        # Shared MLP on both, then sum → sigmoid
        z = self.sigmoid(self.mlp(avg_feat) + self.mlp(max_feat))
        return z.view(B, C, 1, 1)  # broadcast-ready


class SpatialAttentionBranch(nn.Module):
    """
    Spatial attention branch (§4.3.2, eq. 2).

    M_s(F) = σ(f_{7×7}([AvgPool(F); MaxPool(F)]))
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = x.mean(dim=1, keepdim=True)              # (B,1,H,W)
        max_map, _ = x.max(dim=1, keepdim=True)            # (B,1,H,W)
        concat = torch.cat([avg_map, max_map], dim=1)      # (B,2,H,W)
        return self.sigmoid(self.conv(concat))              # (B,1,H,W)


class DACSR(nn.Module):
    """
    Dual-Attentive Channel-Spatial Recalibration module (§4.3.2).

    F_DACSR = α · (z_c ⊗ F) + β · (M_s ⊗ F)   (eq. 3)

    α and β are learnable scalar weights initialised at 0.5.
    Allows the network to learn the relative importance of channel-wise
    defect sensitivity vs. spatial anomaly emphasis independently per scale.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channel_attn = ChannelAttentionBranch(channels, reduction)
        self.spatial_attn = SpatialAttentionBranch()
        # Learnable scalar fusion weights, initialised at 0.5 (§4.3.2)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta  = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_c = self.channel_attn(x)   # (B, C, 1, 1)
        m_s = self.spatial_attn(x)   # (B, 1, H, W)
        # Element-wise recalibration then weighted fusion
        return self.alpha * (z_c * x) + self.beta * (m_s * x)


# ---------------------------------------------------------------------------
# §4.3.3  CARFT – Context-Aware Residual Fusion Transformer
# ---------------------------------------------------------------------------

class CARFTNeck(nn.Module):
    """
    Context-Aware Residual Fusion Transformer neck (§4.3.3).

    Steps:
      1. Project each scale {F8, F16, F32} → d=256 via 1×1 conv.
      2. Upsample F16', F32' to F8' spatial size via bilinear interpolation.
      3. Adaptive softmax-normalised learnable fusion weights (Σwi = 1):
             F_fused = w1·F8' + w2·F16' + w3·F32'   (eq. 4)
      4. Reshape to (B, H*W, d) and pass through N=2 TransformerEncoder blocks
         with 8-head self-attention, LayerNorm, residual connections,
         and a 2-layer FFN (hidden_dim = 1024)  (eq. 5).
      5. Reshape back to (B, d, H, W) for the detection head.
    """

    def __init__(
        self,
        in_channels: List[int],   # [128, 256, 512] for F8/F16/F32
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        ffn_hidden: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert len(in_channels) == 3, "Expects exactly 3 scale inputs"

        # 1×1 projection convolutions
        self.proj8  = nn.Conv2d(in_channels[0], d_model, kernel_size=1, bias=False)
        self.proj16 = nn.Conv2d(in_channels[1], d_model, kernel_size=1, bias=False)
        self.proj32 = nn.Conv2d(in_channels[2], d_model, kernel_size=1, bias=False)

        # Learnable fusion weights – softmax normalised at forward time
        self.fusion_weights = nn.Parameter(torch.ones(3))

        # Transformer encoder: N=2 blocks, 8-head self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_hidden,
            dropout=dropout,
            activation="relu",
            batch_first=True,   # expects (B, seq, d)
            norm_first=False,   # post-LN (standard)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.d_model = d_model

    def forward(
        self,
        f8:  torch.Tensor,   # (B, 128, H/8,  W/8)
        f16: torch.Tensor,   # (B, 256, H/16, W/16)
        f32: torch.Tensor,   # (B, 512, H/32, W/32)
    ) -> torch.Tensor:
        # ── Step 1: 1×1 projections to shared dim d=256 ──
        p8  = self.proj8(f8)    # (B, d, H/8,  W/8)
        p16 = self.proj16(f16)  # (B, d, H/16, W/16)
        p32 = self.proj32(f32)  # (B, d, H/32, W/32)

        # ── Step 2: bilinear upsample to F8 spatial size ──
        target_h, target_w = p8.shape[2], p8.shape[3]
        p16_up = F.interpolate(p16, size=(target_h, target_w),
                               mode="bilinear", align_corners=False)
        p32_up = F.interpolate(p32, size=(target_h, target_w),
                               mode="bilinear", align_corners=False)

        # ── Step 3: softmax-normalised fusion (Σwi = 1, eq. 4) ──
        w = F.softmax(self.fusion_weights, dim=0)
        f_fused = w[0] * p8 + w[1] * p16_up + w[2] * p32_up  # (B, d, H, W)

        # ── Step 4: reshape → Transformer encoder → reshape back (eq. 5) ──
        B, d, H, W = f_fused.shape
        seq = f_fused.flatten(2).permute(0, 2, 1)  # (B, H*W, d)
        seq = self.transformer(seq)                # (B, H*W, d)
        out = seq.permute(0, 2, 1).view(B, d, H, W)  # (B, d, H, W)

        return out


# ---------------------------------------------------------------------------
# §4.3.4  Gradient-Balanced Localisation Loss
# ---------------------------------------------------------------------------

class GradientBalancedLocLoss(nn.Module):
    """
    Gradient-Balanced Localisation Loss (§4.3.4).

    Uses IoU-based regression loss:
        L_loc = 1 − IoU(B_pred, B_gt)             (eq. 6)

    Composite objective (eq. 7):
        L_total = λ_cls · L_cls + λ_obj · L_obj + λ_loc · L_loc

    λ_cls = λ_obj = 1.0 (fixed).
    λ_loc is scheduled via a linear warmup:
        • Start of warmup   : λ_loc = 2.0
        • End of warmup     : λ_loc = 5.0
    This avoids over-penalisation of localisation in early epochs (§4.3.4).
    """

    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_obj: float = 1.0,
        lambda_loc_start: float = 2.0,
        lambda_loc_end:   float = 5.0,
        warmup_epochs: int = 5,
    ) -> None:
        super().__init__()
        self.lambda_cls       = lambda_cls
        self.lambda_obj       = lambda_obj
        self.lambda_loc_start = lambda_loc_start
        self.lambda_loc_end   = lambda_loc_end
        self.warmup_epochs    = warmup_epochs

    def get_lambda_loc(self, current_epoch: int) -> float:
        """
        Linearly interpolate λ_loc from lambda_loc_start → lambda_loc_end
        over warmup_epochs, then hold at lambda_loc_end.
        """
        if self.warmup_epochs <= 0:
            return self.lambda_loc_end
        t = min(current_epoch / self.warmup_epochs, 1.0)
        return self.lambda_loc_start + t * (self.lambda_loc_end - self.lambda_loc_start)

    @staticmethod
    def iou_loss(
        pred_boxes: torch.Tensor,
        gt_boxes:   torch.Tensor,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        Compute 1 - IoU(B_pred, B_gt) element-wise.

        Box format: (x1, y1, x2, y2) – pixel or normalised coordinates.
        """
        # Intersection
        inter_x1 = torch.max(pred_boxes[:, 0], gt_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], gt_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], gt_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], gt_boxes[:, 3])

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter   = inter_w * inter_h

        # Union
        pred_area = ((pred_boxes[:, 2] - pred_boxes[:, 0]) *
                     (pred_boxes[:, 3] - pred_boxes[:, 1])).clamp(min=0)
        gt_area   = ((gt_boxes[:, 2]   - gt_boxes[:, 0]) *
                     (gt_boxes[:, 3]   - gt_boxes[:, 1])).clamp(min=0)
        union = pred_area + gt_area - inter + eps

        iou = inter / union
        return 1.0 - iou  # L_loc

    def forward(
        self,
        pred_boxes:  torch.Tensor,
        gt_boxes:    torch.Tensor,
        cls_loss:    torch.Tensor,
        obj_loss:    torch.Tensor,
        current_epoch: int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute L_total (eq. 7).

        Args:
            pred_boxes:    (N, 4) predicted bounding boxes [x1,y1,x2,y2]
            gt_boxes:      (N, 4) ground-truth bounding boxes [x1,y1,x2,y2]
            cls_loss:      scalar classification loss tensor
            obj_loss:      scalar objectness loss tensor
            current_epoch: current training epoch (for λ_loc warmup schedule)

        Returns:
            total_loss: scalar loss tensor
            loss_dict:  breakdown dict for logging {loc, cls, obj, lambda_loc}
        """
        lambda_loc = self.get_lambda_loc(current_epoch)

        if pred_boxes.numel() > 0 and gt_boxes.numel() > 0:
            loc_loss = self.iou_loss(pred_boxes, gt_boxes).mean()
        else:
            loc_loss = torch.tensor(0.0, device=cls_loss.device)

        total = (
            self.lambda_cls * cls_loss
            + self.lambda_obj * obj_loss
            + lambda_loc    * loc_loss
        )

        return total, {
            "loc_loss":   loc_loss.item(),
            "cls_loss":   cls_loss.item(),
            "obj_loss":   obj_loss.item(),
            "lambda_loc": lambda_loc,
        }


# ---------------------------------------------------------------------------
# Full assembled model (Backbone → DACSR → CARFTNeck → Detection head stub)
# ---------------------------------------------------------------------------

class PCBDefectModel(nn.Module):
    """
    Full PCB Defect Detection model (§4.3.1 – §4.3.3).

    Architecture:
        GeometryAwareBackbone
            ↓ (F8, F16, F32)
        DACSR applied to each scale independently
            ↓ (F8_r, F16_r, F32_r)
        CARFTNeck fuses multi-scale maps + Transformer context reasoning
            ↓ (B, 256, H/8, W/8)
        Detection Head (stub — attach YOLO head or custom head here)

    This model is used during training. For inference in the demo, the
    pre-trained YOLO weights (yolo11m.pt) are used via PCBDetector
    (src/model.py) for speed and compatibility.
    """

    BACKBONE_CHANNELS  = [128, 256, 512]  # F8/F16/F32 out-channels
    NECK_D_MODEL       = 256
    NECK_N_HEADS       = 8
    NECK_N_LAYERS      = 2
    NECK_FFN_HIDDEN    = 1024
    DACSR_REDUCTION    = 16

    def __init__(self, num_classes: int = 6) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Geometry-Aware Backbone (§4.3.1)
        self.backbone = GeometryAwareBackbone()

        # DACSR applied per scale (§4.3.2)
        self.dacsr_f8  = DACSR(self.BACKBONE_CHANNELS[0], self.DACSR_REDUCTION)
        self.dacsr_f16 = DACSR(self.BACKBONE_CHANNELS[1], self.DACSR_REDUCTION)
        self.dacsr_f32 = DACSR(self.BACKBONE_CHANNELS[2], self.DACSR_REDUCTION)

        # CARFT Neck (§4.3.3)
        self.carft = CARFTNeck(
            in_channels = self.BACKBONE_CHANNELS,
            d_model     = self.NECK_D_MODEL,
            n_heads     = self.NECK_N_HEADS,
            n_layers    = self.NECK_N_LAYERS,
            ffn_hidden  = self.NECK_FFN_HIDDEN,
        )

        # Lightweight detection head stub (anchor-free, per-location)
        self.detection_head = nn.Conv2d(
            self.NECK_D_MODEL,
            num_classes + 4 + 1,  # cls + bbox(4) + objectness
            kernel_size=1,
        )

        # Gradient-Balanced Localisation Loss (§4.3.4)
        self.loc_loss_fn = GradientBalancedLocLoss(
            lambda_loc_start = 2.0,
            lambda_loc_end   = 5.0,
            warmup_epochs    = 5,
        )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, 3, H, W) input PCB image tensor

        Returns:
            preds: (B, num_classes+5, H/8, W/8) detection predictions
        """
        # ── Backbone (§4.3.1) ──
        f8, f16, f32 = self.backbone(x)

        # ── DACSR per scale (§4.3.2) ──
        f8_r  = self.dacsr_f8(f8)
        f16_r = self.dacsr_f16(f16)
        f32_r = self.dacsr_f32(f32)

        # ── CARFT Neck (§4.3.3) ──
        fused = self.carft(f8_r, f16_r, f32_r)  # (B, 256, H/8, W/8)

        # ── Detection Head ──
        preds = self.detection_head(fused)       # (B, cls+5, H/8, W/8)
        return preds


# ---------------------------------------------------------------------------
# Utility: model summary helper
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_pcb_model(num_classes: int = 6) -> PCBDefectModel:
    """Factory helper — returns an initialised PCBDefectModel."""
    return PCBDefectModel(num_classes=num_classes)


if __name__ == "__main__":
    # Quick sanity-check: forward pass on a dummy 640×640 batch
    model = build_pcb_model(num_classes=6)
    dummy = torch.zeros(2, 3, 640, 640)

    f8, f16, f32 = model.backbone(dummy)
    print(f"Backbone outputs  — F8: {f8.shape}  F16: {f16.shape}  F32: {f32.shape}")

    preds = model(dummy)
    print(f"Detection output  — preds: {preds.shape}")
    print(f"Total parameters  — {count_parameters(model):,}")

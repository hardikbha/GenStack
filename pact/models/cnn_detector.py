"""
CNN Deepfake Detector — maximally different from CLIP ViT-L/14.

Why this is complementary to XGenDet:
  XGenDet (CLIP ViT-L/14):          THIS model (EfficientNet-B4 + SRM):
  - ViT architecture (attention)     - CNN architecture (convolutions)
  - Semantic features (CLIP)         - Texture/edge features (ImageNet)
  - Frozen backbone, 1.24M trained   - Fully fine-tuned, 19M trained
  - Patch-level (14×14 patches)      - Pixel-level (progressive pooling)
  - Misses noise residuals           - SRM branch captures noise explicitly

Together: CLIP catches semantic anomalies, CNN catches pixel anomalies.
Their errors are uncorrelated → ensemble gains are large.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


# ── SRM filter kernels (subset of 10 most discriminative) ─────────────────────
_SRM_KERNELS = np.array([
    [[ 0, 0, 0],[ 0, 1,-1],[ 0, 0, 0]],
    [[ 0, 0, 0],[ 0, 1, 0],[ 0,-1, 0]],
    [[ 0, 0, 0],[-1, 2,-1],[ 0, 0, 0]],
    [[ 0,-1, 0],[-1, 4,-1],[ 0,-1, 0]],
    [[-1,-1,-1],[-1, 8,-1],[-1,-1,-1]],
    [[-1, 2,-1],[ 2,-4, 2],[-1, 2,-1]],
    [[ 0, 0, 0],[ 0, 1,-2],[ 0, 1, 0]],
    [[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]],
    [[-1,-2,-1],[ 0, 0, 0],[ 1, 2, 1]],
    [[-1, 0, 1],[ 0, 0, 0],[ 1, 0,-1]],
], dtype=np.float32)


class CNNDetector(nn.Module):
    """
    EfficientNet-B4 (ImageNet pretrained) + SRM noise branch.
    Fully fine-tuned — opposite philosophy to XGenDet's frozen backbone.
    """

    def __init__(self, num_classes: int = 1, srm_dim: int = 128, dropout: float = 0.3):
        super().__init__()

        # ── Path 1: EfficientNet-B4 backbone (full fine-tune) ─────────────────
        eff = models.efficientnet_b4(weights='DEFAULT')
        self.backbone = eff.features          # → [B, 1792, 7, 7] at 224px
        self.pool     = nn.AdaptiveAvgPool2d(1)
        backbone_dim  = 1792

        # ── Path 2: SRM noise residual branch ────────────────────────────────
        num_filters = len(_SRM_KERNELS)
        kernels = torch.from_numpy(_SRM_KERNELS).unsqueeze(1)  # [10, 1, 3, 3]
        self.register_buffer("srm_weight", kernels)

        self.noise_cnn = nn.Sequential(
            nn.Conv2d(num_filters, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, srm_dim),
            nn.GELU(),
        )

        # ── Joint classifier ─────────────────────────────────────────────────
        fused_dim = backbone_dim + srm_dim  # 1792 + 128 = 1920
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def _extract_noise(self, x: torch.Tensor) -> torch.Tensor:
        gray = (0.299*x[:,0] + 0.587*x[:,1] + 0.114*x[:,2]).unsqueeze(1)
        residuals = F.conv2d(gray, self.srm_weight, padding=1)
        return torch.clamp(residuals / 3.0, -1.0, 1.0)

    def forward(self, x: torch.Tensor) -> dict:
        # Path 1: EfficientNet spatial features
        feat = self.backbone(x)                        # [B, 1792, H, W]
        feat = self.pool(feat).flatten(1)              # [B, 1792]

        # Path 2: SRM noise features
        noise = self._extract_noise(x)                 # [B, 10, H, W]
        noise_feat = self.noise_cnn(noise)             # [B, 128]

        # Fuse and classify
        fused = torch.cat([feat, noise_feat], dim=-1)  # [B, 1920]
        logit = self.classifier(fused)                 # [B, 1]
        conf  = torch.sigmoid(logit)

        return {"binary_logit": logit, "confidence": conf}

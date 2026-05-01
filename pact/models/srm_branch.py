"""
SRM (Steganalysis Rich Model) Noise Residual Branch for XGenDet.

Why this works for weak generators:
- ICLight / face relighting: changes pixel values but leaves residual lighting
  gradient artifacts invisible in RGB space but clear in noise domain
- Face swaps (FFIW, DeepFaceLab): boundary paste region has inconsistent
  noise fingerprint between face and background
- StarGANv2 / GAN generators: produce characteristic periodic noise patterns
  in high-frequency domain that CLIP's spatial attention ignores

Approach:
  1. Apply 30 fixed SRM high-pass kernels to grayscale image
     (from Fridrich & Kodovsky, "Rich Models for Steganalysis", IEEE Trans 2012)
  2. Compute absolute noise residual maps
  3. Run small CNN → 256-d feature vector
  4. Fuse into XGenDet classification head

SRM filters capture: nearest-neighbor, linear, cubic interpolation residuals,
edge gradients, and second-order statistics — exactly what forgery operations disturb.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ── 30 SRM high-pass filter kernels (3×3 each) ───────────────────────────────
# These are the standard SRM kernels used throughout image forensics literature.
# They are fixed (not learned) — they always capture noise regardless of generator.
_SRM_KERNELS_3x3 = np.array([
    # First-order finite difference
    [[ 0,  0,  0], [ 0,  1, -1], [ 0,  0,  0]],
    [[ 0,  0,  0], [ 0,  1,  0], [ 0, -1,  0]],
    [[ 0,  0,  0], [ 0,  1,  0], [-1,  0,  0]],
    [[ 0,  0,  0], [-1,  1,  0], [ 0,  0,  0]],
    # Second-order finite difference (Laplacian variants)
    [[ 0,  0,  0], [-1,  2, -1], [ 0,  0,  0]],
    [[ 0, -1,  0], [ 0,  2,  0], [ 0, -1,  0]],
    [[ 0,  0, -1], [ 0,  2,  0], [-1,  0,  0]],
    [[-1,  0,  0], [ 0,  2,  0], [ 0,  0, -1]],
    # 3×3 Laplacian
    [[ 0, -1,  0], [-1,  4, -1], [ 0, -1,  0]],
    [[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]],
    # Square filters (noise uniformity)
    [[-1,  2, -1], [ 2, -4,  2], [-1,  2, -1]],
    [[ 1, -2,  1], [-2,  4, -2], [ 1, -2,  1]],
    # Edge emphasis filters
    [[ 0,  0,  0], [ 0,  1, -2], [ 0,  1,  0]],
    [[ 0,  0,  0], [ 0,  1,  0], [ 0, -2,  1]],
    [[ 0,  0,  0], [-2,  1,  0], [ 0,  1,  0]],
    [[ 0, -2,  0], [ 0,  1,  0], [ 0,  1,  0]],
    # Diagonal filters
    [[ 0,  0,  0], [ 0,  1, -1], [ 0, -1,  1]],
    [[ 0,  0,  0], [-1,  2, -1], [ 1, -2,  1]],
    [[ 1, -2,  1], [ 0,  0,  0], [-1,  2, -1]],
    [[ 1, -2,  1], [-2,  4, -2], [ 1, -2,  1]],
    # Cross filters
    [[-1,  0,  1], [ 0,  0,  0], [ 1,  0, -1]],
    [[ 1,  0, -1], [ 0,  0,  0], [-1,  0,  1]],
    # Gradient magnitude approximations
    [[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]],
    [[-1,  0,  1], [-2,  0,  2], [-1,  0,  1]],
    [[ 1,  2,  1], [ 0,  0,  0], [-1, -2, -1]],
    [[ 1,  0, -1], [ 2,  0, -2], [ 1,  0, -1]],
    # High-pass variants
    [[ 0, -1,  0], [ 0,  2,  0], [ 0, -1,  0]],
    [[ 0,  0,  0], [-1,  2, -1], [ 0,  0,  0]],
    [[-1,  2, -1], [ 2, -4,  2], [-1,  2, -1]],
    [[ 0,  0,  0], [ 1, -2,  1], [ 0,  0,  0]],
], dtype=np.float32)  # shape: [30, 3, 3]


class SRMBranch(nn.Module):
    """
    SRM noise residual feature extractor.

    Extracts 30 high-frequency noise residual maps, then uses a compact CNN
    to produce a 256-dimensional forensic feature vector.
    """

    def __init__(self, output_dim: int = 256):
        super().__init__()
        self.output_dim = output_dim
        num_filters = len(_SRM_KERNELS_3x3)  # 30

        # Register fixed SRM filters as buffer (not trained)
        kernels = torch.from_numpy(_SRM_KERNELS_3x3)  # [30, 3, 3]
        kernels = kernels.unsqueeze(1)                  # [30, 1, 3, 3]
        self.register_buffer("srm_weight", kernels)

        # Learned CNN processes the 30-channel noise residual maps
        # Input: [B, 30, H, W]
        self.noise_cnn = nn.Sequential(
            # Stage 1: local noise patterns
            nn.Conv2d(num_filters, 64,  kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # Stage 2: spatial noise correlations
            nn.Conv2d(64,          128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # Stage 3: semantic noise regions
            nn.Conv2d(128,         256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(4),   # → [B, 256, 4, 4]
            nn.Flatten(),              # → [B, 4096]
            nn.Linear(256 * 4 * 4, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # Truncation: clip residuals to [-T, T] to reduce image content leakage
        self.T = 3.0

    def _extract_noise_residual(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SRM filters to grayscale image and return noise residual maps.

        Args:
            x: [B, 3, H, W] — normalized RGB image

        Returns:
            residuals: [B, 30, H, W]
        """
        # Convert to grayscale
        gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)

        # Apply all 30 SRM filters with 'same' padding
        residuals = F.conv2d(gray, self.srm_weight, padding=1)  # [B, 30, H, W]

        # Truncate and normalize
        residuals = torch.clamp(residuals / self.T, -1.0, 1.0)

        return residuals

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] input image

        Returns:
            srm_features: [B, output_dim]
        """
        noise_maps = self._extract_noise_residual(x)   # [B, 30, H, W]
        return self.noise_cnn(noise_maps)               # [B, output_dim]

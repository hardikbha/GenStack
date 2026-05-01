"""
BilateralResidualBranch: Illumination Inconsistency Detector for XGenDet v5+.

Why this works:
- FF++ face swaps: the pasted face region has different illumination statistics than
  the background — the face was captured under different lighting. After a large Gaussian
  blur (approximating a bilateral smooth), the residual exposes this lighting mismatch
  as a strong gradient ring at the paste boundary.
- ICLight synthetic relighting: relighting models introduce smooth synthetic lighting
  gradients that differ from natural illumination falloff. The residual (image minus
  large-sigma blur) captures this artificial gradient shape.
- The Sobel gradient of the residual specifically fires at the boundary edge, giving
  the subsequent CNN a precise spatial signal about where the inconsistency is sharpest.

Together, residual + gradient magnitude form a 4-channel map that the CNN encodes into
a 128-d forensic feature orthogonal to CLIP's semantic features and SRM noise patterns.
This is expected to provide the largest gain on the CF (Cross-Face) split of HydraFake,
where ICLight synthetic relighting gradients are the dominant forgery artifact.

Implementation note:
  Kornia is NOT available. Gaussian blur is implemented as a fixed 61×61 kernel
  registered as a buffer and applied per-channel via F.conv2d with groups.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel_2d(sigma: float, size: int) -> torch.Tensor:
    """
    Create a normalised 2-D Gaussian kernel.

    Args:
        sigma: Standard deviation of the Gaussian.
        size:  Kernel side length (should be odd; will be forced to odd internally).

    Returns:
        kernel_2d: Float tensor of shape [size, size] summing to 1.
    """
    coords = torch.arange(size).float() - size // 2          # [size]
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g = g / g.sum()
    kernel_2d = g.outer(g)                                    # [size, size]
    return kernel_2d


class BilateralResidualBranch(nn.Module):
    """
    Detect illumination inconsistencies at the face-paste boundary.

    Forward path:
        1. Smooth x with a large fixed Gaussian kernel (σ=12, 61×61) to obtain a
           low-frequency illumination estimate (approximates a bilateral filter in
           low-texture regions).
        2. residual = x - smooth(x)  — captures high-frequency + lighting mismatch
        3. Sobel gradient magnitude of the residual — fires at paste boundary edges
        4. Stack [residual(3ch), grad_mag(1ch)] → [B, 4, H, W]
        5. Small CNN → [B, output_dim]

    Args:
        output_dim:  Dimension of the output feature vector (default 128).
        sigma:       Gaussian blur std-dev (default 12 — large enough to span a
                     face region and remove low-freq lighting while preserving edges).
        kernel_size: Kernel side length (default 61, odd).
    """

    def __init__(self, output_dim: int = 128, sigma: float = 12.0, kernel_size: int = 61):
        super().__init__()

        # ── Fixed Gaussian kernel ─────────────────────────────────────────────
        # Shape [1, 1, K, K] — will be broadcast across channels via groups.
        k = _gaussian_kernel_2d(sigma, kernel_size)           # [K, K]
        k = k.unsqueeze(0).unsqueeze(0)                       # [1, 1, K, K]
        self.register_buffer("gauss_kernel", k)
        self.pad = kernel_size // 2

        # ── Fixed Sobel filters ───────────────────────────────────────────────
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        # ── CNN: 4-channel input → output_dim ────────────────────────────────
        # Input: residual (3 ch) + gradient magnitude (1 ch) = 4 channels.
        # Two strided convolutions reduce spatial resolution efficiently.
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(4),                          # → [B, 64, 4, 4]
            nn.Flatten(),                                     # → [B, 1024]
            nn.Linear(64 * 4 * 4, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the fixed Gaussian kernel channel-wise.

        Args:
            x: [B, 3, H, W]

        Returns:
            smoothed: [B, 3, H, W]
        """
        B, C, H, W = x.shape
        # Reshape to [B*C, 1, H, W] so we can apply the [1,1,K,K] kernel with
        # groups=1 (each channel treated independently).
        x_flat = x.view(B * C, 1, H, W)
        smooth = F.conv2d(x_flat, self.gauss_kernel, padding=self.pad)
        return smooth.view(B, C, H, W)

    def _gradient_mag(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Sobel gradient magnitude of the (grayscale) input.

        Args:
            x: [B, 3, H, W]

        Returns:
            mag: [B, 1, H, W]  — gradient magnitude, ≥ 0
        """
        gray = x.mean(dim=1, keepdim=True)                    # [B, 1, H, W]
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] — normalised RGB image (same as backbone input)

        Returns:
            features: [B, output_dim]
        """
        smooth = self._smooth(x)                              # [B, 3, H, W]
        residual = x - smooth                                 # [B, 3, H, W] — lighting inconsistency
        grad_mag = self._gradient_mag(residual)               # [B, 1, H, W] — edge anomalies at boundary
        combined = torch.cat([residual, grad_mag], dim=1)     # [B, 4, H, W]
        return self.cnn(combined)                             # [B, output_dim]

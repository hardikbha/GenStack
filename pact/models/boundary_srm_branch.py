"""
Boundary-SRM Branch for XGenDet — FaceForensics++ seam detection.

Why boundary-restricted SRM works for FF++:
- FF++ blending (DeepFakes, Face2Face, FaceSwap, NeuralTextures) pastes a
  synthesized face region onto a real background frame.  The synthesis is
  spatially confined to the face interior, so the noise fingerprint of the
  synthesized region is fundamentally different from the surrounding real-video
  background.  This mismatch is strongest exactly at the paste boundary: the
  seam is where two statistically different noise processes meet.
- Whole-image SRM (SRMBranch) dilutes this signal across 224×224 pixels — the
  boundary ring comprises only ~15% of the image area, so its residuals are
  dominated by the large uniform face/background interiors.
- CLIP spatial attention ignores high-frequency noise entirely.
- This branch applies SRM *only inside the boundary ring*, so every neuron in
  the downstream CNN sees exclusively the seam region.  No face detector is
  required at inference time because the images are already face-centered
  224×224 crops — the face occupies a predictable, stable elliptical region.

Design:
  1. Pre-compute a fixed elliptical ring mask as a registered buffer:
       outer ellipse (a=95, b=105) minus inner ellipse (a=70, b=80),
       both centered at (112, 112).
     The ring is ~25 px wide and encircles the face-background transition.
  2. Convert RGB to grayscale, apply all 30 SRM high-pass kernels.
  3. Multiply noise residuals by the ring mask → zero out everything outside
     the seam, keep only boundary noise.
  4. Small CNN (3 conv layers + AdaptiveAvgPool) → output_dim feature vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .srm_branch import _SRM_KERNELS_3x3


# ── Elliptical ring mask construction ────────────────────────────────────────

def _make_ellipse_ring(
    H: int,
    W: int,
    cx: float,
    cy: float,
    a_outer: float,
    b_outer: float,
    a_inner: float,
    b_inner: float,
) -> torch.Tensor:
    """
    Build a binary elliptical ring mask of shape [1, 1, H, W] (float32).

    The ring is defined as the set of pixels that fall inside the outer ellipse
    but outside the inner ellipse:

        outer_mask[y, x] = ((x - cx) / a_outer)^2 + ((y - cy) / b_outer)^2 <= 1
        inner_mask[y, x] = ((x - cx) / a_inner)^2 + ((y - cy) / b_inner)^2 <= 1
        ring = outer_mask AND NOT inner_mask

    Args:
        H, W  : Image height and width in pixels.
        cx, cy: Centre of both ellipses (column, row) in pixel coordinates.
        a_outer, b_outer: Semi-axes (horizontal, vertical) of the outer ellipse.
        a_inner, b_inner: Semi-axes (horizontal, vertical) of the inner ellipse.

    Returns:
        mask: float32 tensor of shape [1, 1, H, W] with values 0.0 or 1.0.
    """
    # Pixel coordinate grids — shape [H, W]
    ys = np.arange(H, dtype=np.float32)          # row indices
    xs = np.arange(W, dtype=np.float32)          # column indices
    xx, yy = np.meshgrid(xs, ys)                 # [H, W] each

    # Normalised squared distances from centre
    outer_dist = ((xx - cx) / a_outer) ** 2 + ((yy - cy) / b_outer) ** 2
    inner_dist = ((xx - cx) / a_inner) ** 2 + ((yy - cy) / b_inner) ** 2

    outer_mask = outer_dist <= 1.0               # bool [H, W]
    inner_mask = inner_dist <= 1.0               # bool [H, W]

    ring = outer_mask & ~inner_mask              # bool [H, W]
    ring_f = ring.astype(np.float32)             # float32 [H, W]

    # Reshape to [1, 1, H, W] for broadcasting against [B, C, H, W]
    ring_tensor = torch.from_numpy(ring_f).reshape(1, 1, H, W)
    return ring_tensor


# ── Module ────────────────────────────────────────────────────────────────────

class BoundarySRMBranch(nn.Module):
    """
    SRM noise residual branch restricted to the face-background boundary ring.

    Inputs : [B, 3, 224, 224] RGB images (face-centered crops, values in any
             range accepted by the upstream normalisation — typically [-1, 1]
             or [0, 1]).
    Outputs: [B, output_dim] feature vectors.

    The ring mask is pre-computed once and stored as a non-trainable buffer so
    it moves to the correct device automatically with .to(device).
    """

    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.output_dim = output_dim

        # ── Ring mask ─────────────────────────────────────────────────────────
        # Outer ellipse: semi-axes a=95 (horizontal), b=105 (vertical)
        #   → covers ~85% of the face diameter, clips right at the hairline /
        #     chin / cheek boundary.
        # Inner ellipse: semi-axes a=70, b=80
        #   → excludes the stable face interior (eyes, nose, mouth area).
        # The ring width is therefore roughly 25 px — wide enough to capture
        # the full blending gradient, narrow enough not to dilute the signal.
        mask = _make_ellipse_ring(
            H=224, W=224,
            cx=112.0, cy=112.0,
            a_outer=95.0, b_outer=105.0,
            a_inner=70.0, b_inner=80.0,
        )  # [1, 1, 224, 224] float32
        self.register_buffer("ring_mask", mask)

        # ── Fixed SRM filters ─────────────────────────────────────────────────
        # 30 high-pass kernels imported from SRMBranch — fixed, not trained.
        kernels = torch.from_numpy(_SRM_KERNELS_3x3).unsqueeze(1)  # [30,1,3,3]
        self.register_buffer("srm_weight", kernels)

        # Truncation threshold: clips residuals to [-T, T] to suppress image
        # content leakage while preserving noise structure.
        self.T = 3.0

        # ── CNN on masked noise residuals ─────────────────────────────────────
        # Input : [B, 30, 224, 224] — but most pixels are zeroed by the mask,
        #         so the effective receptive field is concentrated at the ring.
        # Stage 1 — 224 → 112: extract local noise texture at the seam
        # Stage 2 — 112 → 56:  aggregate noise correlations along the ring arc
        # AdaptiveAvgPool2d(4) → 4×4 spatial summary
        # Linear → LayerNorm → GELU → Dropout: project to output_dim
        self.noise_cnn = nn.Sequential(
            nn.Conv2d(30, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(4),   # → [B, 64, 4, 4]
            nn.Flatten(),              # → [B, 1024]
            nn.Linear(64 * 4 * 4, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 224, 224] — normalised RGB face crops.

        Returns:
            features: [B, output_dim]
        """
        # 1. Luminance (grayscale) — standard ITU-R BT.601 coefficients.
        #    Shape: [B, 1, 224, 224]
        gray = (
            0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        ).unsqueeze(1)

        # 2. Apply all 30 SRM high-pass filters with 'same' (reflect-style) padding.
        #    padding=1 preserves spatial size for 3×3 kernels.
        #    Output: [B, 30, 224, 224]
        residuals = F.conv2d(gray, self.srm_weight, padding=1)

        # 3. Truncate to [-1, 1] after dividing by T to normalise scale.
        residuals = torch.clamp(residuals / self.T, -1.0, 1.0)

        # 4. Zero out everything outside the boundary ring.
        #    ring_mask broadcasts over [B, 30, 224, 224] from [1, 1, 224, 224].
        residuals = residuals * self.ring_mask

        # 5. CNN → feature vector.
        return self.noise_cnn(residuals)   # [B, output_dim]

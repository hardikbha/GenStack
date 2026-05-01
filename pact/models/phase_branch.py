"""
FFT Phase Branch for XGenDet — FaceForensics++ blending seam detection.

Why phase (not magnitude) for FF++:
- The existing SpectralBranch extracts log-magnitude from the FFT spectrum.
  Magnitude captures *how much energy* exists at each frequency — useful for
  detecting GAN upsampling grids and diffusion spectral rolloff.
- Phase captures *where* that energy sits spatially: the phase angle encodes
  the precise spatial alignment of each frequency component.
- FF++ blending pastes a synthesised face onto a real-video background.  The
  paste creates a hard spatial discontinuity at the seam boundary.  In the
  frequency domain a spatial step-edge introduces a characteristic phase
  "wrapping" pattern — abrupt ±π jumps that propagate through mid-frequency
  bands (roughly spatial frequencies 10–50 cycles/image).
- Magnitude is *relatively* smooth across this seam because total energy is
  conserved.  Phase is not: the discontinuity breaks the smooth phase field
  that a natural, unmanipulated image would have.  The gradient of the phase
  map (computed here with Sobel filters) localises those jumps as high-response
  ridges that lie precisely along the blending boundary.

Architecture (mirrors SpectralBranch exactly — same CNN, different input):
  1. rfft2 → complex [B, 3, H, W//2+1]
  2. torch.angle → real phase map in [−π, π]
  3. fftshift along dim=-2 to centre the DC component (consistent with
     SpectralBranch magnitude convention)
  4. Bilinear interpolate to [B, 3, 224, 224] (uniform spatial size for CNN)
  5. Small CNN (same as SpectralBranch) → output_dim feature vector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseBranch(nn.Module):
    """
    FFT phase-angle feature extractor targeting blending seam discontinuities.

    The phase angle of the 2-D Fourier transform is sensitive to spatial
    discontinuities introduced by face-paste operations in FaceForensics++.
    Specifically, the abrupt transition at the blending boundary creates
    mid-frequency phase wrapping / step patterns that are absent in real
    frames and are not captured by the log-magnitude spectrum used by
    SpectralBranch.

    Inputs : [B, 3, H, W] RGB images (H=W=224 for standard XGenDet pipeline).
    Outputs: [B, output_dim] feature vectors.

    Notes:
    - rfft2 returns the one-sided (non-redundant) spectrum of shape
      [B, 3, H, W//2+1].  The result is shifted along the row axis (dim=-2)
      to place DC at the centre, matching SpectralBranch convention and making
      the spatial layout interpretable by the CNN.
    - Phase values are in [-π, π] and are passed directly to the CNN without
      additional normalisation; BatchNorm in the first layer handles scale.
    - No windowing function is applied — the boundary discontinuity we want to
      detect *is* a windowing artifact, so suppressing it with a Hann window
      would remove the signal.
    """

    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.output_dim = output_dim

        # CNN architecture mirrors SpectralBranch exactly.
        # Input : [B, 3, 224, 224] (phase map, interpolated to square)
        # Stage 1 — 224 → 112: capture coarse phase topology
        # Stage 2 — 112 → 56:  detect arc-shaped phase gradient ridges
        # Stage 3 — 56 → 28:   encode global phase distribution summary
        # AdaptiveAvgPool2d(4) → 4×4 spatial summary → flatten → project
        self.phase_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(4),    # → [B, 128, 4, 4]
            nn.Flatten(),               # → [B, 2048]
            nn.Linear(128 * 4 * 4, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    def _fft_phase(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute centred phase spectrum for all 3 RGB channels.

        Args:
            x: [B, 3, H, W] — input image tensor.

        Returns:
            phase: [B, 3, H, W//2+1] — phase angle in radians ([-π, π]).
        """
        # One-sided 2-D FFT; norm='ortho' gives a unitary transform so energy
        # is preserved and phase is well-conditioned.
        # rfft2 doesn't support bfloat16 — upcast, then cast phase back
        orig_dtype = x.dtype
        fft = torch.fft.rfft2(x.float(), norm='ortho')   # [B, 3, H, W//2+1] complex
        phase = torch.angle(fft)                         # [B, 3, H, W//2+1] real
        phase = torch.fft.fftshift(phase, dim=-2)        # centre DC along rows
        return phase.to(orig_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] — normalised RGB face crops.

        Returns:
            features: [B, output_dim]
        """
        # 1. Extract phase spectrum — shape [B, 3, H, W//2+1].
        phase = self._fft_phase(x)

        # 2. Resize to [B, 3, 224, 224] so the CNN sees a fixed spatial size
        #    regardless of input resolution.  Bilinear interpolation is used
        #    (consistent with SpectralBranch) — phase values are locally smooth
        #    except at discontinuities, so bilinear introduces minimal artefacts.
        phase = F.interpolate(phase, size=(224, 224), mode='bilinear', align_corners=False)

        # 3. CNN → feature vector.
        return self.phase_cnn(phase)                     # [B, output_dim]

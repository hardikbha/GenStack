"""
FFT Spectral Branch for XGenDet v2 (improved).

Extracts frequency-domain features invisible to CLIP spatial attention:
- GAN upsampling grid artifacts (peaks at N/2 in spectrum)
- Diffusion model spectral rolloff signature
- Face relighting/restoration boundary frequency inconsistencies
- Compression blocking artifacts (8x8 DCT peaks)

Architecture: 3-channel RGB → per-channel FFT → log-magnitude
             → learnable CNN → 256-d feature vector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralBranch(nn.Module):
    """
    FFT-based frequency analysis branch.
    Operates directly on log-magnitude spectrum with a small CNN.
    """

    def __init__(self, output_dim: int = 256, num_filters: int = 3):
        """
        Args:
            output_dim: Output feature dimension (default 256, merged into head)
            num_filters: Ignored — kept for API compatibility. Always uses 3-channel RGB FFT.
        """
        super().__init__()
        self.output_dim = output_dim
        in_channels = 3  # always RGB — rfft2 on 3 channels

        # CNN processes the log-magnitude spectrum
        # Input: [B, 3, H, W//2+1] → interpolated to [B, 3, 224, 224]
        self.spectral_cnn = nn.Sequential(
            # Stage 1: coarse spatial features from spectrum
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            # Stage 2
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # Stage 3: capture periodic artifact patterns
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(4),   # → [B, 128, 4, 4]
            nn.Flatten(),              # → [B, 2048]
            nn.Linear(128 * 4 * 4, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    def _fft_log_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute centered log-magnitude FFT spectrum for all 3 RGB channels.
        Returns: [B, 3, H, W//2+1]
        """
        # rfft2 doesn't support bfloat16 — upcast, then cast back to input dtype
        orig_dtype = x.dtype
        x_f32 = x.float()
        fft    = torch.fft.rfft2(x_f32, norm='ortho')   # [B, 3, H, W//2+1]
        mag    = torch.abs(fft)
        mag    = torch.fft.fftshift(mag, dim=-2)     # center DC component
        return torch.log1p(mag).to(orig_dtype)       # [B, 3, H, W//2+1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 224, 224]

        Returns:
            spectral_features: [B, output_dim]
        """
        log_mag = self._fft_log_magnitude(x)  # [B, C, H, W//2+1]
        # Resize to [B, C, 224, 224] so CNN sees uniform spatial size
        log_mag = F.interpolate(log_mag, size=(224, 224),
                                mode='bilinear', align_corners=False)
        return self.spectral_cnn(log_mag)

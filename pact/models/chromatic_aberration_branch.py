"""
Chromatic-aberration physics branch for deepfake detection.

Physics basis
-------------
All real refractive camera lenses exhibit chromatic aberration (CA), a
direct consequence of Snell's law: the refractive index of glass depends on
wavelength (dispersion), so light of different colors is bent by different
amounts when it passes through a lens element. The effect has two modes:

    * Longitudinal CA: different wavelengths focus at different depths
      along the optical axis, producing color fringes at defocused edges.
    * Lateral CA: different wavelengths project to slightly different
      positions on the sensor plane, producing a sub-pixel shift between
      the R, G, and B channels. The shift grows approximately linearly
      with distance from the optical center, so CA is radially symmetric
      in a natural photograph.

Because CA arises from the physics of refraction, every real photograph
captured through a glass lens carries a spatially coherent, radially
increasing, channel-shift signature at high-contrast edges.

GAN / diffusion image generators have no lens model. They therefore
produce one of three pathological CA signatures:

    1. Zero CA (perfectly registered R/G/B edges),
    2. Spatially incoherent CA (shifts vary randomly, not radially),
    3. Texture-copied CA that is inconsistent between face and
       background (common for face-swap pipelines).

This branch extracts per-channel Sobel edge magnitudes and builds two
signed edge-difference maps (R-G and B-G), which approximate the local
channel-offset mismatch that lateral CA induces. A small CNN then scores
the spatial coherence of this signature.

The approach follows the intuition of Mayer and Stamm (2018), who showed
that lateral chromatic aberration can be localized and used as a strong
forensic cue for detecting compositing and image manipulation. Earlier
work by Johnson and Farid (2006) used CA inconsistencies to detect image
splicing, and Chen et al. (2018) extended CA analysis to CNN-based
forensics.

References
----------
- Snell, W. "Law of refraction." c. 1621 (classical optics).
- Johnson, M. K. and Farid, H. "Exposing Digital Forgeries Through
  Chromatic Aberration." ACM Workshop on Multimedia & Security, 2006.
- Mayer, O. and Stamm, M. C. "Accurate and Efficient Image Forgery
  Detection Using Lateral Chromatic Aberration." IEEE Transactions on
  Information Forensics and Security, 2018.
- Chen, C., Zhao, X., and Stamm, M. C. "Forensic Analysis of Chromatic
  Aberration for Image Authentication." ICASSP, 2018.

Input  : x of shape [B, 3, 224, 224], ImageNet-normalized RGB.
Output : feature vector of shape [B, output_dim].
Works in BF16 and FP32.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ImageNet denormalization constants (standard CLIP / torchvision stats).
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class ChromaticAberrationBranch(nn.Module):
    """Physics-based deepfake detector using chromatic aberration.

    Real cameras produce a spatially coherent, sub-pixel R/G/B offset at
    high-contrast edges due to wavelength-dependent lens refraction
    (Snell's law applied to a dispersive medium). GAN and diffusion image
    generators lack any lens model and therefore produce either zero CA or
    spatially incoherent CA. This module extracts a 5-channel CA physics
    map from the input and passes it through a small CNN head.

    Parameters
    ----------
    output_dim : int
        Dimensionality of the returned feature vector.
    """

    def __init__(self, output_dim: int = 128):
        super().__init__()

        # ImageNet denormalization buffers (move with .to(device/dtype)).
        self.register_buffer("inet_mean", _IMAGENET_MEAN.clone())
        self.register_buffer("inet_std", _IMAGENET_STD.clone())

        # Per-channel Sobel kernels (applied channel-by-channel, not grouped,
        # so each edge map retains its full resolution before comparison).
        sx = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sy = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sx)
        self.register_buffer("sobel_y", sy)

        # CNN on 5-channel physics map:
        #   [edge_R(1), edge_G(1), edge_B(1), CA_RG(1), CA_BG(1)]
        self.cnn = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map ImageNet-normalized tensor back to linear [0, 1] RGB."""
        mean = self.inet_mean.to(dtype=x.dtype)
        std = self.inet_std.to(dtype=x.dtype)
        return torch.clamp(x * std + mean, 0.0, 1.0)

    def _channel_edges(self, img: torch.Tensor, ch: int) -> torch.Tensor:
        """Sobel edge magnitude map for a single channel.

        Parameters
        ----------
        img : torch.Tensor of shape [B, 3, H, W].
        ch  : channel index (0=R, 1=G, 2=B).

        Returns
        -------
        torch.Tensor of shape [B, 1, H, W] with the Sobel gradient
        magnitude of the selected channel. A small epsilon is added under
        the square root for BF16 stability.
        """
        c = img[:, ch:ch + 1]
        kx = self.sobel_x.to(dtype=c.dtype)
        ky = self.sobel_y.to(dtype=c.dtype)
        gx = F.conv2d(c, kx, padding=1)
        gy = F.conv2d(c, ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-8)

    def _ca_offset_approximation(self, ch_a: torch.Tensor,
                                 ch_b: torch.Tensor) -> torch.Tensor:
        """Approximate the local lateral-CA offset between two channels.

        A true sub-pixel lateral offset between two channels produces
        proportionally different Sobel responses at high-contrast edges.
        Their signed pixel-wise difference therefore approximates the
        local CA mismatch while remaining differentiable. The sign is
        preserved so the downstream CNN can distinguish R-ward from
        B-ward dispersion, which is part of the radial CA signature.

        Parameters
        ----------
        ch_a, ch_b : torch.Tensor of shape [B, 1, H, W].

        Returns
        -------
        torch.Tensor of shape [B, 1, H, W].
        """
        return ch_a - ch_b

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute chromatic-aberration physics features.

        Parameters
        ----------
        x : torch.Tensor of shape [B, 3, 224, 224], ImageNet-normalized.

        Returns
        -------
        torch.Tensor of shape [B, output_dim].
        """
        img = self._denormalize(x)

        # Per-channel Sobel edge magnitude maps.
        e_r = self._channel_edges(img, 0)  # [B, 1, H, W]
        e_g = self._channel_edges(img, 1)  # [B, 1, H, W]
        e_b = self._channel_edges(img, 2)  # [B, 1, H, W]

        # Signed CA offset approximations (green used as reference, as is
        # standard in demosaicing and CA-correction pipelines because the
        # Bayer grid samples G at twice the rate of R and B).
        ca_rg = self._ca_offset_approximation(e_r, e_g)  # [B, 1, H, W]
        ca_bg = self._ca_offset_approximation(e_b, e_g)  # [B, 1, H, W]

        # Stack 5-channel physics map and run the CNN head.
        feats = torch.cat([e_r, e_g, e_b, ca_rg, ca_bg], dim=1)  # [B, 5, H, W]
        return self.cnn(feats)

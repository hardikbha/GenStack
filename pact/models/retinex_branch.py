"""
Retinex-based physics branch for deepfake detection.

Physics basis
-------------
The Retinex theory of image formation (Land, 1971; Land & McCann, 1971;
Horn, 1974) models an image as the product of two factors:

    I(x, y) = R(x, y) * L(x, y)

where R is the surface reflectance (an intrinsic property of the scene) and
L is the spatially varying illumination. Taking the logarithm separates the
two components additively:

    log I = log R + log L

For a real photograph captured in a coherent scene, the illumination field L
is smooth almost everywhere: light sources and their inter-reflections vary
slowly across space. Sharp gradients in log_L therefore almost never occur
inside a real image.

A deepfake that pastes a synthetic or re-illuminated face into a host image
violates this assumption: the illumination of the face region is lit by a
different (often estimated or hallucinated) light field, so the composite
image produces a step discontinuity in the estimated illumination map at
the paste boundary. Even purely generative fakes (GAN / diffusion) often
fail to produce a globally coherent L because each patch is generated under
a local lighting prior.

This branch therefore:

    1. Denormalizes the input from ImageNet statistics back to linear RGB,
    2. Computes log I,
    3. Estimates log L by a large-sigma Gaussian blur of log I
       (the classical Single-Scale Retinex estimator, Jobson et al. 1997,
       derived from Horn 1974's low-pass illumination assumption),
    4. Derives log R = log I - log L,
    5. Computes the gradient magnitude |grad log L| which should be near
       zero globally and spike at paste seams or generator inconsistencies,
    6. Feeds the resulting 7-channel physics map to a small CNN.

References
----------
- Land, E. H. "The Retinex Theory of Color Vision." Scientific American,
  1971.
- Land, E. H. and McCann, J. J. "Lightness and Retinex Theory." JOSA, 1971.
- Horn, B. K. P. "Determining Lightness from an Image." Computer Graphics
  and Image Processing, 1974.
- Jobson, D. J., Rahman, Z., and Woodell, G. A. "Properties and
  Performance of a Center/Surround Retinex." IEEE TIP, 1997.

Input  : x of shape [B, 3, 224, 224], ImageNet-normalized RGB.
Output : feature vector of shape [B, output_dim].
Works in BF16 and FP32.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # noqa: F401  (kept for downstream utilities / parity)


# ImageNet denormalization constants (standard CLIP / torchvision stats).
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _gaussian_kernel_2d(sigma: float, size: int) -> torch.Tensor:
    """Build a 2-D isotropic Gaussian kernel, normalized to sum to 1.

    Parameters
    ----------
    sigma : float
        Standard deviation in pixels.
    size : int
        Kernel side length (odd integer recommended).

    Returns
    -------
    torch.Tensor of shape [size, size].
    """
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g = g / g.sum()
    return g.outer(g)


class RetinexBranch(nn.Module):
    """Physics-based deepfake detector using Retinex theory (Land 1971).

    Decomposes an input image into illumination (L, low-frequency) and
    reflectance (R, high-frequency) and measures the spatial coherence of
    the illumination map. Real photographs produce smooth L; deepfakes
    violate the image-formation model by producing inconsistent L at paste
    boundaries or within generator-hallucinated regions.

    Parameters
    ----------
    output_dim : int
        Dimensionality of the returned feature vector.
    sigma : float
        Standard deviation (in pixels) of the Gaussian used to approximate
        the illumination low-pass (Horn 1974).
    kernel_size : int
        Spatial extent of the Gaussian kernel. Should be roughly 4*sigma+1.
    """

    def __init__(self, output_dim: int = 256, sigma: float = 15.0,
                 kernel_size: int = 61):
        super().__init__()

        # ImageNet denormalization buffers (move with .to(device/dtype)).
        self.register_buffer("inet_mean", _IMAGENET_MEAN.clone())
        self.register_buffer("inet_std", _IMAGENET_STD.clone())

        # Gaussian kernel for illumination estimation, shape [1, 1, K, K].
        gk = _gaussian_kernel_2d(sigma, kernel_size).unsqueeze(0).unsqueeze(0)
        self.register_buffer("gauss_kernel", gk)
        self.pad = kernel_size // 2

        # Sobel filters for gradient magnitude (seam detector on log_L).
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

        # CNN on 7-channel physics map: [log_L(3), log_R(3), |grad_log_L|(1)].
        self.cnn = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=5, stride=2, padding=2),
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
        """Map ImageNet-normalized tensor back to linear [0, 1] RGB.

        A small positive floor (0.01) is used because the subsequent log
        transform is undefined at zero and numerically unstable near zero.
        """
        mean = self.inet_mean.to(dtype=x.dtype)
        std = self.inet_std.to(dtype=x.dtype)
        return torch.clamp(x * std + mean, 0.01, 1.0)

    def _smooth_per_channel(self, x: torch.Tensor) -> torch.Tensor:
        """Per-channel Gaussian blur via the grouped-conv trick.

        Parameters
        ----------
        x : torch.Tensor of shape [B, C, H, W].

        Returns
        -------
        Smoothed tensor of the same shape.
        """
        B, C, H, W = x.shape
        flat = x.reshape(B * C, 1, H, W)
        kernel = self.gauss_kernel.to(dtype=x.dtype)
        smooth = F.conv2d(flat, kernel, padding=self.pad)
        return smooth.reshape(B, C, H, W)

    def _grad_mag(self, x: torch.Tensor) -> torch.Tensor:
        """Sobel gradient magnitude of a 1- or 3-channel map.

        When x has 3 channels the mean (luminance proxy) is used. The
        magnitude is regularized with a small epsilon for BF16 stability.
        """
        gray = x.mean(dim=1, keepdim=True) if x.shape[1] > 1 else x
        kx = self.sobel_x.to(dtype=gray.dtype)
        ky = self.sobel_y.to(dtype=gray.dtype)
        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-8)

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Retinex physics features.

        Parameters
        ----------
        x : torch.Tensor of shape [B, 3, 224, 224], ImageNet-normalized.

        Returns
        -------
        torch.Tensor of shape [B, output_dim].
        """
        # 1. Denormalize to linear [0, 1] RGB.
        img = self._denormalize(x)
        # 2. Move to log space (Retinex additive decomposition).
        log_I = torch.log(img + 1e-4)
        # 3. Estimate log L by a large Gaussian blur (Horn 1974, Jobson 1997).
        log_L = self._smooth_per_channel(log_I)
        # 4. Reflectance log R = log I - log L.
        log_R = log_I - log_L
        # 5. Gradient of the illumination map: smooth except at seams.
        grad_log_L = self._grad_mag(log_L)
        # 6. Stack physics channels and run a small CNN head.
        feats = torch.cat([log_L, log_R, grad_log_L], dim=1)  # [B, 7, H, W]
        return self.cnn(feats)

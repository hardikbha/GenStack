"""
XGenDet v4 Augmentations — Native-Crop Consistency Fix.

Key change over v1:
  Training preprocessing now stochastically uses BOTH:
    (a) Resize(224,224)           — standard view (large images like ICLight 1536px)
    (b) Resize(256,256)+CenterCrop(224) — native-crop view (256px images: FF++, FFIW, StarGANv2)

  At inference, TTA averages these two views + hflip.
  Training-test mismatch is eliminated: the model is optimized for the same
  distribution it will see at test time via TTA.

Why this matters:
  FF++ (256px), FFIW (256px), StarGANv2 (256px) all have pixel-level
  manipulation artifacts that bilinear Resize(224) interpolates away.
  CenterCrop(224) from a 256px image preserves them at native resolution.
"""

import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from random import random, uniform
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


def jpeg_compress(img: np.ndarray, quality: int) -> np.ndarray:
    img_pil = Image.fromarray(img)
    buffer = BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    result = np.array(Image.open(buffer))
    buffer.close()
    return result


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


class NativeCropResize:
    """
    Stochastically chooses between two resize strategies:
      - p_native: Resize(256) + CenterCrop(224)  — preserves 256px artifacts
      - 1-p_native: Resize(224,224)               — standard downsampling

    During training both paths are seen, making the model robust to both.
    At TTA time, both views are used and averaged.
    """

    def __init__(self, crop_size: int = 224, p_native: float = 0.5):
        self.crop_size = crop_size
        self.p_native  = p_native

    def __call__(self, img: Image.Image) -> Image.Image:
        if random() < self.p_native:
            # Native-crop path: resize to intermediate size, then center crop.
            # For crop_size=224: Resize(256)+CenterCrop(224) — 256px images kept at native res.
            # For crop_size=336: Resize(384)+CenterCrop(336) — same benefit at higher res.
            intermediate = int(self.crop_size * (256 / 224))  # 256 for 224, 384 for 336
            img = img.resize((intermediate, intermediate), Image.BILINEAR)
            img = TF.center_crop(img, self.crop_size)
        else:
            # Standard path
            img = img.resize((self.crop_size, self.crop_size), Image.BILINEAR)
        return img


class ForgeryAugmentationV4:
    """
    Same as v1/v2 but with moderate settings.
    JPEG range (20, 95) — not too aggressive, covers compression artifacts.
    ColorJitter 0.3 — moderate, avoids over-augmentation trap of v3.
    """

    def __init__(
        self,
        jpeg_prob: float = 0.5,
        jpeg_quality_range: tuple = (20, 95),
        blur_prob: float = 0.4,
        blur_sigma_range: tuple = (0.1, 2.0),
        color_jitter_prob: float = 0.3,
        noise_prob: float = 0.15,
    ):
        self.jpeg_prob          = jpeg_prob
        self.jpeg_quality_range = jpeg_quality_range
        self.blur_prob          = blur_prob
        self.blur_sigma_range   = blur_sigma_range
        self.color_jitter_prob  = color_jitter_prob
        self.noise_prob         = noise_prob
        self.jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        img_array = np.array(img)

        if random() < self.blur_prob:
            img_array = gaussian_blur(img_array, uniform(*self.blur_sigma_range))

        if random() < self.jpeg_prob:
            img_array = jpeg_compress(img_array, int(uniform(*self.jpeg_quality_range)))

        if random() < self.noise_prob:
            noise = np.random.normal(0, uniform(1, 5), img_array.shape).astype(np.float32)
            img_array = np.clip(img_array.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        img = Image.fromarray(img_array)

        if random() < self.color_jitter_prob:
            img = self.jitter(img)

        return img


def get_train_transforms_v4(
    crop_size: int = 224,
    jpeg_prob: float = 0.5,
    blur_prob: float = 0.4,
    p_native: float = 0.5,
) -> transforms.Compose:
    return transforms.Compose([
        NativeCropResize(crop_size=crop_size, p_native=p_native),
        ForgeryAugmentationV4(jpeg_prob=jpeg_prob, blur_prob=blur_prob),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def get_eval_transforms(crop_size: int = 224) -> transforms.Compose:
    """Standard eval — same as v1. TTA handles multi-view at inference."""
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])

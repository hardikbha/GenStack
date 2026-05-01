"""
XGenDet Data Augmentation Pipeline.

Includes standard augmentations plus forgery-relevant augmentations
(JPEG compression, Gaussian blur) for robustness.
"""

import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from random import random, uniform
import torchvision.transforms as transforms


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def jpeg_compress(img: np.ndarray, quality: int) -> np.ndarray:
    """Apply JPEG compression to image array."""
    img_pil = Image.fromarray(img)
    buffer = BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    img_compressed = Image.open(buffer)
    result = np.array(img_compressed)
    buffer.close()
    return result


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur to image array."""
    if sigma <= 0:
        return img
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


class ForgeryAugmentation:
    """Apply forgery-relevant augmentations (JPEG + blur)."""

    def __init__(
        self,
        jpeg_prob: float = 0.5,
        jpeg_quality_range: tuple = (30, 100),
        blur_prob: float = 0.5,
        blur_sigma_range: tuple = (0.1, 3.0),
    ):
        self.jpeg_prob = jpeg_prob
        self.jpeg_quality_range = jpeg_quality_range
        self.blur_prob = blur_prob
        self.blur_sigma_range = blur_sigma_range

    def __call__(self, img: Image.Image) -> Image.Image:
        img_array = np.array(img)

        if random() < self.blur_prob:
            sigma = uniform(*self.blur_sigma_range)
            img_array = gaussian_blur(img_array, sigma)

        if random() < self.jpeg_prob:
            quality = int(uniform(*self.jpeg_quality_range))
            img_array = jpeg_compress(img_array, quality)

        return Image.fromarray(img_array)


def get_train_transforms(
    crop_size: int = 224,
    jpeg_prob: float = 0.5,
    blur_prob: float = 0.5,
) -> transforms.Compose:
    """Get training transforms with augmentations."""
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        ForgeryAugmentation(
            jpeg_prob=jpeg_prob,
            blur_prob=blur_prob,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def get_eval_transforms(crop_size: int = 224) -> transforms.Compose:
    """Get evaluation transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])

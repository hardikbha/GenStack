"""
XGenDet v3 Augmentations — Targeted Hard-Generator Augmentations.

Key additions over v2:
- Stronger ColorJitter (brightness=0.6, saturation=0.6, hue=0.15) → targets ICLight face relighting
- Aggressive JPEG (quality 10-50) → targets hailuo video compression artifacts
- Hard downscale pathway 128→224 (p=0.5) → targets StarGANv2 256px source images
- RandomGrayscale (p=0.1) → robustness to color manipulation
"""

import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from random import random, uniform, choice
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


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


class ResolutionAwareAugmentationV3:
    """
    Enhanced resolution-aware augmentation.
    Adds harder downscale pathway to simulate StarGANv2 (256→224) more aggressively.
    """

    def __init__(self, target_size=224, prob=0.5):
        self.target_size = target_size
        self.prob = prob

    def __call__(self, img: Image.Image) -> Image.Image:
        if random() > self.prob:
            return img

        w, h = img.size
        pathway = choice(['upsample', 'downsample', 'multistep', 'crop_resize',
                           'starganv2_sim', 'starganv2_sim'])  # double weight on starganv2_sim

        if pathway == 'upsample':
            small = choice([128, 192, 256])
            img = img.resize((small, small), Image.BILINEAR)
            img = img.resize((w, h), Image.BICUBIC)

        elif pathway == 'downsample':
            large = choice([1024, 1280, 1536])
            img = img.resize((large, large), Image.BICUBIC)
            img = img.resize((w, h), Image.BILINEAR)

        elif pathway == 'multistep':
            steps = choice([[512, 256], [1024, 512], [768, 384], [256, 512]])
            for s in steps:
                img = img.resize((s, s), choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]))

        elif pathway == 'crop_resize':
            scale = uniform(0.6, 0.95)
            crop_w, crop_h = int(w * scale), int(h * scale)
            left = int(uniform(0, w - crop_w))
            top = int(uniform(0, h - crop_h))
            img = img.crop((left, top, left + crop_w, top + crop_h))

        elif pathway == 'starganv2_sim':
            # Simulate StarGANv2: downscale to 128-256px (their generation size), back up
            target = choice([128, 160, 192, 224, 256])
            interp_down = choice([Image.BILINEAR, Image.NEAREST])
            interp_up   = choice([Image.BILINEAR, Image.BICUBIC])
            img = img.resize((target, target), interp_down)
            img = img.resize((w, h), interp_up)

        return img


class ForgeryAugmentationV3:
    """
    Stronger forgery augmentations targeting hard generators.

    Changes from v2:
    - ColorJitter: 0.2→0.6 brightness/contrast/saturation, 0.05→0.15 hue (ICLight)
    - JPEG quality range: (20,100)→(10,60) — aggressive video compression (hailuo)
    - Added RandomGrayscale p=0.1 (lighting manipulation robustness)
    - Noise sigma range: (1,5)→(1,8) — stronger noise simulation
    """

    def __init__(
        self,
        jpeg_prob=0.6,
        jpeg_quality_range=(10, 60),     # much harder than v2's (20, 100)
        blur_prob=0.4,
        blur_sigma_range=(0.1, 2.0),
        color_jitter_prob=0.5,            # up from 0.3
        grayscale_prob=0.1,               # NEW: for lighting robustness
        noise_prob=0.2,
    ):
        self.jpeg_prob = jpeg_prob
        self.jpeg_quality_range = jpeg_quality_range
        self.blur_prob = blur_prob
        self.blur_sigma_range = blur_sigma_range
        self.color_jitter_prob = color_jitter_prob
        self.grayscale_prob = grayscale_prob
        self.noise_prob = noise_prob

        # Strong color jitter for ICLight (face relighting changes brightness/saturation)
        self.strong_jitter = transforms.ColorJitter(
            brightness=0.6, contrast=0.6, saturation=0.6, hue=0.15
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        img_array = np.array(img)

        if random() < self.blur_prob:
            sigma = uniform(*self.blur_sigma_range)
            img_array = gaussian_blur(img_array, sigma)

        if random() < self.jpeg_prob:
            quality = int(uniform(*self.jpeg_quality_range))
            img_array = jpeg_compress(img_array, quality)

        if random() < self.noise_prob:
            noise = np.random.normal(0, uniform(1, 8), img_array.shape).astype(np.float32)
            img_array = np.clip(img_array.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        img = Image.fromarray(img_array)

        if random() < self.color_jitter_prob:
            img = self.strong_jitter(img)

        if random() < self.grayscale_prob:
            img = TF.to_grayscale(img, num_output_channels=3)

        return img


def get_train_transforms_v3(
    crop_size: int = 224,
    jpeg_prob: float = 0.6,
    blur_prob: float = 0.4,
    resolution_aug_prob: float = 0.5,
) -> transforms.Compose:
    return transforms.Compose([
        ResolutionAwareAugmentationV3(target_size=crop_size, prob=resolution_aug_prob),
        transforms.Resize((crop_size, crop_size)),
        ForgeryAugmentationV3(jpeg_prob=jpeg_prob, blur_prob=blur_prob),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def get_eval_transforms(crop_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])

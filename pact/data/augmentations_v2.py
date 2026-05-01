"""
XGenDet v2 Augmentations — Resolution-Aware + Standard.

Key addition: Simulates different resolution pathways that test images go through.
ICLight is 1536→224, StarGANv2 is 256→224, training is 1024→224.
The model must learn that the SAME artifacts look different at different scales.
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


class ResolutionAwareAugmentation:
    """
    Simulates different resolution pathways:
    1. Direct resize (current approach)
    2. Upsample first then resize (simulates low-res test images like DeepFaceLab 256→224)
    3. Downsample first then resize (simulates high-res test images like ICLight 1536→224)
    4. Multi-step resize (simulates social media resampling pipeline)
    """

    def __init__(self, target_size=224, prob=0.5):
        self.target_size = target_size
        self.prob = prob
        self.intermediate_sizes = [128, 256, 384, 512, 768, 1024, 1536]

    def __call__(self, img: Image.Image) -> Image.Image:
        if random() > self.prob:
            return img

        w, h = img.size
        pathway = choice(['upsample', 'downsample', 'multistep', 'crop_resize'])

        if pathway == 'upsample':
            # First downsample to small, then upsample — simulates low-res generators
            small = choice([128, 192, 256])
            img = img.resize((small, small), Image.BILINEAR)
            img = img.resize((w, h), Image.BICUBIC)

        elif pathway == 'downsample':
            # First upsample to large, then back — simulates high-res generators
            large = choice([1024, 1280, 1536])
            img = img.resize((large, large), Image.BICUBIC)
            img = img.resize((w, h), Image.BILINEAR)

        elif pathway == 'multistep':
            # Multi-step resize chain (social media pipeline simulation)
            steps = choice([
                [512, 256],
                [1024, 512],
                [768, 384],
                [256, 512],
            ])
            for s in steps:
                img = img.resize((s, s), choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]))

        elif pathway == 'crop_resize':
            # Random crop at different scale then resize back
            scale = uniform(0.6, 0.95)
            crop_w, crop_h = int(w * scale), int(h * scale)
            left = int(uniform(0, w - crop_w))
            top = int(uniform(0, h - crop_h))
            img = img.crop((left, top, left + crop_w, top + crop_h))

        return img


class ForgeryAugmentationV2:
    """Enhanced forgery augmentations: JPEG + blur + color jitter + noise."""

    def __init__(
        self,
        jpeg_prob=0.5,
        jpeg_quality_range=(20, 100),
        blur_prob=0.5,
        blur_sigma_range=(0.1, 3.0),
        color_jitter_prob=0.3,
        noise_prob=0.2,
    ):
        self.jpeg_prob = jpeg_prob
        self.jpeg_quality_range = jpeg_quality_range
        self.blur_prob = blur_prob
        self.blur_sigma_range = blur_sigma_range
        self.color_jitter_prob = color_jitter_prob
        self.noise_prob = noise_prob

    def __call__(self, img: Image.Image) -> Image.Image:
        img_array = np.array(img)

        if random() < self.blur_prob:
            sigma = uniform(*self.blur_sigma_range)
            img_array = gaussian_blur(img_array, sigma)

        if random() < self.jpeg_prob:
            quality = int(uniform(*self.jpeg_quality_range))
            img_array = jpeg_compress(img_array, quality)

        if random() < self.noise_prob:
            noise = np.random.normal(0, uniform(1, 5), img_array.shape).astype(np.float32)
            img_array = np.clip(img_array.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        img = Image.fromarray(img_array)

        if random() < self.color_jitter_prob:
            jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            img = jitter(img)

        return img


def get_train_transforms_v2(
    crop_size: int = 224,
    jpeg_prob: float = 0.5,
    blur_prob: float = 0.5,
    resolution_aug_prob: float = 0.4,
) -> transforms.Compose:
    return transforms.Compose([
        ResolutionAwareAugmentation(target_size=crop_size, prob=resolution_aug_prob),
        transforms.Resize((crop_size, crop_size)),
        ForgeryAugmentationV2(
            jpeg_prob=jpeg_prob,
            blur_prob=blur_prob,
        ),
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

"""
XGenDet Dataset: RealFakeDataset with generator family labels.

Supports loading from multiple generator directories with automatic
family label assignment (Real=0, GAN=1, Diffusion=2, Autoregressive=3).
"""

import os
from typing import List, Optional, Tuple
from random import shuffle as random_shuffle

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image, ImageFile

from .augmentations import get_train_transforms, get_eval_transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Generator family mapping
GENERATOR_FAMILIES = {
    # GANs
    "progan": 1, "ProGAN": 1, "stylegan": 1, "StyleGAN": 1,
    "stylegan2": 1, "StyleGAN2": 1, "biggan": 1, "BigGAN": 1,
    "cyclegan": 1, "CycleGAN": 1, "stargan": 1, "StarGAN": 1,
    "gaugan": 1, "GauGAN": 1, "crn": 1, "CRN": 1,
    "imle": 1, "IMLE": 1, "san": 1, "SAN": 1,
    "deepfake": 1, "whichfaceisreal": 1,
    # Diffusion models
    "ADM": 2, "adm": 2, "glide": 2, "GLIDE": 2,
    "LDM": 2, "ldm": 2, "VQDM": 2, "vqdm": 2,
    "dalle": 2, "DALLE": 2, "wukong": 2,
    "stable_diffusion": 2, "sd": 2,
    "seeingdark": 2, "SeeingDark": 2,
    # Midjourney is diffusion-based (latent diffusion architecture)
    "Midjourney": 2, "midjourney": 2,
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def scan_image_folder(folder: str) -> List[str]:
    """Recursively scan folder for image files."""
    images = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                images.append(os.path.join(root, f))
    return images


def infer_generator_family(generator_name: str) -> int:
    """Infer generator family from name. Returns 0 for unknown/real."""
    for key, family in GENERATOR_FAMILIES.items():
        if key.lower() in generator_name.lower():
            return family
    return 0  # Default to Real/Unknown


class RealFakeDataset(Dataset):
    """Dataset for real/fake image classification with generator family labels."""

    def __init__(
        self,
        real_folders: List[str],
        fake_folders: List[str],
        generator_names: Optional[List[str]] = None,
        is_train: bool = True,
        crop_size: int = 224,
        max_samples_per_class: Optional[int] = None,
        jpeg_prob: float = 0.5,
        blur_prob: float = 0.5,
    ):
        self.is_train = is_train

        # Set up transforms
        if is_train:
            self.transform = get_train_transforms(crop_size, jpeg_prob, blur_prob)
        else:
            self.transform = get_eval_transforms(crop_size)

        # Collect image paths
        real_images = []
        for folder in real_folders:
            if os.path.isdir(folder):
                real_images.extend(scan_image_folder(folder))

        fake_images = []
        for folder in fake_folders:
            if os.path.isdir(folder):
                fake_images.extend(scan_image_folder(folder))

        # Limit samples if requested
        if max_samples_per_class is not None:
            if len(real_images) > max_samples_per_class:
                random_shuffle(real_images)
                real_images = real_images[:max_samples_per_class]
            if len(fake_images) > max_samples_per_class:
                random_shuffle(fake_images)
                fake_images = fake_images[:max_samples_per_class]

        # Build data list: (path, binary_label, family_label)
        self.data = []
        for path in real_images:
            self.data.append((path, 0, 0))  # Real, family=0

        # Infer family from generator names or folder names
        for path in fake_images:
            family = 0
            # Try to infer from generator_names
            if generator_names:
                for gen_name in generator_names:
                    if gen_name.lower() in path.lower():
                        family = infer_generator_family(gen_name)
                        break
            else:
                # Infer from path
                for key in GENERATOR_FAMILIES:
                    if key.lower() in path.lower():
                        family = GENERATOR_FAMILIES[key]
                        break
            self.data.append((path, 1, family))

        random_shuffle(self.data)
        print(f"Dataset: {len(real_images)} real + {len(fake_images)} fake = {len(self.data)} total")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        path, label, family = self.data[idx]
        try:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
            return img, label, family
        except Exception:
            # Skip broken images
            return self.__getitem__((idx + 1) % len(self.data))


def create_dataloader(
    real_folders: List[str],
    fake_folders: List[str],
    is_train: bool = True,
    batch_size: int = 64,
    num_workers: int = 8,
    max_samples_per_class: Optional[int] = None,
    generator_names: Optional[List[str]] = None,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader for training or evaluation."""
    dataset = RealFakeDataset(
        real_folders=real_folders,
        fake_folders=fake_folders,
        generator_names=generator_names,
        is_train=is_train,
        max_samples_per_class=max_samples_per_class,
        **kwargs,
    )

    if is_train:
        # Balanced sampling
        labels = [d[1] for d in dataset.data]
        class_counts = [labels.count(0), labels.count(1)]
        weights = [1.0 / class_counts[l] for l in labels]
        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)
        return DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
    else:
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

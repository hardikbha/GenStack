"""
HydraFake Dataset Loader for XGenDet.

Loads face images from HydraFake JSON annotations with binary labels
and forgery type family labels (Real=0, FS=1, EFG=2, FR=3).
"""

import os
import json
from typing import List, Optional, Tuple, Dict
from random import shuffle as random_shuffle

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image, ImageFile

from .augmentations import get_train_transforms, get_eval_transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

# HydraFake forgery type → family label
HYDRAFAKE_FAMILIES = {
    "real": 0,
    "face swapping": 1,
    "entire face generation": 2,
    "face reenactment": 3,
    "face relighting": 3,       # CF split — treat as reenactment-like
    "face restoration": 3,      # CF split (CodeFormer)
    "face editing": 1,          # CF split (StarGANv2) — manipulation
    "face personalization": 2,  # CF split (PuLID, InfiniteYou) — generation
}


class HydraFakeDataset(Dataset):
    """Dataset for HydraFake JSON-based annotations."""

    def __init__(
        self,
        json_path: str,
        data_root: str = "/home/sachin.chaudhary",
        is_train: bool = True,
        crop_size: int = 224,
        max_samples_per_class: Optional[int] = None,
        jpeg_prob: float = 0.5,
        blur_prob: float = 0.5,
    ):
        self.data_root = data_root
        self.is_train = is_train

        if is_train:
            self.transform = get_train_transforms(crop_size, jpeg_prob, blur_prob)
        else:
            self.transform = get_eval_transforms(crop_size)

        # Load JSON annotations
        with open(json_path, "r") as f:
            annotations = json.load(f)

        # Build data list: (path, binary_label, family_label)
        self.data = []
        missing = 0
        for item in annotations:
            rel_path = item["images"][0]
            full_path = os.path.join(data_root, rel_path)

            if not os.path.exists(full_path):
                missing += 1
                continue

            label = item["label"]  # 0=real, 1=fake
            ftype = item.get("type", "real" if label == 0 else "unknown")
            family = HYDRAFAKE_FAMILIES.get(ftype, 2 if label == 1 else 0)

            self.data.append((full_path, label, family))

        if missing > 0:
            print(f"WARNING: {missing} images not found on disk (skipped)")

        # Balance classes if requested
        if max_samples_per_class is not None:
            real = [d for d in self.data if d[1] == 0]
            fake = [d for d in self.data if d[1] == 1]
            if len(real) > max_samples_per_class:
                random_shuffle(real)
                real = real[:max_samples_per_class]
            if len(fake) > max_samples_per_class:
                random_shuffle(fake)
                fake = fake[:max_samples_per_class]
            self.data = real + fake

        random_shuffle(self.data)

        # Generator-level oversampling: duplicate samples whose path matches a pattern
        # e.g. gen_oversample={"FaceForensics++": 3.0} triples FF++ samples
        if hasattr(self, '_gen_oversample') and self._gen_oversample:
            extra = []
            for path, label, family in self.data:
                for pattern, factor in self._gen_oversample.items():
                    if pattern in path:
                        extra.extend([(path, label, family)] * int(factor - 1))
            self.data.extend(extra)
            random_shuffle(self.data)

        # Print stats
        n_real = sum(1 for d in self.data if d[1] == 0)
        n_fake = sum(1 for d in self.data if d[1] == 1)
        families = {}
        for _, _, fam in self.data:
            families[fam] = families.get(fam, 0) + 1
        print(f"HydraFake Dataset: {n_real} real + {n_fake} fake = {len(self.data)} total")
        print(f"  Families: {families}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        path, label, family = self.data[idx]
        try:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
            return img, label, family
        except Exception:
            return self.__getitem__((idx + 1) % len(self.data))


class HydraFakeTestDataset(Dataset):
    """Dataset for HydraFake test splits (per-generator JSON files)."""

    def __init__(
        self,
        json_path: str,
        data_root: str = "/home/sachin.chaudhary",
        image_root: str = "/home/sachin.chaudhary/hydrafake/test",
        crop_size: int = 224,
    ):
        self.transform = get_eval_transforms(crop_size)

        with open(json_path, "r") as f:
            annotations = json.load(f)

        self.data = []
        missing = 0
        for item in annotations:
            rel_path = item["images"][0]
            # Test images: try data_root first, then image_root with basename matching
            full_path = os.path.join(data_root, rel_path)
            if not os.path.exists(full_path):
                # Try image_root directly
                parts = rel_path.split("/")
                # hydrafake/test/ICLight/1_fake/00722.png → ICLight/1_fake/00722.png
                if len(parts) >= 3:
                    sub_path = "/".join(parts[2:])  # skip hydrafake/test/
                    full_path = os.path.join(image_root, sub_path)

            if not os.path.exists(full_path):
                missing += 1
                continue

            label = item["label"]
            self.data.append((full_path, label))

        if missing > 0:
            print(f"WARNING: {missing}/{len(annotations)} test images not found")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.data[idx]
        try:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
            return img, label
        except Exception:
            return self.__getitem__((idx + 1) % len(self.data))


def create_hydrafake_dataloader(
    json_path: str,
    data_root: str = "/home/sachin.chaudhary",
    is_train: bool = True,
    batch_size: int = 64,
    num_workers: int = 4,
    max_samples_per_class: Optional[int] = None,
    **kwargs,
) -> DataLoader:
    """Create dataloader from HydraFake JSON."""
    dataset = HydraFakeDataset(
        json_path=json_path,
        data_root=data_root,
        is_train=is_train,
        max_samples_per_class=max_samples_per_class,
        **kwargs,
    )

    if is_train:
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

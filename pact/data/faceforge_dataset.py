"""
FaceForge Dataset — loads HydraFake JSON annotations and serves 8-region crops
plus a full-resolution image for use with FaceForgeNet.

Design decisions
----------------
Offline region caching:
    Region extraction with MediaPipe FaceMesh is slow (~50-200 ms/image).
    Pre-extracting all regions to disk at `{cache_dir}/{split}/{image_stem}.pkl`
    amortises this cost.  At __getitem__ time we only load the pkl (fast pickle
    deserialise) and apply per-region tensor augmentations on the CPU.

Per-region augmentations:
    Each region has augmentations tailored to the artifacts it is meant to catch:
    - boundary_strip: JPEG compression, Gaussian blur, H.264 block noise, HFlip.
      Mimics real-world compression of face-swap boundaries.
    - eye / ear regions: ColorJitter, HFlip, small rotation.
    - nose / mouth: ColorJitter + HFlip (subtle geometry changes).
    - full_face: standard crop+jitter+flip.
    Augmentations only run when split='train' and augment=True.

Full image for SRM:
    Loaded at native resolution, shorter side resized to 336px to preserve
    high-frequency noise that SRM filters exploit.  NOT augmented (SRM needs
    unmodified noise statistics).

JSON format (HydraFake convention, matching hydrafake_dataset.py):
    List of dicts: {"images": ["rel/path.jpg", ...], "label": 0|1, "type": "..."}
    image_path = data_root / item["images"][0]
    label      = item["label"]  (0=real, 1=fake)

Failure handling:
    - If a pkl cache file is missing at __getitem__ time, the image is extracted
      on the fly and cached for subsequent access.
    - If MediaPipe fails (no face detected), FaceRegionExtractor returns blank
      224×224 images for all regions except full_face — the sample is still
      served; the model falls back to the full_face + SRM path.
    - If the image file is corrupt or unreadable, the next sample (idx+1) is
      returned instead to avoid DataLoader crashes.
"""

import io
import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

try:
    from ..models.region_extractor import FaceRegionExtractor
except ImportError:
    from models.region_extractor import FaceRegionExtractor

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Normalisation constants (ImageNet) ────────────────────────────────────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Full-image shorter-side target for SRM (higher res = better noise signal) ─
_FULL_IMAGE_SHORTER_SIDE = 336


# ── Augmentation helpers ──────────────────────────────────────────────────────

def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    """
    Apply JPEG compression to a PIL Image and return the decompressed result.

    Simulates the lossy quantisation and block DCT artefacts introduced when
    face-swap images are saved / re-encoded at low quality.

    Args:
        img:     Input PIL Image (RGB or RGBA, will be handled gracefully).
        quality: JPEG quality factor in [1, 95].

    Returns:
        PIL Image (same mode as input, re-opened from JPEG bytes).
    """
    buf = io.BytesIO()
    # JPEG does not support alpha; convert to RGB if needed
    save_img = img.convert("RGB") if img.mode != "RGB" else img
    save_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    result = Image.open(buf).copy()   # .copy() detaches from the BytesIO buffer
    return result


def h264_block_noise(
    img: Image.Image,
    block_size: int = 8,
    noise_std: float = 3.0,
) -> Image.Image:
    """
    Simulate H.264 / video codec block quantisation noise.

    H.264 processes frames in 8×8 (or 16×16) macroblocks; re-encoded video
    frames show uniform low-amplitude noise within each block that differs
    between adjacent blocks.  This simulates that pattern by adding independent
    Gaussian noise per block — an approximation sufficient for data augmentation.

    Args:
        img:        Input PIL Image (RGB).
        block_size: Size of each DCT block (default 8, matching H.264 luma).
        noise_std:  Standard deviation of per-block Gaussian noise (default 3.0).

    Returns:
        PIL Image with block noise applied, pixel values clipped to [0, 255].
    """
    arr = np.array(img.convert("RGB"), dtype=np.float32)  # [H, W, 3]
    h, w = arr.shape[:2]

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block_h = min(block_size, h - i)
            block_w = min(block_size, w - j)
            noise = np.random.normal(0.0, noise_std, (block_h, block_w, arr.shape[2]))
            arr[i:i + block_h, j:j + block_w] += noise

    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr)


# ── Per-region augmentation pipelines ─────────────────────────────────────────
# Each returns a torchvision-style callable applied to a PIL Image → Tensor.
# The final two steps (ToTensor + Normalize) are shared across all regions.

_TO_TENSOR_NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


class _BoundaryStripAug:
    """
    Augmentation pipeline for the boundary_strip region.

    Applies in random order:
        1. JPEG compression (quality 30-80)  — simulates re-encoding artefacts
        2. Gaussian blur (σ 0.5-1.5)         — simulates blending / anti-aliasing
        3. H.264 block noise                 — simulates video codec noise
        4. Random horizontal flip (p=0.5)
    """

    def __call__(self, img: Image.Image) -> Tensor:
        # JPEG compression
        quality = int(np.random.uniform(30, 80))
        img = jpeg_compress(img, quality)

        # Gaussian blur (blur_radius maps approximately to σ for PIL)
        sigma = float(np.random.uniform(0.5, 1.5))
        img = transforms.GaussianBlur(kernel_size=3, sigma=sigma)(img)

        # H.264 block noise
        img = h264_block_noise(img, block_size=8, noise_std=3.0)

        # Random horizontal flip
        if np.random.rand() < 0.5:
            img = TF.hflip(img)

        return _TO_TENSOR_NORMALIZE(img)


class _EyeAug:
    """
    Augmentation pipeline for left_eye and right_eye regions.

    Subtle colour jitter and geometry — eye regions are sensitive;
    over-augmenting degrades iris / reflection cues.
    """

    def __init__(self) -> None:
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
        )

    def __call__(self, img: Image.Image) -> Tensor:
        img = self.color_jitter(img)
        if np.random.rand() < 0.5:
            img = TF.hflip(img)
        # Small rotation: ±5 degrees
        angle = float(np.random.uniform(-5.0, 5.0))
        img = TF.rotate(img, angle)
        return _TO_TENSOR_NORMALIZE(img)


class _NoseAug:
    """Augmentation pipeline for the nose region."""

    def __init__(self) -> None:
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
        )

    def __call__(self, img: Image.Image) -> Tensor:
        img = self.color_jitter(img)
        if np.random.rand() < 0.5:
            img = TF.hflip(img)
        return _TO_TENSOR_NORMALIZE(img)


class _MouthAug:
    """Augmentation pipeline for the mouth region."""

    def __init__(self) -> None:
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
        )

    def __call__(self, img: Image.Image) -> Tensor:
        img = self.color_jitter(img)
        if np.random.rand() < 0.5:
            img = TF.hflip(img)
        return _TO_TENSOR_NORMALIZE(img)


class _EarAug:
    """
    Augmentation pipeline for left_ear and right_ear regions.

    Ears are the most diagnostic region for GAN/diffusion fakes.
    Minimal augmentation to preserve subtle anatomical incoherence cues.
    """

    def __init__(self) -> None:
        self.color_jitter = transforms.ColorJitter(brightness=0.2)

    def __call__(self, img: Image.Image) -> Tensor:
        if np.random.rand() < 0.5:
            img = TF.hflip(img)
        img = self.color_jitter(img)
        return _TO_TENSOR_NORMALIZE(img)


class _FullFaceAug:
    """
    Standard augmentation pipeline for the full_face region.

    Provides broad generalisation signal: colour variation, scale variation,
    and horizontal flip.  RandomResizedCrop uses a tight scale (0.9-1.0) so
    the face is not aggressively cropped — we want global texture, not part.
    """

    def __init__(self) -> None:
        self.pipeline = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
            ),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

    def __call__(self, img: Image.Image) -> Tensor:
        img = self.pipeline(img)
        return _TO_TENSOR_NORMALIZE(img)


def _build_region_augs() -> Dict[str, object]:
    """Build and return a dict of per-region augmentation callables."""
    boundary_aug = _BoundaryStripAug()
    eye_aug      = _EyeAug()
    nose_aug     = _NoseAug()
    mouth_aug    = _MouthAug()
    ear_aug      = _EarAug()
    full_face_aug = _FullFaceAug()

    return {
        "left_eye":       eye_aug,
        "right_eye":      eye_aug,
        "nose":           nose_aug,
        "mouth":          mouth_aug,
        "left_ear":       ear_aug,
        "right_ear":      ear_aug,
        "boundary_strip": boundary_aug,
        "full_face":      full_face_aug,
    }


# ── Evaluation transform (no randomness) ──────────────────────────────────────

def _eval_transform(img: Image.Image) -> Tensor:
    """Resize to 224×224 (if not already), ToTensor, Normalize."""
    if img.size != (224, 224):
        img = img.resize((224, 224), Image.LANCZOS)
    return _TO_TENSOR_NORMALIZE(img)


# ── Full-image transform (for SRM, any resolution) ───────────────────────────

def _full_image_transform(img: Image.Image) -> Tensor:
    """
    Resize shorter side to _FULL_IMAGE_SHORTER_SIDE, preserving aspect ratio,
    then ToTensor + Normalize.

    SRM works at the noise level — we want enough resolution to capture the
    distinct noise fingerprint of generators, which is often at the scale of
    individual pixels.  336px gives SRM enough spatial detail without consuming
    excessive GPU memory.
    """
    # Force fixed square shape so DataLoader can stack into [B,3,H,W].
    # Previously preserved aspect ratio → variable-size tensors → stack fails.
    img = img.resize((_FULL_IMAGE_SHORTER_SIDE, _FULL_IMAGE_SHORTER_SIDE), Image.LANCZOS)
    return _TO_TENSOR_NORMALIZE(img)


# ── FaceForge Dataset ─────────────────────────────────────────────────────────

class FaceForgeDataset(Dataset):
    """
    Dataset that loads HydraFake JSON annotations and serves 8-region crops
    plus a full-resolution image for FaceForgeNet.

    Usage::

        # One-time preprocessing (run before training):
        ds = FaceForgeDataset(json_path, data_root, cache_dir, split='train')
        ds.pre_extract_all(num_workers=8)

        # Then use as normal Dataset:
        loader = DataLoader(ds, batch_size=16, num_workers=4)
        for batch in loader:
            crops      = batch['crops']       # dict[str, Tensor [B,3,224,224]]
            full_image = batch['full_image']  # Tensor [B,3,H,W]
            label      = batch['label']       # Tensor [B] float32
    """

    def __init__(
        self,
        json_path: str,
        data_root: str,
        cache_dir: str,
        split: str = "train",
        augment: bool = True,
        num_workers_extract: int = 4,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            json_path:            Path to HydraFake JSON annotation file.
            data_root:            Root directory for resolving relative image paths
                                  (e.g., /home/sachin.chaudhary).
            cache_dir:            Root directory for pickle region caches.
                                  Caches are stored at {cache_dir}/{split}/{stem}.pkl.
            split:                One of 'train', 'val', 'test'.
            augment:              If True and split='train', apply per-region
                                  augmentations.  Always False for val/test.
            num_workers_extract:  Workers for pre_extract_all() parallel extraction.
                                  Not used during normal __getitem__ calls.
            max_samples:          Cap dataset size (useful for quick debug runs).
        """
        self.data_root  = Path(data_root)
        self.cache_dir  = Path(cache_dir) / split
        self.split      = split
        self.augment    = augment and (split == "train")
        self.num_workers_extract = num_workers_extract

        # Build per-region augmentation pipelines (instantiated once per worker)
        self._region_augs: Optional[Dict[str, object]] = None   # lazy init

        # ── Load JSON annotations ─────────────────────────────────────────────
        with open(json_path, "r") as f:
            annotations = json.load(f)

        # Build list of (full_image_path, label)
        self.samples: List[Tuple[Path, int]] = []
        missing = 0

        for item in annotations:
            rel_path  = item["images"][0]
            full_path = self.data_root / rel_path
            label     = int(item["label"])  # 0=real, 1=fake

            if not full_path.exists():
                missing += 1
                continue

            self.samples.append((full_path, label))

        if missing > 0:
            print(
                f"[FaceForgeDataset:{split}] WARNING: {missing} images not found "
                f"on disk — skipped."
            )

        # Apply max_samples cap
        if max_samples is not None and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        n_real = sum(1 for _, lbl in self.samples if lbl == 0)
        n_fake = sum(1 for _, lbl in self.samples if lbl == 1)
        print(
            f"[FaceForgeDataset:{split}] {n_real} real + {n_fake} fake = "
            f"{len(self.samples)} total | augment={self.augment} | "
            f"cache={self.cache_dir}"
        )

    # ── Lazy augmentation init (per DataLoader worker) ─────────────────────────

    def _get_region_augs(self) -> Dict[str, object]:
        """
        Return per-region aug callables, initialising them lazily on first access.

        Initialising in __init__ would cause issues with DataLoader fork: PIL,
        numpy RNG states, and some torchvision ops should be created after fork.
        """
        if self._region_augs is None:
            self._region_augs = _build_region_augs()
        return self._region_augs

    # ── Cache helpers ──────────────────────────────────────────────────────────

    def _cache_path(self, image_path: Path) -> Path:
        """Return the .pkl cache path for a given image path."""
        # Use image_path stem to avoid collision; include parent dir hash for
        # images with the same filename in different subdirectories.
        parent_tag = image_path.parent.name
        stem       = f"{parent_tag}__{image_path.stem}"
        return self.cache_dir / f"{stem}.pkl"

    def _load_from_cache(self, cache_path: Path) -> Optional[Dict[str, Image.Image]]:
        """Load region dict from pickle cache.  Returns None if cache is missing or corrupt."""
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            warnings.warn(
                f"[FaceForgeDataset] Corrupt cache at {cache_path}: {exc}.  "
                f"Will re-extract.",
                stacklevel=2,
            )
            return None

    def _save_to_cache(
        self,
        cache_path: Path,
        regions: Dict[str, Image.Image],
    ) -> None:
        """Save region dict to pickle cache.  Silently skips on I/O error."""
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(regions, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            warnings.warn(
                f"[FaceForgeDataset] Could not save cache at {cache_path}: {exc}",
                stacklevel=2,
            )

    def _extract_and_cache(
        self,
        image_path: Path,
        cache_path: Path,
    ) -> Dict[str, Image.Image]:
        """
        Run FaceRegionExtractor on one image, save the result to cache, and
        return the region dict.  Creates a fresh extractor instance (thread-safe).
        """
        extractor = FaceRegionExtractor(
            max_num_faces=1,
            min_detection_confidence=0.5,
        )
        try:
            regions = extractor.extract(image_path)
        except Exception as exc:
            warnings.warn(
                f"[FaceForgeDataset] Extraction failed for {image_path}: {exc}.  "
                f"Returning blank regions.",
                stacklevel=2,
            )
            # Return blank dict — model still works via full_face token
            try:
                from ..models.region_extractor import _blank_image
            except ImportError:
                from models.region_extractor import _blank_image
            full_pil = Image.open(str(image_path)).convert("RGB")
            full_224 = full_pil.resize((224, 224), Image.LANCZOS)
            region_names = [
                "left_eye", "right_eye", "nose", "mouth",
                "left_ear", "right_ear", "boundary_strip",
            ]
            regions = {name: _blank_image() for name in region_names}
            regions["full_face"] = full_224

        self._save_to_cache(cache_path, regions)
        return regions

    # ── Region → Tensor ───────────────────────────────────────────────────────

    def _regions_to_tensors(
        self,
        regions: Dict[str, Image.Image],
    ) -> Dict[str, Tensor]:
        """
        Apply augmentations (or eval transform) to each region PIL Image and
        return a dict of [3, 224, 224] float32 Tensors.
        """
        tensors: Dict[str, Tensor] = {}

        if self.augment:
            augs = self._get_region_augs()
            for name, pil_img in regions.items():
                try:
                    tensors[name] = augs[name](pil_img)
                except Exception as exc:
                    warnings.warn(
                        f"[FaceForgeDataset] Aug failed for region '{name}': {exc}.  "
                        f"Falling back to eval transform.",
                        stacklevel=2,
                    )
                    tensors[name] = _eval_transform(pil_img)
        else:
            for name, pil_img in regions.items():
                tensors[name] = _eval_transform(pil_img)

        return tensors

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            {
                'crops':      dict[str, Tensor [3,224,224]]  — 8 region crops
                'full_image': Tensor [3,H,W]                  — SRM input
                'label':      float32 scalar tensor            — 0.0 or 1.0
                'image_path': str                              — absolute path
            }
        """
        image_path, label = self.samples[idx]

        # ── Load full image for SRM branch ────────────────────────────────────
        try:
            full_pil = Image.open(str(image_path)).convert("RGB")
        except Exception as exc:
            warnings.warn(
                f"[FaceForgeDataset] Could not open {image_path}: {exc}.  "
                f"Returning next sample.",
                stacklevel=2,
            )
            return self.__getitem__((idx + 1) % len(self.samples))

        full_image_tensor = _full_image_transform(full_pil)  # [3, H', W']

        # ── Load or extract region crops ──────────────────────────────────────
        cache_path = self._cache_path(image_path)
        regions    = self._load_from_cache(cache_path)

        if regions is None:
            # Cache miss: extract on the fly and persist for next access
            regions = self._extract_and_cache(image_path, cache_path)

        # ── Apply augmentations / eval transforms ─────────────────────────────
        crop_tensors = self._regions_to_tensors(regions)

        return {
            "crops":      crop_tensors,
            "full_image": full_image_tensor,
            "label":      torch.tensor(float(label), dtype=torch.float32),
            "image_path": str(image_path),
        }

    # ── Offline pre-extraction ─────────────────────────────────────────────────

    def pre_extract_all(self, num_workers: int = 4) -> None:
        """
        Pre-extract regions for all images in the dataset and save to cache.

        Safe to call multiple times — already-cached images are skipped.
        Run this once before starting training to avoid on-the-fly extraction
        overhead in the DataLoader.

        Args:
            num_workers: Number of parallel worker processes for extraction.
                         Each worker gets its own FaceRegionExtractor instance
                         (MediaPipe is not thread-safe with shared instances).
        """
        # Identify images that are not yet cached
        uncached: List[Tuple[int, Path]] = []
        for i, (image_path, _) in enumerate(self.samples):
            cp = self._cache_path(image_path)
            if not cp.exists():
                uncached.append((i, image_path))

        if not uncached:
            print(
                f"[FaceForgeDataset:{self.split}] All {len(self.samples)} images "
                f"already cached.  Nothing to do."
            )
            return

        print(
            f"[FaceForgeDataset:{self.split}] Pre-extracting {len(uncached)} "
            f"/ {len(self.samples)} uncached images "
            f"with {num_workers} workers ..."
        )

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _worker_task(args: Tuple[int, Path]) -> Tuple[int, bool]:
            """Extract and cache one image.  Returns (idx, success)."""
            i, img_path = args
            cp = self._cache_path(img_path)
            try:
                self._extract_and_cache(img_path, cp)
                return i, True
            except Exception as exc:
                warnings.warn(
                    f"[FaceForgeDataset] pre_extract_all failed for {img_path}: {exc}",
                    stacklevel=2,
                )
                return i, False

        n_ok  = 0
        n_err = 0

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_worker_task, item): item[0]
                for item in uncached
            }
            with tqdm(
                total=len(uncached),
                desc=f"Extracting [{self.split}]",
                unit="img",
                dynamic_ncols=True,
            ) as pbar:
                for future in as_completed(futures):
                    _, success = future.result()
                    if success:
                        n_ok += 1
                    else:
                        n_err += 1
                    pbar.update(1)
                    if n_err > 0:
                        pbar.set_postfix({"errors": n_err})

        print(
            f"[FaceForgeDataset:{self.split}] Pre-extraction complete: "
            f"{n_ok} succeeded, {n_err} failed."
        )

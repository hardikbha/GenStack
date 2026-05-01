"""
MediaPipe-based Face Region Extractor for FaceForge-Net.

Why region-based extraction matters for generic deepfake detection:
- Face swaps (FaceForensics++): the boundary seam is exposed by the 'boundary_strip'
  region — a narrow ring around the face convex hull where blending artifacts live
- Face reenactment (facevid2vid, Hallo2): eye and mouth crops catch unnatural motion
  blur, temporal inconsistency in teeth/iris texture from the driving signal
- Full generation (StyleGAN, Midjourney): ear crops are especially revealing —
  generators consistently fail to produce anatomically coherent ear structure
- Attribute editing (ICLight, FaceAdapter): nose/forehead lighting gradient
  discontinuities are caught by 'nose' and the 'full_face' context token
- Cross-domain (FFIW, DeepFaceLab): compression + boundary artifacts compound;
  'left_ear'/'right_ear' often include the jaw seam where paste operations end

Fallback design:
  If MediaPipe fails to detect a face (e.g., FFIW full-scene images or very
  low-resolution inputs), all regions return blank black 224×224 PIL Images.
  The model can still classify via the 'full_face' token (unmasked resize of the
  original) which carries global texture cues even without landmark guidance.

Region definitions follow the MediaPipe FaceMesh 468-landmark scheme.
All crops are returned as 224×224 PIL Images (RGB) for uniform downstream use.
"""

import io
import math
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image

# Suppress mediapipe C++ log spam on import
import os
os.environ.setdefault("GLOG_minloglevel", "3")

import mediapipe as mp


# ── Landmark index sets ───────────────────────────────────────────────────────
# Each tuple is the *complete* set of indices used to compute the bounding box
# for that region.  Duplicate indices are harmless.

_REGION_LANDMARKS: Dict[str, List[int]] = {
    "left_eye": [
        33, 7, 163, 144, 145, 153, 154, 155,
        133, 173, 157, 158, 159, 160, 161, 246,
    ],
    "right_eye": [
        362, 382, 381, 380, 374, 373, 390, 249,
        263, 466, 388, 387, 386, 385, 384, 398,
    ],
    "nose": [
        1, 2, 5, 4, 6, 168, 195, 197, 19,
        94, 2, 164, 0, 11, 12, 13, 14, 15, 16, 17,
    ],
    "mouth": [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    ],
    "left_ear": [
        234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
    ],
    "right_ear": [
        454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
    ],
}

# Face contour indices for boundary_strip construction
_FACE_CONTOUR_LANDMARKS: List[int] = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# Boundary strip morphological parameters (pixels at 224×224 resolution)
_BOUNDARY_DILATION_PX = 12
_BOUNDARY_EROSION_PX  = 12

# Bounding box margin for standard regions (fraction of box dimension)
_REGION_MARGIN = 0.20

# Output resolution for all crops
_OUTPUT_SIZE = 224


# ── Helpers ───────────────────────────────────────────────────────────────────

def _blank_image() -> Image.Image:
    """Return a black 224×224 RGB PIL Image used as fallback."""
    return Image.fromarray(np.zeros((_OUTPUT_SIZE, _OUTPUT_SIZE, 3), dtype=np.uint8))


def _load_image(source: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
    """
    Accept a file path, PIL Image, or numpy array and return an RGB uint8
    numpy array [H, W, 3].
    """
    if isinstance(source, (str, Path)):
        pil = Image.open(str(source)).convert("RGB")
        return np.array(pil, dtype=np.uint8)
    if isinstance(source, Image.Image):
        return np.array(source.convert("RGB"), dtype=np.uint8)
    if isinstance(source, np.ndarray):
        arr = source
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        elif arr.shape[2] == 3 and arr.dtype == np.uint8:
            pass  # already RGB uint8 — assume caller passed RGB not BGR
        return arr.astype(np.uint8)
    raise TypeError(f"Unsupported image type: {type(source)}")


def _landmarks_to_pixels(
    landmarks, img_h: int, img_w: int, indices: List[int]
) -> np.ndarray:
    """
    Convert a subset of MediaPipe normalised landmarks to absolute pixel coords.

    Returns:
        pts: [N, 2] int32 array of (x, y) coordinates (clipped to image bounds)
    """
    pts = []
    for idx in indices:
        lm  = landmarks[idx]
        px  = int(lm.x * img_w)
        py  = int(lm.y * img_h)
        px  = max(0, min(img_w - 1, px))
        py  = max(0, min(img_h - 1, py))
        pts.append((px, py))
    return np.array(pts, dtype=np.int32)


def _crop_region_with_margin(
    image: np.ndarray,
    pts: np.ndarray,
    margin: float = _REGION_MARGIN,
) -> Image.Image:
    """
    Compute bounding box of pts, add margin, crop, and resize to 224×224.

    Args:
        image:  [H, W, 3] RGB uint8
        pts:    [N, 2] int32 pixel coordinates
        margin: fractional margin added to each side of the bounding box

    Returns:
        224×224 PIL Image (RGB)
    """
    img_h, img_w = image.shape[:2]

    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()

    w = x_max - x_min
    h = y_max - y_min

    # Degenerate box guard: if the region collapses to a point use a minimum size
    if w < 4:
        w = max(4, int(img_w * 0.05))
    if h < 4:
        h = max(4, int(img_h * 0.05))

    pad_x = int(w * margin)
    pad_y = int(h * margin)

    x0 = max(0, x_min - pad_x)
    y0 = max(0, y_min - pad_y)
    x1 = min(img_w, x_max + pad_x)
    y1 = min(img_h, y_max + pad_y)

    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return _blank_image()

    pil = Image.fromarray(crop)
    return pil.resize((_OUTPUT_SIZE, _OUTPUT_SIZE), Image.LANCZOS)


def _extract_boundary_strip(
    image: np.ndarray,
    contour_pts: np.ndarray,
) -> Image.Image:
    """
    Build a boundary strip around the face convex hull using morphological ops.

    Pipeline:
        1. Draw filled convex hull of face contour landmarks → binary mask
        2. Dilate mask by BOUNDARY_DILATION_PX
        3. Erode  mask by BOUNDARY_EROSION_PX
        4. ring_mask = dilated XOR eroded  (the ring between them)
        5. Apply ring mask to original image → zero out non-ring pixels
        6. Crop bounding box of non-zero ring region, resize to 224×224

    Args:
        image:       [H, W, 3] RGB uint8
        contour_pts: [N, 2] int32 face contour landmark coordinates

    Returns:
        224×224 PIL Image (RGB)
    """
    img_h, img_w = image.shape[:2]

    # ── Step 1: filled convex hull mask ──────────────────────────────────────
    hull     = cv2.convexHull(contour_pts)
    face_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(face_mask, [hull], 255)

    # ── Step 2 & 3: dilation and erosion ─────────────────────────────────────
    kernel_d = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (_BOUNDARY_DILATION_PX * 2 + 1, _BOUNDARY_DILATION_PX * 2 + 1),
    )
    kernel_e = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (_BOUNDARY_EROSION_PX * 2 + 1, _BOUNDARY_EROSION_PX * 2 + 1),
    )

    dilated = cv2.dilate(face_mask, kernel_d, iterations=1)
    eroded  = cv2.erode(face_mask,  kernel_e, iterations=1)

    # ── Step 4: ring = dilated minus eroded ──────────────────────────────────
    # pixels in dilated but not in eroded = the outer ring
    ring_mask = cv2.bitwise_xor(dilated, eroded)   # [H, W] uint8

    # Guard: if ring collapses (very small face), fall back to dilated
    if ring_mask.sum() == 0:
        ring_mask = dilated

    # ── Step 5: apply mask ────────────────────────────────────────────────────
    masked = image.copy()
    masked[ring_mask == 0] = 0   # zero out non-ring pixels

    # ── Step 6: crop bounding box of ring, resize ─────────────────────────────
    ys, xs = np.where(ring_mask > 0)
    if len(xs) == 0:
        return _blank_image()

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    # Add a small margin so we don't clip the outermost ring pixels
    pad = max(4, _BOUNDARY_DILATION_PX)
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(img_w, x1 + pad)
    y1 = min(img_h, y1 + pad)

    crop = masked[y0:y1, x0:x1]
    if crop.size == 0:
        return _blank_image()

    pil = Image.fromarray(crop)
    return pil.resize((_OUTPUT_SIZE, _OUTPUT_SIZE), Image.LANCZOS)


# ── Main class ────────────────────────────────────────────────────────────────

class FaceRegionExtractor:
    """
    Extracts 8 facial regions from an image using MediaPipe FaceMesh.

    Regions returned (all 224×224 RGB PIL Images):
        left_eye       — left eye region with 20% margin
        right_eye      — right eye region with 20% margin
        nose           — nose region with 20% margin
        mouth          — mouth region with 20% margin
        left_ear       — left ear / jaw junction region
        right_ear      — right ear / jaw junction region
        boundary_strip — narrow morphological ring along face hull edge
        full_face      — whole image resized to 224×224 (no landmarks needed)

    The MediaPipe FaceMesh model is initialised once at construction and reused
    across all calls — avoid creating multiple instances in tight loops.

    Example::

        extractor = FaceRegionExtractor()
        regions   = extractor.extract("path/to/face.jpg")
        left_eye_pil = regions["left_eye"]   # PIL.Image 224×224

    Batch usage (parallel, offline preprocessing)::

        paths  = ["img1.jpg", "img2.jpg", ...]
        batch  = extractor.extract_batch(paths, num_workers=8)
        # batch[0]["boundary_strip"] → PIL.Image
    """

    def __init__(self, max_num_faces: int = 1, min_detection_confidence: float = 0.5):
        """
        Args:
            max_num_faces:            Maximum number of faces to detect (we use
                                      the first detected face for region extraction).
            min_detection_confidence: MediaPipe detection confidence threshold.
                                      Lower = more detections on hard images,
                                      higher = fewer false positives.
        """
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_num_faces,
            refine_landmarks=True,       # iris landmarks enabled
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def extract(
        self,
        image_path_or_pil: Union[str, Path, Image.Image, np.ndarray],
    ) -> Dict[str, Image.Image]:
        """
        Extract all 8 face regions from a single image.

        Args:
            image_path_or_pil: File path (str / Path), PIL Image, or RGB numpy array.

        Returns:
            dict with keys: 'left_eye', 'right_eye', 'nose', 'mouth',
                            'left_ear', 'right_ear', 'boundary_strip', 'full_face'
            All values are 224×224 PIL Images (RGB).
            If face detection fails, all regions are blank black images
            EXCEPT 'full_face' which is still the resized original image.
        """
        image_np = _load_image(image_path_or_pil)  # [H, W, 3] RGB uint8
        img_h, img_w = image_np.shape[:2]

        # full_face is always available regardless of landmark detection
        full_face = Image.fromarray(image_np).resize(
            (_OUTPUT_SIZE, _OUTPUT_SIZE), Image.LANCZOS
        )

        # ── Run MediaPipe FaceMesh ────────────────────────────────────────────
        # MediaPipe expects BGR but works with RGB if we pass it as RGB consistently;
        # to be safe and follow mp convention we convert here.
        image_bgr    = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        mp_result    = self._face_mesh.process(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # back to RGB for mp
        )

        if not mp_result.multi_face_landmarks:
            # Fallback: no face detected
            fallback = {name: _blank_image() for name in _REGION_LANDMARKS}
            fallback["boundary_strip"] = _blank_image()
            fallback["full_face"]      = full_face
            return fallback

        # Use only the first detected face
        face_landmarks = mp_result.multi_face_landmarks[0].landmark

        # ── Extract standard regions ──────────────────────────────────────────
        regions: Dict[str, Image.Image] = {}

        for region_name, lm_indices in _REGION_LANDMARKS.items():
            pts = _landmarks_to_pixels(face_landmarks, img_h, img_w, lm_indices)
            regions[region_name] = _crop_region_with_margin(image_np, pts)

        # ── Extract boundary strip ────────────────────────────────────────────
        contour_pts = _landmarks_to_pixels(
            face_landmarks, img_h, img_w, _FACE_CONTOUR_LANDMARKS
        )
        regions["boundary_strip"] = _extract_boundary_strip(image_np, contour_pts)

        # ── Attach full_face ──────────────────────────────────────────────────
        regions["full_face"] = full_face

        return regions

    def extract_batch(
        self,
        image_paths: List[Union[str, Path]],
        num_workers: int = 4,
    ) -> List[Dict[str, Image.Image]]:
        """
        Extract regions from a list of image paths in parallel.

        Uses a ThreadPoolExecutor — suitable for I/O-bound preprocessing pipelines.
        Note: MediaPipe is not thread-safe with a shared model instance; each worker
        creates its own FaceRegionExtractor to avoid race conditions.

        Args:
            image_paths: List of file paths (str or Path).
            num_workers: Number of parallel worker threads.

        Returns:
            List of region dicts in the same order as image_paths.
            Each dict matches the schema returned by extract().
        """
        # Pre-allocate result list to preserve ordering
        results: List[Optional[Dict[str, Image.Image]]] = [None] * len(image_paths)

        def _worker(idx_path):
            idx, path = idx_path
            # Each thread gets its own extractor (thread-safe isolation)
            extractor = FaceRegionExtractor(
                max_num_faces=1,
                min_detection_confidence=0.5,
            )
            try:
                return idx, extractor.extract(path)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"[FaceRegionExtractor] Failed on {path}: {exc}",
                    stacklevel=2,
                )
                blank = {name: _blank_image() for name in list(_REGION_LANDMARKS) + ["boundary_strip", "full_face"]}
                return idx, blank

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {
                pool.submit(_worker, (i, p)): i
                for i, p in enumerate(image_paths)
            }
            for future in as_completed(futures):
                idx, region_dict = future.result()
                results[idx] = region_dict

        return results   # type: ignore[return-value]

    def __del__(self):
        """Clean up MediaPipe resources on garbage collection."""
        try:
            self._face_mesh.close()
        except Exception:  # noqa: BLE001
            pass

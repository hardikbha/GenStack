"""
XGenDetV5Plus: Frozen v5 + Three Forensic Branches + Fusion Head.

Architecture overview:

    Input image [B, 3, 224, 224]
        ├── frozen XGenDet (v5)     → binary_logit [B, 1]   (CLIP semantic features)
        ├── BoundarySRMBranch       → [B, 128]               (face boundary noise mismatch)
        ├── PhaseBranch             → [B, 128]               (blending seam phase discontinuity)
        └── BilateralResidualBranch → [B, 128]               (illumination inconsistency)
                ↓
        concat → [B, 385]   (1 + 128 + 128 + 128)
                ↓
        FusionHead: Linear(385,128) → GELU → Dropout(0.1) → Linear(128,32) → GELU → Linear(32,1)
                ↓
        logit [B, 1]  →  sigmoid  →  prob [B, 1]

Design rationale — why fuse over the scalar logit rather than the 1702-d feature vector?
  - No hooks or surgery of the frozen v5 model needed.
  - The scalar logit is a sufficient statistic for v5's entire prediction; it already
    encodes all CLIP-based semantic reasoning.
  - The three branches add orthogonal forensic signals (noise fingerprints, phase
    discontinuities, illumination gradients) that are invisible to CLIP.
  - The fusion head learns when to trust or override the v5 logit for the specific
    failure modes observed in HydraFake (CF split, FFIW boundary artefacts, etc.).
  - Keeping the fused input small (385-d) prevents overfitting and speeds training.

Training workflow:
  - v5 params are permanently frozen; only branches + fusion_head are trained.
  - Use get_trainable_params() to pass two lr groups to the optimiser:
      branches lr=1e-4, head lr=5e-4  (head converges faster with the small input).
  - Loss: BCEWithLogitsLoss on output['logit']  (do NOT apply sigmoid before the loss).
  - For logging, compare logit_v5 vs logit to track when branches override v5.
"""

import os
import logging
from typing import Optional

import torch
import torch.nn as nn

from .xgendet import XGenDet
from .bilateral_residual_branch import BilateralResidualBranch

# These two branches are written by other agents in the same models/ directory.
# They expose the same interface: __init__(output_dim=128), forward(x) → [B, 128].
from .boundary_srm_branch import BoundarySRMBranch
from .phase_branch import PhaseBranch

logger = logging.getLogger(__name__)


# ── Fusion head ───────────────────────────────────────────────────────────────

class FusionHead(nn.Module):
    """
    Lightweight MLP that combines v5 logit + branch features into a final logit.

    Input dimension: 1 (v5 logit) + 128 (SRM) + 128 (Phase) + 128 (Bilateral) = 385.
    Output: [B, 1] raw logit (BCEWithLogitsLoss compatible).
    """

    def __init__(self, input_dim: int = 385, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)                                    # [B, 1]


# ── Main model ────────────────────────────────────────────────────────────────

class XGenDetV5Plus(nn.Module):
    """
    Frozen XGenDet v5 augmented with three forensic branches and a fusion head.

    Args:
        v5_checkpoint_path: Path to best_model.pth for XGenDet v5.  If None the
                            model is initialised with random weights (useful for
                            unit-testing the architecture without the checkpoint).

    Typical usage::

        model = XGenDetV5Plus(
            v5_checkpoint_path="checkpoints/v5_resume_hl/best_model.pth"
        )
        optimizer = torch.optim.AdamW(model.get_trainable_params())
    """

    # Branch output dimension — all three branches produce the same width.
    _BRANCH_DIM = 128

    def __init__(self, v5_checkpoint_path: Optional[str] = None):
        super().__init__()

        # ── Build v5 backbone ────────────────────────────────────────────────
        # Params match config.json in checkpoints/v5_resume_hl/.
        self.v5 = XGenDet(
            clip_model_name="ViT-L/14",
            num_prompt_tokens=8,
            num_prototypes=128,
            proto_dim=128,
            shuffle_patch_size=32,
        )

        if v5_checkpoint_path is not None:
            self._load_v5_checkpoint(v5_checkpoint_path)

        # Freeze every v5 parameter — they must never receive gradients.
        for p in self.v5.parameters():
            p.requires_grad = False
        logger.info("XGenDet v5 frozen — %d parameters",
                    sum(p.numel() for p in self.v5.parameters()))

        # ── Forensic branches (trainable) ────────────────────────────────────
        self.boundary_srm = BoundarySRMBranch(output_dim=self._BRANCH_DIM)
        self.phase = PhaseBranch(output_dim=self._BRANCH_DIM)
        self.bilateral = BilateralResidualBranch(output_dim=self._BRANCH_DIM)

        # ── Fusion head (trainable) ──────────────────────────────────────────
        # Input: logit_v5(1) + srm(128) + phase(128) + bilateral(128) = 385
        fused_dim = 1 + self._BRANCH_DIM * 3                 # 385
        self.fusion_head = FusionHead(input_dim=fused_dim, dropout=0.1)

        trainable = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )
        logger.info("XGenDetV5Plus trainable parameters: %d", trainable)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_v5_checkpoint(self, path: str) -> None:
        """
        Load v5 weights from a checkpoint saved with the full model state_dict.

        Handles two common checkpoint formats:
          1. Raw state_dict (keys match XGenDet attribute names directly).
          2. State_dict stored under a 'model' key, with a 'model.' prefix on
             every parameter name (produced by some training harnesses).
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"v5 checkpoint not found: {path}")

        logger.info("Loading v5 checkpoint from %s", path)
        ckpt = torch.load(path, map_location="cpu")

        # Unwrap checkpoint wrappers that store state_dict under a key.
        state_dict = ckpt
        for key in ("state_dict", "model_state_dict", "model"):
            if isinstance(state_dict, dict) and key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                logger.info("Unwrapped checkpoint key: '%s'", key)
                break

        # Strip 'model.' prefix if present (common in DDP / DeepSpeed saves).
        if any(k.startswith("model.") for k in state_dict):
            state_dict = {k[len("model."):]: v for k, v in state_dict.items()}
            logger.info("Stripped 'model.' prefix from checkpoint keys")

        # Strip 'module.' prefix from DataParallel checkpoints.
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            logger.info("Stripped 'module.' prefix from checkpoint keys")

        # Remap flat prototype_*/heatmap_* → nested prototype_module.*/heatmap_generator.*
        # Earlier versions of XGenDet saved keys flat; the current class uses submodules.
        remapped = {}
        for k, v in state_dict.items():
            nk = k
            if k.startswith('prototype_') and not k.startswith('prototype_module.'):
                nk = 'prototype_module.' + k[len('prototype_'):]
            elif k.startswith('heatmap_') and not k.startswith(('heatmap_generator.', 'heatmap_stats.')):
                rest = k[len('heatmap_'):]
                if rest.startswith('stats.') or rest.startswith('stat_'):
                    nk = 'heatmap_stats.' + (rest.split('.', 1)[1] if '.' in rest else rest)
                else:
                    nk = 'heatmap_generator.' + rest
            remapped[nk] = v
        state_dict = remapped

        missing, unexpected = self.v5.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("v5 checkpoint missing keys (%d): %s ...",
                           len(missing), missing[:5])
        if unexpected:
            logger.warning("v5 checkpoint unexpected keys (%d): %s ...",
                           len(unexpected), unexpected[:5])
        logger.info("v5 checkpoint loaded (strict=False)")

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> dict:
        """
        Full forward pass.

        Args:
            x: [B, 3, 224, 224] — normalised RGB image tensor

        Returns:
            dict with:
                logit     [B, 1]  raw logit — use with BCEWithLogitsLoss
                prob      [B, 1]  sigmoid(logit) — use for inference / metrics
                logit_v5  [B, 1]  v5 baseline logit (log to track branch contribution)
        """
        # ── Frozen v5 inference ──────────────────────────────────────────────
        # torch.no_grad() ensures no activations are stored for v5, saving ~40%
        # of GPU memory during training.
        with torch.no_grad():
            v5_out = self.v5(x, return_heatmap=False)
        logit_v5 = v5_out["binary_logit"]                     # [B, 1]

        # ── Forensic branches (with grad) ────────────────────────────────────
        feat_srm = self.boundary_srm(x)                       # [B, 128]
        feat_phase = self.phase(x)                            # [B, 128]
        feat_bilateral = self.bilateral(x)                    # [B, 128]

        # ── Late fusion ──────────────────────────────────────────────────────
        fused = torch.cat(
            [logit_v5, feat_srm, feat_phase, feat_bilateral], dim=1
        )                                                     # [B, 385]
        logit = self.fusion_head(fused)                       # [B, 1]
        prob = torch.sigmoid(logit)                           # [B, 1]

        return {
            "logit": logit,           # BCEWithLogitsLoss target
            "prob": prob,             # inference / metrics
            "logit_v5": logit_v5,     # v5 baseline — log separately
        }

    # ── Optimiser interface ───────────────────────────────────────────────────

    def get_trainable_params(self) -> list:
        """
        Return two optimiser parameter groups with independent learning rates.

        Group 0 — 'branches' (lr=1e-4):
            BoundarySRMBranch, PhaseBranch, BilateralResidualBranch CNN params.
            Lower lr because the CNNs need careful fine-tuning on forensic signals.

        Group 1 — 'head' (lr=5e-4):
            FusionHead MLP params.
            Higher lr because the head is small and converges quickly from scratch.

        Usage::

            optimizer = torch.optim.AdamW(model.get_trainable_params(),
                                          weight_decay=1e-2)
        """
        branch_params = (
            list(self.boundary_srm.parameters()) +
            list(self.phase.parameters()) +
            list(self.bilateral.parameters())
        )
        head_params = list(self.fusion_head.parameters())

        return [
            {"params": branch_params, "lr": 1e-4, "name": "branches"},
            {"params": head_params,   "lr": 5e-4, "name": "head"},
        ]

    # ── Checkpoint helpers (branches + head only) ────────────────────────────

    def save_branches_checkpoint(self, path: str) -> None:
        """
        Save only the trainable components (branches + fusion head).

        The frozen v5 weights (~1.6 GB) are intentionally omitted — they are
        loaded separately from the original v5 checkpoint at inference time.

        Args:
            path: Destination file path (e.g. 'checkpoints/v5plus/best.pth').
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "boundary_srm": self.boundary_srm.state_dict(),
            "phase": self.phase.state_dict(),
            "bilateral": self.bilateral.state_dict(),
            "fusion_head": self.fusion_head.state_dict(),
        }
        torch.save(payload, path)
        logger.info("Saved branches checkpoint → %s", path)

    def load_branches_checkpoint(self, path: str) -> None:
        """
        Restore only the branch + fusion_head params from a saved checkpoint.

        Does not touch v5 weights.  Safe to call after __init__ regardless of
        whether a v5 checkpoint was provided.

        Args:
            path: Path to a checkpoint produced by save_branches_checkpoint().
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Branches checkpoint not found: {path}")

        payload = torch.load(path, map_location="cpu")
        self.boundary_srm.load_state_dict(payload["boundary_srm"])
        self.phase.load_state_dict(payload["phase"])
        self.bilateral.load_state_dict(payload["bilateral"])
        self.fusion_head.load_state_dict(payload["fusion_head"])
        logger.info("Loaded branches checkpoint from %s", path)

    # ── Convenience ───────────────────────────────────────────────────────────

    def count_trainable_params(self) -> dict:
        """Return parameter counts per component (excludes frozen v5)."""
        return {
            "boundary_srm": sum(p.numel() for p in self.boundary_srm.parameters() if p.requires_grad),
            "phase":        sum(p.numel() for p in self.phase.parameters()         if p.requires_grad),
            "bilateral":    sum(p.numel() for p in self.bilateral.parameters()     if p.requires_grad),
            "fusion_head":  sum(p.numel() for p in self.fusion_head.parameters()   if p.requires_grad),
            "total":        sum(p.numel() for p in self.parameters()               if p.requires_grad),
        }

    @torch.no_grad()
    def detect(self, x: torch.Tensor) -> dict:
        """
        Convenience inference wrapper (eval mode, no grad).

        Args:
            x: [B, 3, 224, 224]

        Returns:
            dict with:
                prediction [B]    — 0 = real, 1 = fake
                prob       [B]    — confidence in fake class
        """
        self.eval()
        out = self.forward(x)
        return {
            "prediction": (out["prob"] > 0.5).long().squeeze(-1),
            "prob":        out["prob"].squeeze(-1),
        }

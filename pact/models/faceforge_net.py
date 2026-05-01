"""
FaceForge-Net: Multi-Region Deepfake Detection Model.

Architecture overview:
    - DINOv2BackboneLoRA (shared, single instance): encodes each of 8 facial region
      crops into a 1024-d CLS token.  Sharing weights enforces representation
      consistency across regions and cuts GPU memory vs 8 separate backbones.
    - SRMBranch: applies 30 fixed noise-residual filters to the full-resolution image
      and learns a 256-d forensic noise embedding — captures compression/blending
      artifacts invisible in RGB space.
    - Cross-Region Attention Fusion: uses the boundary_strip token as query over all
      8 region tokens.  This forces the model to ask "how does the boundary relate to
      every other region?" — the most diagnostic question for face-swap detection.
      Single-head attention keeps gradient flow simple and avoids head-collapse issues.
    - MLP head: fuses 1024-d attended visual features + 256-d SRM noise features into
      a binary confidence score via two-stage dimensionality reduction.

Generic design rationale:
    - full_face token handles generative fakes (StyleGAN, Midjourney): global texture
      statistics that no single region captures.
    - boundary_strip is the primary query because face-swap artifacts always manifest
      at the paste boundary — even after Gaussian blending.
    - ear tokens are irreplaceable for GAN/diffusion fakes: generators consistently
      produce incoherent ear anatomy.
    - SRM noise branch is generator-agnostic: every manipulation leaves a noise
      fingerprint regardless of forgery type.

Input:
    crops:      dict[str, Tensor]  — 8 region crops, each [B, 3, 224, 224]
    full_image: Tensor             — [B, 3, H, W] at native/336px resolution

Output:
    {
        'logit':          [B, 1]   — raw pre-sigmoid logit
        'prob':           [B, 1]   — probability of being fake (sigmoid of logit)
        'region_weights': [B, 8]  — cross-attention weights over the 8 regions
    }
"""

import math
from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .dino_backbone import DINOv2BackboneLoRA
from .srm_branch import SRMBranch


# ── Region ordering ────────────────────────────────────────────────────────────
# This order is fixed across training, inference, and loss computation.
# Index 6 = boundary_strip, which is used as the cross-attention query.

REGION_NAMES: List[str] = [
    "left_eye",       # 0
    "right_eye",      # 1
    "nose",           # 2
    "mouth",          # 3
    "left_ear",       # 4
    "right_ear",      # 5
    "boundary_strip", # 6  ← cross-attention query
    "full_face",      # 7
]

BOUNDARY_IDX: int = 6


# ── Cross-Region Attention ─────────────────────────────────────────────────────

class CrossRegionAttention(nn.Module):
    """
    Single-head cross-attention where the boundary_strip token queries all 8
    region tokens.

    Explicitly avoids nn.MultiheadAttention because the [B, 1, 1024] query
    shape triggers shape assertion issues in some PyTorch versions when
    batch_first=False is assumed.  The manual bmm implementation is 3 lines
    and has zero overhead.

    Inputs:
        tokens: [B, 8, 1024]  — stacked region tokens in REGION_NAMES order

    Outputs:
        attended: [B, 1024]   — weighted sum across all 8 tokens
        weights:  [B, 8]      — softmax attention distribution
    """

    def __init__(self, embed_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = math.sqrt(embed_dim)

        # Layer-norm applied after attending, before downstream fusion
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Dropout on attention weights (regularises which regions the model attends)
        self.attn_dropout = nn.Dropout(p=dropout)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            tokens: [B, 8, 1024]

        Returns:
            attended: [B, 1024]
            weights:  [B, 8]
        """
        B, N, D = tokens.shape  # B, 8, 1024

        # Query = boundary_strip token, unsqueezed to [B, 1, D]
        Q = tokens[:, BOUNDARY_IDX, :].unsqueeze(1)   # [B, 1, 1024]
        K = tokens                                     # [B, 8, 1024]
        V = tokens                                     # [B, 8, 1024]

        # Scaled dot-product: [B, 1, 8]
        attn_logits = torch.bmm(Q, K.transpose(1, 2)) / self.scale   # [B, 1, 8]
        attn_weights = F.softmax(attn_logits, dim=-1)                 # [B, 1, 8]
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum: [B, 1, 1024] → [B, 1024]
        attended = torch.bmm(attn_weights, V).squeeze(1)              # [B, 1024]
        attended = self.layer_norm(attended)

        # Squeeze weights for output: [B, 8]
        weights_out = attn_weights.squeeze(1)                         # [B, 8]

        return attended, weights_out


# ── FaceForge-Net ──────────────────────────────────────────────────────────────

class FaceForgeNet(nn.Module):
    """
    FaceForge-Net deepfake detection model.

    Single model, no ensemble logic.  Works across all forgery types by design:
        - Face swaps   → boundary_strip query + SRM noise
        - Reenactment  → eye / mouth crops + SRM temporal artifacts
        - Full gen     → full_face + ear crops (anatomical incoherence)
        - Attr editing → nose / full_face + SRM lighting gradients
    """

    def __init__(
        self,
        lora_r: int = 16,
        lora_alpha: float = 32.0,
        lora_blocks: int = 8,
        srm_out_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Args:
            lora_r:       LoRA rank (passed through; DINOv2BackboneLoRA uses its own
                          class-level defaults — this param is accepted for API
                          consistency and logged but does not override backbone config).
            lora_alpha:   LoRA scaling alpha (same note as lora_r).
            lora_blocks:  Number of ViT blocks to adapt with LoRA (same note).
            srm_out_dim:  Output dimensionality of the SRM branch (default 256).
            dropout:      Dropout probability used in attention and MLP head.
        """
        super().__init__()

        # Store config for checkpointing / repr
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_blocks = lora_blocks
        self.srm_out_dim = srm_out_dim
        self.dropout_p = dropout

        # ── 1. Shared DINOv2 + LoRA backbone ───────────────────────────────────
        # Single instance — all 8 crops are passed through sequentially.
        # Sharing weights is correct: region crops are all face sub-images from
        # the same distribution; a single backbone learns universally.
        self.backbone = DINOv2BackboneLoRA()
        # Note: DINOv2BackboneLoRA uses fixed class-level LoRA hyper-parameters
        # (r=16, alpha=32, blocks 16-23).  lora_r / lora_alpha / lora_blocks above
        # are stored for reference and future extension.

        backbone_dim = 1024  # ViT-L/14 CLS token size

        # ── 2. SRM noise branch ─────────────────────────────────────────────────
        self.srm_branch = SRMBranch(output_dim=srm_out_dim)

        # ── 3. Cross-region attention ───────────────────────────────────────────
        self.cross_attn = CrossRegionAttention(embed_dim=backbone_dim, dropout=dropout)

        # ── 4. MLP head ─────────────────────────────────────────────────────────
        # Input dim: backbone_dim (1024) + srm_out_dim (256) = 1280
        fusion_dim = backbone_dim + srm_out_dim  # 1280

        self.mlp_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1),
            # No Sigmoid here — raw logit for BCEWithLogitsLoss during training.
            # Sigmoid applied in forward() for the 'prob' output only.
        )

        self._log_param_counts()

    # ── Parameter counting ─────────────────────────────────────────────────────

    def _log_param_counts(self) -> None:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[FaceForgeNet] Total params:     {total:,}\n"
            f"[FaceForgeNet] Trainable params: {trainable:,} "
            f"({100.0 * trainable / max(total, 1):.2f}%)"
        )

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self, crops: Dict[str, Tensor], full_image: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            crops:
                dict with keys matching REGION_NAMES.
                Each value is [B, 3, 224, 224] float32 (ImageNet normalised).
            full_image:
                [B, 3, H, W] float32 (ImageNet normalised).
                Can be any resolution — SRMBranch uses adaptive pooling.

        Returns:
            {
                'logit':          [B, 1]  — raw pre-sigmoid logit
                'prob':           [B, 1]  — sigmoid probability (fake likelihood)
                'region_weights': [B, 8]  — cross-attention weights
            }
        """
        # ── Step 1: Encode each region crop with shared backbone ────────────────
        # Process each region sequentially through the shared backbone.
        # tokens: list of [B, 1024] tensors in REGION_NAMES order.
        tokens: List[Tensor] = []
        for region_name in REGION_NAMES:
            crop = crops[region_name]                      # [B, 3, 224, 224]
            token = self.backbone(crop)                    # [B, 1024]
            tokens.append(token)

        # Stack → [B, 8, 1024]
        token_stack = torch.stack(tokens, dim=1)           # [B, 8, 1024]

        # ── Step 2: SRM noise features from full image ──────────────────────────
        srm_feats = self.srm_branch(full_image)            # [B, 256]

        # ── Step 3: Cross-region attention ──────────────────────────────────────
        # boundary_strip token queries all 8 tokens → weighted fusion
        attended, region_weights = self.cross_attn(token_stack)
        # attended: [B, 1024],  region_weights: [B, 8]

        # ── Step 4: Concatenate attended visual + SRM noise ─────────────────────
        fused = torch.cat([attended, srm_feats], dim=1)    # [B, 1280]

        # ── Step 5: MLP head → logit ─────────────────────────────────────────────
        logit = self.mlp_head(fused)                       # [B, 1]
        prob  = torch.sigmoid(logit)                       # [B, 1]

        return {
            "logit":          logit,
            "prob":           prob,
            "region_weights": region_weights,
        }

    # ── Trainable parameter groups ─────────────────────────────────────────────

    def get_trainable_params(self) -> List[Dict[str, Any]]:
        """
        Returns parameter groups with per-component learning rates for use with
        any PyTorch optimiser.

        Groups:
            backbone_lora  — LoRA A/B matrices in DINOv2 blocks 16-23  lr=1e-5
            srm            — SRM CNN learnable weights                  lr=5e-5
            cross_attn     — attention layer_norm + dropout params      lr=1e-4
            fusion_head    — MLP head linear layers                     lr=1e-4

        Rationale for lr ordering:
            backbone_lora < srm < cross_attn = fusion_head
            The backbone is pretrained and large; small lr prevents catastrophic
            forgetting.  SRM CNN is also somewhat converged after warm-up.
            The fusion components are randomly initialised and learn fastest.
        """
        backbone_lora_params = [
            p for p in self.backbone.parameters() if p.requires_grad
        ]

        srm_params = list(self.srm_branch.parameters())
        # SRM kernels are buffers (no grad); only CNN weights are here.
        srm_params = [p for p in srm_params if p.requires_grad]

        cross_attn_params = [
            p for p in self.cross_attn.parameters() if p.requires_grad
        ]

        mlp_params = [
            p for p in self.mlp_head.parameters() if p.requires_grad
        ]

        groups = []

        if backbone_lora_params:
            groups.append({
                "params":       backbone_lora_params,
                "lr":           1e-5,
                "weight_decay": 1e-2,
                "name":         "backbone_lora",
            })

        if srm_params:
            groups.append({
                "params":       srm_params,
                "lr":           5e-5,
                "weight_decay": 1e-4,
                "name":         "srm",
            })

        if cross_attn_params:
            groups.append({
                "params":       cross_attn_params,
                "lr":           1e-4,
                "weight_decay": 1e-4,
                "name":         "cross_attn",
            })

        if mlp_params:
            groups.append({
                "params":       mlp_params,
                "lr":           1e-4,
                "weight_decay": 1e-4,
                "name":         "fusion_head",
            })

        if not groups:
            raise RuntimeError(
                "[FaceForgeNet] No trainable parameters found.  "
                "Check LoRA injection in DINOv2BackboneLoRA."
            )

        return groups

    # ── Checkpoint utilities ───────────────────────────────────────────────────

    def save_pretrained(self, path: str) -> None:
        """
        Save only the trainable parameters to a .pt checkpoint.

        Saves a dict:
            {
                'state_dict': {name: param_tensor for trainable params only},
                'config':     {lora_r, lora_alpha, lora_blocks, srm_out_dim, dropout},
            }

        Frozen backbone weights are NOT saved — they will be reloaded from the
        pretrained hub checkpoint at model construction time.  This keeps
        checkpoints small (~30 MB vs ~1.2 GB for the full model).

        Args:
            path: File path for the .pt checkpoint.  Parent directory is created
                  if it does not exist.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        trainable_state: Dict[str, Tensor] = {
            name: param.detach().cpu()
            for name, param in self.named_parameters()
            if param.requires_grad
        }

        config = {
            "lora_r":       self.lora_r,
            "lora_alpha":   self.lora_alpha,
            "lora_blocks":  self.lora_blocks,
            "srm_out_dim":  self.srm_out_dim,
            "dropout":      self.dropout_p,
        }

        torch.save({"state_dict": trainable_state, "config": config}, out_path)
        print(
            f"[FaceForgeNet] Saved {len(trainable_state)} trainable tensors "
            f"to {out_path}"
        )

    def load_pretrained_weights(self, path: str) -> None:
        """
        Load trainable parameters from a checkpoint saved by save_pretrained().

        Frozen parameters are left untouched (they were re-loaded from the
        pretrained hub checkpoint during model construction).

        Args:
            path: Path to the .pt checkpoint produced by save_pretrained().

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            KeyError: If the checkpoint format is unexpected.
        """
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"[FaceForgeNet] Checkpoint not found: {ckpt_path}"
            )

        checkpoint = torch.load(ckpt_path, map_location="cpu")

        if "state_dict" not in checkpoint:
            raise KeyError(
                f"[FaceForgeNet] Expected 'state_dict' key in checkpoint, "
                f"got: {list(checkpoint.keys())}"
            )

        saved_state = checkpoint["state_dict"]

        # Load only the keys that exist in saved_state; skip frozen params.
        model_state = self.state_dict()
        missing, unexpected = [], []
        for name, tensor in saved_state.items():
            if name in model_state:
                model_state[name].copy_(tensor)
            else:
                unexpected.append(name)

        for name, param in self.named_parameters():
            if param.requires_grad and name not in saved_state:
                missing.append(name)

        if missing:
            print(f"[FaceForgeNet] WARNING: Missing trainable keys in ckpt: {missing}")
        if unexpected:
            print(f"[FaceForgeNet] WARNING: Unexpected keys in ckpt: {unexpected}")

        self.load_state_dict(model_state, strict=False)
        print(
            f"[FaceForgeNet] Loaded {len(saved_state)} trainable tensors "
            f"from {ckpt_path}"
        )

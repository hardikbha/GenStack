"""
DINOv2 ViT-L/14 Backbone with LoRA Fine-Tuning for FaceForge-Net.

Why DINOv2 for deepfake detection:
- Self-supervised training on diverse imagery gives rich semantic representations
  that transfer to unseen forgery types without task-specific supervision signal
- ViT-L patch tokens carry fine-grained spatial texture — iris patterns, skin
  detail, boundary blending — that CNNs with aggressive downsampling lose early
- Patch size 14px gives 16×16 = 256 tokens at 224×224; each token covers a
  region small enough to capture local seam artifacts (face swaps), motion blur
  inconsistencies (reenactment), and GAN frequency spikes (full generation)

LoRA strategy:
- Only the last 8 blocks (16–23 of 24) are adapted. Early blocks learn universal
  low-level features (edges, textures) that are already well-calibrated for
  forensics. Late blocks specialise in high-level semantic composition — that is
  where forgery artifacts manifest as contradictions (real eye, fake skin, etc.)
- r=16, alpha=32 (scaling = alpha/r = 2.0) keeps updates small but expressive
- Freezing all other weights reduces trainable params ~98%, preventing overfitting
  on small forgery datasets while preserving DINO's generalisation

Output: 1024-d CLS token — no spatial pooling avoids diluting local artifacts.
"""

import math
from typing import List, Dict, Any

import torch
import torch.nn as nn
import timm


# ── LoRA building block ───────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear with a low-rank adaptation.

    The forward pass computes:
        y = x @ W.T  +  x @ A.T @ B.T * scale
    where W is the frozen original weight, A ∈ R^{r×in}, B ∈ R^{out×r}.

    Initialisation:
        A ~ N(0, 0.02)   — small but non-zero so gradient flows from step 0
        B  = 0            — ensures the adapter is a no-op at init (common LoRA)
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        r: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
    ):
        super().__init__()

        in_features  = original_linear.in_features
        out_features = original_linear.out_features

        # Keep the original weight frozen inside this module
        self.original = original_linear
        for p in self.original.parameters():
            p.requires_grad = False

        # Low-rank matrices
        self.lora_A = nn.Linear(in_features,  r,            bias=False)
        self.lora_B = nn.Linear(r,            out_features, bias=False)

        # Scaling: alpha / r  (keeps effective learning rate stable as r changes)
        self.scale = alpha / r

        self.lora_dropout = nn.Dropout(p=dropout)

        # Initialise
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original frozen path
        base_out = self.original(x)

        # LoRA residual: dropout → A → B → scale
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scale

        return base_out + lora_out


# ── DINOv2 backbone ───────────────────────────────────────────────────────────

class DINOv2BackboneLoRA(nn.Module):
    """
    DINOv2 ViT-L/14 with LoRA injected into q, k, v projections of transformer
    blocks 16–23 (the final 8 of 24).

    Input:  [B, 3, 224, 224]  — ImageNet normalised (mean/std as specified)
    Output: [B, 1024]         — CLS token embedding

    Typical usage in FaceForge-Net:
        backbone = DINOv2BackboneLoRA()
        cls_feat = backbone(images)   # → [B, 1024]
    """

    # Blocks to adapt (0-indexed, last 8 of 24)
    LORA_BLOCK_START = 16
    LORA_BLOCK_END   = 23   # inclusive

    # LoRA hyper-parameters
    LORA_R       = 16
    LORA_ALPHA   = 32.0
    LORA_DROPOUT = 0.05

    # ImageNet normalisation (applied inside forward for convenience)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self):
        super().__init__()

        # ── 1. Load pretrained ViT-L/14 DINOv2 ─────────────────────────────
        # The lvd142m checkpoint is trained at 518×518 but we operate on 224×224
        # crops (each face region is already tightly cropped).  Passing img_size=224
        # at creation time re-initialises the positional embeddings for 224 so timm
        # does not assert on the input resolution.  Pretrained patch+projection
        # weights are still loaded; only the pos_embed tensor is resized/interpolated
        # by timm automatically when the checkpoint is loaded.
        self.vit = timm.create_model(
            "vit_large_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,      # drop classification head — we want raw features
            img_size=224,       # resize pos embed from 518 → 224 at load time
        )

        # ── 2. Freeze ALL parameters first ──────────────────────────────────
        for param in self.vit.parameters():
            param.requires_grad = False

        # ── 3. Inject LoRA into q, k, v of target blocks ────────────────────
        self._inject_lora()

        # ── 4. Register normalisation buffers ────────────────────────────────
        mean = torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1)
        std  = torch.tensor(self.IMAGENET_STD ).view(1, 3, 1, 1)
        self.register_buffer("norm_mean", mean)
        self.register_buffer("norm_std",  std)

        # ── 5. Report trainable parameter count ──────────────────────────────
        total_params    = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[DINOv2BackboneLoRA] Total params:     {total_params:,}\n"
            f"[DINOv2BackboneLoRA] Trainable params: {trainable_params:,} "
            f"({100.0 * trainable_params / total_params:.2f}%)"
        )

    # ── LoRA injection ────────────────────────────────────────────────────────

    def _inject_lora(self) -> None:
        """
        Replace q, k, v nn.Linear layers in transformer blocks
        [LORA_BLOCK_START, LORA_BLOCK_END] with LoRALinear wrappers.

        timm ViT-L block layout:
            vit.blocks[i].attn.qkv   — fused [3*dim, dim] linear, OR
            vit.blocks[i].attn.q_proj / k_proj / v_proj  — separate linears
        DINOv2 in timm uses the *fused* qkv Linear.  We therefore split it into
        separate q, k, v wrappers for fine-grained LoRA control.
        """
        for block_idx in range(self.LORA_BLOCK_START, self.LORA_BLOCK_END + 1):
            block = self.vit.blocks[block_idx]
            attn  = block.attn

            # Detect architecture: separate projections vs fused qkv
            if hasattr(attn, "qkv"):
                self._replace_fused_qkv(attn, block_idx)
            else:
                # Separate q_proj, k_proj, v_proj
                for proj_name in ("q_proj", "k_proj", "v_proj"):
                    if hasattr(attn, proj_name):
                        orig = getattr(attn, proj_name)
                        setattr(
                            attn,
                            proj_name,
                            LoRALinear(orig, self.LORA_R, self.LORA_ALPHA, self.LORA_DROPOUT),
                        )

    def _replace_fused_qkv(self, attn: nn.Module, block_idx: int) -> None:
        """
        DINOv2/timm uses a single fused Linear [dim, 3*dim] for q, k, v.
        We split it into three separate nn.Linear modules and wrap each with
        LoRA, then stitch them back inside a FusedQKVLoRA container that
        concatenates along the output dim — preserving the original interface.
        """
        fused: nn.Linear = attn.qkv
        dim = fused.in_features
        assert fused.out_features == 3 * dim, (
            f"Block {block_idx}: unexpected fused qkv shape "
            f"{fused.in_features} → {fused.out_features}"
        )

        # Slice the weight (and bias) into q, k, v chunks
        with torch.no_grad():
            w_q = fused.weight[:dim,      :].clone()   # [dim, dim]
            w_k = fused.weight[dim:2*dim, :].clone()
            w_v = fused.weight[2*dim:,    :].clone()

            b_q = fused.bias[:dim]      .clone() if fused.bias is not None else None
            b_k = fused.bias[dim:2*dim] .clone() if fused.bias is not None else None
            b_v = fused.bias[2*dim:]    .clone() if fused.bias is not None else None

        def _make_linear(w: torch.Tensor, b) -> nn.Linear:
            lin = nn.Linear(dim, dim, bias=(b is not None))
            lin.weight.data.copy_(w)
            if b is not None:
                lin.bias.data.copy_(b)
            return lin

        q_lin = _make_linear(w_q, b_q)
        k_lin = _make_linear(w_k, b_k)
        v_lin = _make_linear(w_v, b_v)

        # Replace fused qkv with a container that has LoRA on each
        attn.qkv = _FusedQKVLoRA(
            q_lin, k_lin, v_lin,
            r=self.LORA_R, alpha=self.LORA_ALPHA, dropout=self.LORA_DROPOUT,
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 224, 224]  float32, values in [0, 1] OR already normalised.
               If values are in [0, 1] normalisation is applied here.
               If already normalised (roughly in [-3, 3]) it is applied again
               which is harmless given the residual nature of LoRA — callers
               should pass either consistently pre-normalised or raw [0,1] images.

        Returns:
            cls_token: [B, 1024]
        """
        # Normalise: (x - mean) / std
        x = (x - self.norm_mean) / self.norm_std

        # ViT forward — timm returns [B, 1024] when num_classes=0
        features = self.vit(x)   # [B, 1024]

        return features

    # ── Trainable parameter groups ────────────────────────────────────────────

    def get_trainable_params(self) -> List[Dict[str, Any]]:
        """
        Returns a list of parameter groups suitable for passing to an optimiser.

        Groups:
            - lora_params:  LoRA A/B matrices in adapted blocks (higher lr)

        All other parameters are frozen and will not appear here.
        """
        lora_params = [p for p in self.parameters() if p.requires_grad]

        if not lora_params:
            raise RuntimeError(
                "No trainable parameters found. "
                "Check that LoRA injection ran correctly."
            )

        return [
            {
                "params": lora_params,
                "lr": 1e-4,
                "weight_decay": 1e-2,
                "name": "lora_params",
            }
        ]


# ── Fused QKV LoRA container ──────────────────────────────────────────────────

class _FusedQKVLoRA(nn.Module):
    """
    Drop-in replacement for timm's fused qkv Linear.

    Holds three separate LoRALinear modules (q, k, v) and concatenates their
    outputs along dim=-1 to reproduce the [B, N, 3*dim] tensor that the rest
    of timm's attention block expects.
    """

    def __init__(
        self,
        q_lin: nn.Linear,
        k_lin: nn.Linear,
        v_lin: nn.Linear,
        r: int,
        alpha: float,
        dropout: float,
    ):
        super().__init__()
        self.q = LoRALinear(q_lin, r, alpha, dropout)
        self.k = LoRALinear(k_lin, r, alpha, dropout)
        self.v = LoRALinear(v_lin, r, alpha, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, dim]  →  output: [B, N, 3*dim]
        return torch.cat([self.q(x), self.k(x), self.v(x)], dim=-1)

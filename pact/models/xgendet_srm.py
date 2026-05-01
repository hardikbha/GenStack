"""
XGenDet + SRM Noise Residual Branch (v15).

Two parallel pathways:
  1. Spatial path:  frozen CLIP ViT-L/14@336px + FPT + PGAD  → spatial_logit
  2. Noise path:    fixed SRM 30-filter bank + learned CNN    → noise_logit

Fusion: concatenate both feature vectors into a joint MLP head.
This allows the model to learn WHEN to trust which pathway
(e.g. noise path dominates for ICLight/face-swaps, spatial for GAN generation).

Why @336px as backbone:
  - Handles 256px generators natively (no downsampling artifact destruction)
  - Larger spatial grid (24×24 vs 16×16) gives richer PGAD prototype attention
"""

import torch
import torch.nn as nn
from .xgendet import XGenDet
from .srm_branch import SRMBranch


class XGenDetSRM(nn.Module):
    def __init__(
        self,
        srm_dim: int = 256,
        clip_model: str = "ViT-L/14@336px",
        **xgendet_kwargs,
    ):
        super().__init__()

        # Spatial path: full XGenDet with @336px backbone
        self.spatial = XGenDet(clip_model_name=clip_model, **xgendet_kwargs)

        # Noise path: SRM residual CNN
        self.noise = SRMBranch(output_dim=srm_dim)

        # Joint fusion head: takes spatial logit + SRM features → final logit
        # The spatial logit is already well-calibrated; SRM corrects it
        self.fusion = nn.Sequential(
            nn.Linear(srm_dim + 1, 128),    # SRM feat (256-d) + spatial logit (1)
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        self.temperature = nn.Parameter(torch.ones(1))
        self.prototype_module = self.spatial.prototype_module

    def forward(self, x: torch.Tensor, return_heatmap: bool = True) -> dict:
        # Path 1: spatial CLIP features
        spatial_out = self.spatial(x, return_heatmap=return_heatmap)
        spatial_logit = spatial_out["binary_logit"]          # [B, 1]

        # Path 2: SRM noise residual
        noise_feat = self.noise(x)                           # [B, srm_dim]

        # Fusion
        fusion_in  = torch.cat([noise_feat, spatial_logit], dim=-1)
        final_logit = self.fusion(fusion_in)                 # [B, 1]
        confidence  = torch.sigmoid(final_logit / self.temperature)

        return {
            **spatial_out,
            "binary_logit":  final_logit,
            "confidence":    confidence,
            "spatial_logit": spatial_logit,
            "noise_features": noise_feat,
        }

    def get_trainable_params(self):
        params = self.spatial.get_trainable_params()
        params.append({
            "params": (list(self.noise.noise_cnn.parameters()) +
                       list(self.fusion.parameters()) +
                       [self.temperature]),
            "lr_scale": 1.0,
            "name": "srm",
        })
        return params

    def count_trainable_params(self) -> dict:
        base = self.spatial.count_trainable_params()
        srm  = sum(p.numel() for p in self.noise.noise_cnn.parameters()
                   if p.requires_grad)
        srm += sum(p.numel() for p in self.fusion.parameters()
                   if p.requires_grad)
        base["srm"]   = srm
        base["total"] += srm
        return base

"""
XGenDet + Frequency Branch (v2).

Architecture: two parallel pathways merged at classification:
  1. Base XGenDet:    frozen CLIP + FPT + PGAD  → base_logit
  2. Spectral branch: FFT log-magnitude CNN      → spectral_logit

Fusion: learned weighted sum via a small MLP on top of both logits.
The blend weight is learned (not fixed), allowing the model to discover
the optimal balance per-sample during training.
"""

import torch
import torch.nn as nn
from .xgendet import XGenDet
from .spectral_branch import SpectralBranch


class XGenDetFreq(nn.Module):
    def __init__(
        self,
        spectral_dim: int = 256,
        num_spectral_filters: int = 3,   # 3 = RGB channels through FFT
        **xgendet_kwargs,
    ):
        super().__init__()
        self.base = XGenDet(**xgendet_kwargs)
        self.spectral_branch = SpectralBranch(
            output_dim=spectral_dim,
            num_filters=num_spectral_filters,
        )

        # Fusion: takes both logits + spectral features → final logit
        # This allows non-linear blending learned during training
        self.fusion_head = nn.Sequential(
            nn.Linear(spectral_dim + 1, 64),   # spectral_feat + base_logit
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

        # Calibrated temperature for final confidence
        self.temperature = nn.Parameter(torch.ones(1))

        # Expose prototype_module for loss computation
        self.prototype_module = self.base.prototype_module

    def forward(self, x: torch.Tensor, return_heatmap: bool = True) -> dict:
        # Base XGenDet path
        base_out = self.base(x, return_heatmap=return_heatmap)
        base_logit = base_out["binary_logit"]                   # [B, 1]

        # Spectral path
        spectral_feat = self.spectral_branch(x)                 # [B, spectral_dim]

        # Learned fusion: concat spectral features with base logit
        fusion_input = torch.cat([spectral_feat, base_logit], dim=-1)  # [B, spectral_dim+1]
        final_logit = self.fusion_head(fusion_input)            # [B, 1]

        # Final confidence with learned temperature
        confidence = torch.sigmoid(final_logit / self.temperature)

        return {
            **base_out,
            "binary_logit": final_logit,
            "confidence": confidence,
            "spectral_features": spectral_feat,
            "base_logit": base_logit,
        }

    def get_trainable_params(self):
        params = self.base.get_trainable_params()
        params.append({
            "params": (list(self.spectral_branch.parameters()) +
                       list(self.fusion_head.parameters()) +
                       [self.temperature]),
            "lr_scale": 1.0,
            "name": "spectral",
        })
        return params

    def count_trainable_params(self) -> dict:
        base = self.base.count_trainable_params()
        spec = (sum(p.numel() for p in self.spectral_branch.parameters() if p.requires_grad) +
                sum(p.numel() for p in self.fusion_head.parameters() if p.requires_grad))
        base["spectral"] = spec
        base["total"] += spec
        return base

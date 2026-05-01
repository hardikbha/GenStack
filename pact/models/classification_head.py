"""
XGenDet Classification Head: Binary (real/fake) + Generator Family + Calibration.

Takes fused features from backbone, prototype module, and heatmap statistics
to produce binary classification, generator family prediction, and calibrated confidence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    def __init__(
        self,
        cls_dim: int = 768,           # CLS token dimension (CLIP output)
        proto_dim: int = 128,          # Prototype feature dimension
        heatmap_stat_dim: int = 32,    # Heatmap statistics dimension
        num_attributes: int = 6,       # Number of attribute scores
        hidden_dim: int = 256,         # Hidden layer dimension
        num_families: int = 4,         # Real / GAN / Diffusion / Autoregressive
        dropout: float = 0.2,
    ):
        super().__init__()
        # Total input: CLS(orig) + CLS(shuf) + proto_weighted + heatmap_stats + attr_scores
        input_dim = cls_dim * 2 + proto_dim + heatmap_stat_dim + num_attributes

        # Binary classification head
        self.binary_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

        # Generator family classification head
        self.family_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_families),
        )

        # Learnable temperature for calibration (initialized to 1.0)
        self.temperature = nn.Parameter(torch.ones(1))

        # Prototype-weighted pooling projection
        self.proto_pool_proj = nn.Linear(proto_dim, proto_dim)

    def prototype_weighted_pool(
        self,
        proto_features: torch.Tensor,
        proto_activations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Weighted pooling of prototype features by activation scores.

        Args:
            proto_features: [B, K, proto_dim]
            proto_activations: [B, K]

        Returns:
            pooled: [B, proto_dim]
        """
        weights = proto_activations.unsqueeze(-1)  # [B, K, 1]
        weighted = (proto_features * weights).sum(dim=1)  # [B, proto_dim]
        weighted = weighted / (proto_activations.sum(dim=-1, keepdim=True) + 1e-8)
        return self.proto_pool_proj(weighted)

    def forward(
        self,
        cls_orig: torch.Tensor,        # [B, cls_dim]
        cls_shuf: torch.Tensor,        # [B, cls_dim]
        proto_features: torch.Tensor,   # [B, K, proto_dim]
        proto_activations: torch.Tensor,# [B, K]
        heatmap_stats: torch.Tensor,    # [B, heatmap_stat_dim]
        attr_scores: torch.Tensor,      # [B, num_attributes]
    ) -> dict:
        """
        Forward pass through classification heads.

        Returns:
            dict with:
                - binary_logit: [B, 1] raw logit for real/fake
                - confidence: [B, 1] calibrated probability
                - family_logit: [B, num_families] generator family logits
        """
        # Prototype-weighted pooling
        proto_pooled = self.prototype_weighted_pool(proto_features, proto_activations)

        # Concatenate all features
        features = torch.cat([
            cls_orig,
            cls_shuf,
            proto_pooled,
            heatmap_stats,
            attr_scores,
        ], dim=-1)

        # Binary classification
        binary_logit = self.binary_head(features)  # [B, 1]

        # Calibrated confidence via temperature scaling
        confidence = torch.sigmoid(binary_logit / self.temperature)  # [B, 1]

        # Generator family classification
        family_logit = self.family_head(features)  # [B, num_families]

        return {
            "binary_logit": binary_logit,
            "confidence": confidence,
            "family_logit": family_logit,
        }

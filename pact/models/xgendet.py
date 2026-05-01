"""
XGenDet: Full Stage 1 Pipeline.

Combines backbone, prototype module, heatmap generator, and classification head
into a single end-to-end model for detection + localization + attribute analysis.
"""

import torch
import torch.nn as nn
from typing import Optional

from .backbone import CLIPBackboneWithPrompts
from .prototype_module import PrototypeModule, NUM_ATTRIBUTES
from .heatmap_generator import HeatmapGenerator, HeatmapStatistics
from .classification_head import ClassificationHead


class XGenDet(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "ViT-L/14",
        num_prompt_tokens: int = 8,
        tune_layer_norm: bool = True,
        num_prototypes: int = 128,
        proto_dim: int = 128,
        proto_heads: int = 4,
        extract_layers: tuple = (6, 12, 18, 23),
        shuffle_patch_size: int = 32,
        heatmap_output_size: int = 224,
        num_families: int = 4,
        dropout: float = 0.2,
        adapter_blocks: list = None,   # e.g. list(range(12, 24))
        adapter_bottleneck: int = 64,
    ):
        super().__init__()
        self.shuffle_patch_size = shuffle_patch_size

        # Backbone
        self.backbone = CLIPBackboneWithPrompts(
            clip_model_name=clip_model_name,
            num_prompt_tokens=num_prompt_tokens,
            tune_layer_norm=tune_layer_norm,
            extract_layers=extract_layers,
            adapter_blocks=adapter_blocks or [],
            adapter_bottleneck=adapter_bottleneck,
        )

        hidden_dim = self.backbone.hidden_dim  # 1024 for ViT-L
        output_dim = self.backbone.output_dim  # 768 for ViT-L
        grid_size = self.backbone.grid_size     # 16 for ViT-L/14

        # Prototype module (PGAD)
        self.prototype_module = PrototypeModule(
            input_dim=hidden_dim,
            proto_dim=proto_dim,
            num_prototypes=num_prototypes,
            num_heads=proto_heads,
            grid_size=grid_size,
        )

        # Heatmap generator
        self.heatmap_generator = HeatmapGenerator(
            extract_layers=extract_layers,
            grid_size=grid_size,
            output_size=heatmap_output_size,
            num_prompt_tokens=num_prompt_tokens,
        )
        self.heatmap_stats = HeatmapStatistics(output_dim=32)

        # Classification head
        self.classification_head = ClassificationHead(
            cls_dim=output_dim,
            proto_dim=proto_dim,
            heatmap_stat_dim=32,
            num_attributes=NUM_ATTRIBUTES,
            num_families=num_families,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_heatmap: bool = True,
    ) -> dict:
        """
        Full forward pass.

        Args:
            x: Input images [B, 3, 224, 224]
            return_heatmap: Whether to compute and return heatmaps

        Returns:
            dict with all Stage 1 outputs
        """
        # Step 1: Backbone encoding (original + shuffled)
        backbone_out = self.backbone(
            x,
            return_shuffled=True,
            shuffle_patch_size=self.shuffle_patch_size,
        )

        # Step 2: Prototype module on original patch tokens
        proto_activations, proto_spatial_maps, attr_scores, proto_features = \
            self.prototype_module(backbone_out["patches_orig"])

        # Step 3: Heatmap generation
        if return_heatmap:
            heatmap = self.heatmap_generator(
                attn_maps_orig=backbone_out.get("attn_maps_orig", {}),
                attn_maps_shuf=backbone_out.get("attn_maps_shuf", {}),
                proto_activations=proto_activations,
                proto_spatial_maps=proto_spatial_maps,
            )
            heatmap_stat_features = self.heatmap_stats(heatmap)
        else:
            heatmap = None
            heatmap_stat_features = torch.zeros(
                x.shape[0], 32, device=x.device
            )

        # Step 4: Classification
        cls_out = self.classification_head(
            cls_orig=backbone_out["cls_orig"],
            cls_shuf=backbone_out.get("cls_shuf", torch.zeros_like(backbone_out["cls_orig"])),
            proto_features=proto_features,
            proto_activations=proto_activations,
            heatmap_stats=heatmap_stat_features,
            attr_scores=attr_scores,
        )

        return {
            "binary_logit": cls_out["binary_logit"],        # [B, 1]
            "confidence": cls_out["confidence"],            # [B, 1]
            "family_logit": cls_out["family_logit"],        # [B, 4]
            "heatmap": heatmap,                             # [B, 1, 224, 224]
            "proto_activations": proto_activations,         # [B, K]
            "proto_spatial_maps": proto_spatial_maps,       # [B, K, 16, 16]
            "attr_scores": attr_scores,                     # [B, 6]
            "proto_features": proto_features,               # [B, K, proto_dim]
            "patches_orig": backbone_out["patches_orig"],   # for loss computation
        }

    def get_trainable_params(self):
        """Get all trainable parameters grouped for optimizer."""
        param_groups = []

        # Backbone params (forgery prompts + layer norms)
        param_groups.extend(self.backbone.get_trainable_params())

        # Prototype module params
        param_groups.append({
            "params": list(self.prototype_module.parameters()),
            "lr_scale": 1.0,
            "name": "prototype_module",
        })

        # Heatmap generator params
        param_groups.append({
            "params": list(self.heatmap_generator.parameters()) +
                      list(self.heatmap_stats.parameters()),
            "lr_scale": 1.0,
            "name": "heatmap",
        })

        # Classification head params
        param_groups.append({
            "params": list(self.classification_head.parameters()),
            "lr_scale": 1.0,
            "name": "classification",
        })

        return param_groups

    def count_trainable_params(self) -> dict:
        """Count trainable parameters per component."""
        counts = {
            "backbone": self.backbone.count_trainable_params(),
            "prototype_module": sum(p.numel() for p in self.prototype_module.parameters() if p.requires_grad),
            "heatmap": sum(p.numel() for p in self.heatmap_generator.parameters() if p.requires_grad) +
                       sum(p.numel() for p in self.heatmap_stats.parameters() if p.requires_grad),
            "classification": sum(p.numel() for p in self.classification_head.parameters() if p.requires_grad),
        }
        counts["total"] = sum(counts.values())
        return counts

    @torch.no_grad()
    def detect(self, x: torch.Tensor) -> dict:
        """Inference-only detection with all outputs."""
        self.eval()
        outputs = self.forward(x, return_heatmap=True)
        return {
            "prediction": (outputs["confidence"] > 0.5).long().squeeze(-1),
            "confidence": outputs["confidence"].squeeze(-1),
            "family": outputs["family_logit"].argmax(dim=-1),
            "heatmap": outputs["heatmap"],
            "attr_scores": outputs["attr_scores"],
            "proto_activations": outputs["proto_activations"],
        }

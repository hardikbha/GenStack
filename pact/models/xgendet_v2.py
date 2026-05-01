"""
XGenDet v2: 9-bank prototype module (192 prototypes).
Everything else same as v1.
"""

import torch
import torch.nn as nn
from .backbone import CLIPBackboneWithPrompts
from .prototype_module_v2 import PrototypeModuleV2, NUM_ATTRIBUTES_V2
from .heatmap_generator import HeatmapGenerator, HeatmapStatistics
from .classification_head import ClassificationHead


class XGenDetV2(nn.Module):
    def __init__(
        self,
        clip_model_name="ViT-L/14",
        num_prompt_tokens=8,
        tune_layer_norm=True,
        num_prototypes=192,
        proto_dim=128,
        proto_heads=4,
        top_k=3,
        extract_layers=(6, 12, 18, 23),
        shuffle_patch_size=32,
        heatmap_output_size=224,
        num_families=4,
        dropout=0.2,
    ):
        super().__init__()
        self.shuffle_patch_size = shuffle_patch_size

        self.backbone = CLIPBackboneWithPrompts(
            clip_model_name=clip_model_name,
            num_prompt_tokens=num_prompt_tokens,
            tune_layer_norm=tune_layer_norm,
            extract_layers=extract_layers,
        )

        hidden_dim = self.backbone.hidden_dim
        output_dim = self.backbone.output_dim
        grid_size = self.backbone.grid_size

        # v2 prototype module: 9 banks, 192 prototypes, top-K scoring
        self.prototype_module = PrototypeModuleV2(
            input_dim=hidden_dim,
            proto_dim=proto_dim,
            num_prototypes=num_prototypes,
            num_heads=proto_heads,
            top_k=top_k,
            grid_size=grid_size,
        )

        self.heatmap_generator = HeatmapGenerator(
            extract_layers=extract_layers,
            grid_size=grid_size,
            output_size=heatmap_output_size,
            num_prompt_tokens=num_prompt_tokens,
        )
        self.heatmap_stats = HeatmapStatistics(output_dim=32)

        # Classification head with 9 attributes instead of 6
        self.classification_head = ClassificationHead(
            cls_dim=output_dim,
            proto_dim=proto_dim,
            heatmap_stat_dim=32,
            num_attributes=NUM_ATTRIBUTES_V2,  # 9 instead of 6
            num_families=num_families,
            dropout=dropout,
        )

    def forward(self, x, return_heatmap=True):
        backbone_out = self.backbone(x, return_shuffled=True, shuffle_patch_size=self.shuffle_patch_size)

        proto_activations, proto_spatial_maps, attr_scores, proto_features = \
            self.prototype_module(backbone_out["patches_orig"])

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
            heatmap_stat_features = torch.zeros(x.shape[0], 32, device=x.device)

        cls_out = self.classification_head(
            cls_orig=backbone_out["cls_orig"],
            cls_shuf=backbone_out.get("cls_shuf", torch.zeros_like(backbone_out["cls_orig"])),
            proto_features=proto_features,
            proto_activations=proto_activations,
            heatmap_stats=heatmap_stat_features,
            attr_scores=attr_scores,
        )

        return {
            "binary_logit": cls_out["binary_logit"],
            "confidence": cls_out["confidence"],
            "family_logit": cls_out["family_logit"],
            "heatmap": heatmap,
            "proto_activations": proto_activations,
            "proto_spatial_maps": proto_spatial_maps,
            "attr_scores": attr_scores,
            "proto_features": proto_features,
            "patches_orig": backbone_out["patches_orig"],
        }

    def get_trainable_params(self):
        param_groups = []
        param_groups.extend(self.backbone.get_trainable_params())
        param_groups.append({"params": list(self.prototype_module.parameters()), "lr_scale": 1.0, "name": "prototype_module"})
        param_groups.append({"params": list(self.heatmap_generator.parameters()) + list(self.heatmap_stats.parameters()), "lr_scale": 1.0, "name": "heatmap"})
        param_groups.append({"params": list(self.classification_head.parameters()), "lr_scale": 1.0, "name": "classification"})
        return param_groups

    def count_trainable_params(self):
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
    def detect(self, x):
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

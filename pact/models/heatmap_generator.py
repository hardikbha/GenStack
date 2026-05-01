"""
XGenDet Heatmap Generator: Multi-source spatial heatmap fusion.

Combines three sources:
1. CLIP attention rollout (CLS attending to patches across layers)
2. Shuffle-difference heatmap (what changed when spatial coherence disrupted)
3. Prototype spatial activation maps (where each prototype fires)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class HeatmapGenerator(nn.Module):
    def __init__(
        self,
        extract_layers: tuple = (6, 12, 18, 23),
        layer_weights: tuple = (0.1, 0.2, 0.3, 0.4),
        grid_size: int = 16,
        output_size: int = 224,
        num_prompt_tokens: int = 8,
    ):
        super().__init__()
        self.extract_layers = extract_layers
        self.layer_weights = layer_weights
        self.grid_size = grid_size
        self.output_size = output_size
        self.num_prompt_tokens = num_prompt_tokens

        # Fusion conv: 3 input channels (clip_attn, diff_attn, proto_attn) -> 1
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

        # Learnable weights for layer combination
        self.layer_weight_params = nn.Parameter(
            torch.tensor(list(layer_weights), dtype=torch.float32)
        )

    def _compute_clip_attention_heatmap(
        self,
        attn_maps: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute attention rollout from CLS token across specified layers.

        Args:
            attn_maps: {layer_idx: [B, heads, seq_len, seq_len]}

        Returns:
            heatmap: [B, 1, grid_size, grid_size]
        """
        weights = F.softmax(self.layer_weight_params, dim=0)
        combined = None

        for i, layer_idx in enumerate(self.extract_layers):
            if layer_idx not in attn_maps:
                continue

            attn = attn_maps[layer_idx]  # [B, heads, seq_len, seq_len]
            B = attn.shape[0]

            # CLS token (index 0) attending to patch tokens
            # Skip prompt tokens: patches start at index 1+num_prompt_tokens
            prompt_offset = 1 + self.num_prompt_tokens
            cls_attn = attn[:, :, 0, prompt_offset:]  # [B, heads, num_patches]
            cls_attn = cls_attn.mean(dim=1)  # [B, num_patches] - average over heads

            # Reshape to spatial grid
            cls_attn = cls_attn[:, :self.grid_size * self.grid_size]
            heatmap_layer = cls_attn.reshape(B, self.grid_size, self.grid_size)

            if combined is None:
                combined = weights[i] * heatmap_layer
            else:
                combined = combined + weights[i] * heatmap_layer

        if combined is None:
            # Fallback: get batch size from any available attention map
            any_attn = next(iter(attn_maps.values()))
            B = any_attn.shape[0]
            dev = any_attn.device
            return torch.zeros(B, 1, self.grid_size, self.grid_size, device=dev)

        return combined.unsqueeze(1)  # [B, 1, grid_size, grid_size]

    def _compute_shuffle_diff_heatmap(
        self,
        attn_maps_orig: Dict[int, torch.Tensor],
        attn_maps_shuf: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute difference between original and shuffled attention patterns.
        Highlights regions where spatial coherence matters most.
        """
        # Use the deepest layer for maximum semantic information
        deepest_layer = max(self.extract_layers)

        if deepest_layer not in attn_maps_orig or deepest_layer not in attn_maps_shuf:
            B = next(iter(attn_maps_orig.values())).shape[0]
            return torch.zeros(B, 1, self.grid_size, self.grid_size,
                             device=next(iter(attn_maps_orig.values())).device)

        attn_orig = attn_maps_orig[deepest_layer]  # [B, heads, seq, seq]
        attn_shuf = attn_maps_shuf[deepest_layer]

        B = attn_orig.shape[0]
        prompt_offset = 1 + self.num_prompt_tokens

        # CLS attention to patches
        cls_orig = attn_orig[:, :, 0, prompt_offset:].mean(dim=1)  # [B, num_patches]
        cls_shuf = attn_shuf[:, :, 0, prompt_offset:].mean(dim=1)

        # Absolute difference
        diff = (cls_orig - cls_shuf).abs()
        diff = diff[:, :self.grid_size * self.grid_size]
        diff = diff.reshape(B, self.grid_size, self.grid_size)

        return diff.unsqueeze(1)  # [B, 1, grid_size, grid_size]

    def _compute_prototype_heatmap(
        self,
        proto_activations: torch.Tensor,
        proto_spatial_maps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Weighted combination of prototype spatial maps.

        Args:
            proto_activations: [B, K] - per-prototype activation scores
            proto_spatial_maps: [B, K, grid_h, grid_w] - per-prototype spatial maps
        """
        # Weight spatial maps by activation scores
        weights = proto_activations.unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
        weighted_maps = proto_spatial_maps * weights  # [B, K, H, W]

        # Sum and normalize
        heatmap = weighted_maps.sum(dim=1, keepdim=True)  # [B, 1, H, W]
        # Normalize per sample
        B = heatmap.shape[0]
        heatmap_flat = heatmap.reshape(B, -1)
        heatmap_min = heatmap_flat.min(dim=-1, keepdim=True).values.reshape(B, 1, 1, 1)
        heatmap_max = heatmap_flat.max(dim=-1, keepdim=True).values.reshape(B, 1, 1, 1)
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)

        return heatmap

    def forward(
        self,
        attn_maps_orig: Dict[int, torch.Tensor],
        attn_maps_shuf: Optional[Dict[int, torch.Tensor]],
        proto_activations: torch.Tensor,
        proto_spatial_maps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate fused heatmap from multiple sources.

        Returns:
            heatmap: [B, 1, output_size, output_size] - pixel-level suspicion map
        """
        # Source 1: CLIP attention rollout
        clip_heatmap = self._compute_clip_attention_heatmap(attn_maps_orig)

        # Source 2: Shuffle difference (if available)
        if attn_maps_shuf is not None:
            diff_heatmap = self._compute_shuffle_diff_heatmap(attn_maps_orig, attn_maps_shuf)
        else:
            diff_heatmap = torch.zeros_like(clip_heatmap)

        # Source 3: Prototype spatial activation
        proto_heatmap = self._compute_prototype_heatmap(proto_activations, proto_spatial_maps)

        # Fuse all three sources
        combined = torch.cat([clip_heatmap, diff_heatmap, proto_heatmap], dim=1)  # [B, 3, grid, grid]
        fused = self.fusion_conv(combined)  # [B, 1, grid, grid]
        fused = torch.sigmoid(fused)

        # Upsample to output size
        heatmap = F.interpolate(
            fused, size=(self.output_size, self.output_size),
            mode="bilinear", align_corners=False
        )

        return heatmap


class HeatmapStatistics(nn.Module):
    """Extract statistical features from heatmap for classification input."""

    def __init__(self, output_dim: int = 32):
        super().__init__()
        self.output_dim = output_dim
        self.fc = nn.Linear(20, output_dim)  # 20 raw statistics -> output_dim

    def forward(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        Args:
            heatmap: [B, 1, H, W]

        Returns:
            stats: [B, output_dim]
        """
        B = heatmap.shape[0]
        h = heatmap.squeeze(1)  # [B, H, W]

        # Global statistics
        global_mean = h.mean(dim=(1, 2))  # [B]
        global_std = h.std(dim=(1, 2))
        global_max = h.reshape(B, -1).max(dim=-1).values
        global_min = h.reshape(B, -1).min(dim=-1).values

        # Quadrant statistics (4 quadrants)
        H, W = h.shape[1], h.shape[2]
        mid_h, mid_w = H // 2, W // 2
        quadrants = [
            h[:, :mid_h, :mid_w],       # top-left
            h[:, :mid_h, mid_w:],        # top-right
            h[:, mid_h:, :mid_w],        # bottom-left
            h[:, mid_h:, mid_w:],        # bottom-right
        ]
        quad_means = torch.stack([q.mean(dim=(1, 2)) for q in quadrants], dim=1)  # [B, 4]
        quad_maxes = torch.stack([q.reshape(B, -1).max(dim=-1).values for q in quadrants], dim=1)

        # Percentiles
        flat = h.reshape(B, -1)
        sorted_vals = flat.sort(dim=-1).values
        n = sorted_vals.shape[1]
        p25 = sorted_vals[:, n // 4]
        p50 = sorted_vals[:, n // 2]
        p75 = sorted_vals[:, 3 * n // 4]
        p90 = sorted_vals[:, int(0.9 * n)]

        # Combine all statistics
        stats = torch.stack([
            global_mean, global_std, global_max, global_min,
            p25, p50, p75, p90,
        ], dim=1)  # [B, 8]
        stats = torch.cat([stats, quad_means, quad_maxes], dim=1)  # [B, 8+4+4=16]

        # Pad to 20 if needed
        if stats.shape[1] < 20:
            padding = torch.zeros(B, 20 - stats.shape[1], device=stats.device)
            stats = torch.cat([stats, padding], dim=1)

        return self.fc(stats)  # [B, output_dim]

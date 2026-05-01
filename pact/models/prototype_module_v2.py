"""
XGenDet v2 Prototype Module — 9 banks (192 prototypes).

Changes from v1:
- 3 new face-specific banks: symmetry, lighting, identity
- Top-K weighted scoring instead of max
- Per-bank learnable temperature
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


ATTRIBUTE_BANKS_V2 = {
    "texture":   (0, 20),     # Surface consistency, pores, smoothing
    "edges":     (20, 40),    # Boundary quality, blending artifacts
    "color":     (40, 60),    # Chromatic aberration, color banding
    "geometry":  (60, 80),    # Structural coherence, perspective errors
    "semantics": (80, 100),   # Content plausibility, impossible features
    "frequency": (100, 120),  # Spectral patterns, GAN grid artifacts
    "symmetry":  (120, 144),  # Left-right facial symmetry, eye alignment
    "lighting":  (144, 168),  # Shadow direction, specular highlights, subsurface scattering
    "identity":  (168, 192),  # Facial landmark coherence, inter-part consistency
}

NUM_ATTRIBUTES_V2 = len(ATTRIBUTE_BANKS_V2)


class PrototypeModuleV2(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        proto_dim: int = 128,
        num_prototypes: int = 192,
        num_heads: int = 4,
        top_k: int = 3,
        grid_size: int = 16,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.proto_dim = proto_dim
        self.num_prototypes = num_prototypes
        self.num_heads = num_heads
        self.top_k = top_k
        self.grid_size = grid_size
        self.head_dim = proto_dim // num_heads
        self.num_banks = len(ATTRIBUTE_BANKS_V2)

        # Learnable prototype bank
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, proto_dim) * 0.02)

        # Project patch tokens from CLIP dim to prototype dim
        self.patch_projection = nn.Sequential(
            nn.Linear(input_dim, proto_dim),
            nn.LayerNorm(proto_dim),
            nn.GELU(),
        )

        # Cross-attention: prototypes as queries, patches as keys/values
        self.q_proj = nn.Linear(proto_dim, proto_dim)
        self.k_proj = nn.Linear(proto_dim, proto_dim)
        self.v_proj = nn.Linear(proto_dim, proto_dim)
        self.out_proj = nn.Linear(proto_dim, proto_dim)

        # LayerNorm
        self.ln_proto = nn.LayerNorm(proto_dim)
        self.ln_patch = nn.LayerNorm(proto_dim)

        # Per-bank learnable temperature (different sensitivity per artifact type)
        self.bank_temperatures = nn.Parameter(torch.ones(self.num_banks) * 0.07)

        # Top-K weights (learnable importance for top-K aggregation)
        self.topk_weights = nn.Parameter(torch.ones(top_k) / top_k)

        self.attribute_banks = ATTRIBUTE_BANKS_V2

    def forward(self, patch_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = patch_tokens.shape

        patches = self.patch_projection(patch_tokens)
        patches = self.ln_patch(patches)

        protos = self.ln_proto(self.prototypes)
        protos = protos.unsqueeze(0).expand(B, -1, -1)

        Q = self.q_proj(protos).reshape(B, self.num_prototypes, self.num_heads, self.head_dim)
        K = self.k_proj(patches).reshape(B, N, self.num_heads, self.head_dim)
        V = self.v_proj(patches).reshape(B, N, self.num_heads, self.head_dim)

        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, self.num_prototypes, self.proto_dim)
        proto_features = self.out_proj(attn_output)

        # Per-prototype activation via cosine similarity with per-bank temperature
        proto_features_norm = F.normalize(proto_features, dim=-1)
        proto_bank_norm = F.normalize(self.prototypes.unsqueeze(0).expand(B, -1, -1), dim=-1)
        raw_sim = (proto_features_norm * proto_bank_norm).sum(dim=-1)  # [B, K]

        # Apply per-bank temperature
        proto_activations = torch.zeros_like(raw_sim)
        for i, (attr_name, (start, end)) in enumerate(self.attribute_banks.items()):
            temp = torch.clamp(self.bank_temperatures[i], min=0.01)
            proto_activations[:, start:end] = torch.sigmoid(raw_sim[:, start:end] / temp)

        # Spatial attention maps
        spatial_attn = attn_weights.mean(dim=1)
        proto_spatial_maps = spatial_attn.reshape(B, self.num_prototypes, self.grid_size, self.grid_size)

        # Attribute scores: weighted top-K instead of max
        attr_scores = self._compute_attribute_scores(proto_activations)

        return proto_activations, proto_spatial_maps, attr_scores, proto_features

    def _compute_attribute_scores(self, proto_activations: torch.Tensor) -> torch.Tensor:
        B = proto_activations.shape[0]
        attr_scores = torch.zeros(B, self.num_banks, device=proto_activations.device)
        weights = F.softmax(self.topk_weights, dim=0)  # Normalize top-K weights

        for i, (attr_name, (start, end)) in enumerate(self.attribute_banks.items()):
            bank_acts = proto_activations[:, start:end]  # [B, bank_size]
            # Top-K weighted mean instead of max
            k = min(self.top_k, bank_acts.shape[1])
            topk_vals, _ = bank_acts.topk(k, dim=-1)  # [B, k]
            attr_scores[:, i] = (topk_vals * weights[:k].unsqueeze(0)).sum(dim=-1)

        return attr_scores

    def prototype_diversity_loss(self) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.prototypes.device)
        count = 0
        for attr_name, (start, end) in self.attribute_banks.items():
            bank_protos = F.normalize(self.prototypes[start:end], dim=-1)
            sim_matrix = torch.matmul(bank_protos, bank_protos.T)
            mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool, device=sim_matrix.device)
            loss = loss + sim_matrix[mask].mean()
            count += 1
        return loss / count

    def prototype_compactness_loss(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        patches = self.patch_projection(patch_tokens)
        cls_features = patches.mean(dim=1)
        cls_norm = F.normalize(cls_features, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        similarities = torch.matmul(cls_norm, proto_norm.T)
        max_sim = similarities.max(dim=-1).values
        return (1.0 - max_sim).mean()

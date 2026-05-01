"""
XGenDet Prototype-Grounded Attention Decomposition (PGAD) Module.

Key design:
- 128 learnable prototypes organized into 6 attribute banks
- Each prototype acts as a query that cross-attends to patch tokens
- Produces both activation scores AND spatial attention maps
- Attribute score = max activation within each bank
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


# Attribute bank definitions
ATTRIBUTE_BANKS = {
    "texture": (0, 21),       # Texture artifacts (blur, noise, plastic appearance)
    "edges": (21, 43),        # Edge artifacts (boundary inconsistency, halos)
    "color": (43, 64),        # Color artifacts (chromatic aberration, oversaturation)
    "geometry": (64, 86),     # Geometric artifacts (perspective errors, deformation)
    "semantics": (86, 107),   # Semantic artifacts (impossible content, wrong shadows)
    "frequency": (107, 128),  # Frequency artifacts (spectral signatures, grid patterns)
}

NUM_ATTRIBUTES = len(ATTRIBUTE_BANKS)


class PrototypeModule(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,      # CLIP ViT-L internal dim
        proto_dim: int = 128,        # Prototype embedding dimension
        num_prototypes: int = 128,   # Total prototypes across all banks
        num_heads: int = 4,          # Cross-attention heads
        temperature: float = 0.07,   # Cosine similarity scaling
        grid_size: int = 16,         # Spatial grid size (16x16 for ViT-L/14)
    ):
        super().__init__()
        self.input_dim = input_dim
        self.proto_dim = proto_dim
        self.num_prototypes = num_prototypes
        self.num_heads = num_heads
        self.temperature = temperature
        self.grid_size = grid_size
        self.head_dim = proto_dim // num_heads

        # Learnable prototype bank
        self.prototypes = nn.Parameter(
            torch.randn(num_prototypes, proto_dim) * 0.02
        )

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

        # LayerNorm for cross-attention
        self.ln_proto = nn.LayerNorm(proto_dim)
        self.ln_patch = nn.LayerNorm(proto_dim)

        # Attribute bank boundaries
        self.attribute_banks = ATTRIBUTE_BANKS

    def forward(
        self, patch_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            patch_tokens: [B, num_patches, input_dim] from CLIP backbone

        Returns:
            proto_activations: [B, num_prototypes] - per-prototype activation scores
            proto_spatial_maps: [B, num_prototypes, grid_h, grid_w] - spatial attention per prototype
            attr_scores: [B, num_attributes] - per-attribute scores (max within bank)
            proto_features: [B, num_prototypes, proto_dim] - prototype-enriched features
        """
        B, N, _ = patch_tokens.shape  # N = num_patches

        # Project patches to prototype space
        patches = self.patch_projection(patch_tokens)  # [B, N, proto_dim]
        patches = self.ln_patch(patches)

        # Prepare prototypes as queries
        protos = self.ln_proto(self.prototypes)  # [num_prototypes, proto_dim]
        protos = protos.unsqueeze(0).expand(B, -1, -1)  # [B, K, proto_dim]

        # Multi-head cross-attention
        Q = self.q_proj(protos).reshape(B, self.num_prototypes, self.num_heads, self.head_dim)
        K = self.k_proj(patches).reshape(B, N, self.num_heads, self.head_dim)
        V = self.v_proj(patches).reshape(B, N, self.num_heads, self.head_dim)

        # [B, heads, K, head_dim] x [B, heads, head_dim, N] -> [B, heads, K, N]
        Q = Q.permute(0, 2, 1, 3)  # [B, heads, K, head_dim]
        K = K.permute(0, 2, 1, 3)  # [B, heads, N, head_dim]
        V = V.permute(0, 2, 1, 3)  # [B, heads, N, head_dim]

        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, heads, K, N]
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Aggregate values
        attn_output = torch.matmul(attn_weights, V)  # [B, heads, K, head_dim]
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, self.num_prototypes, self.proto_dim)
        proto_features = self.out_proj(attn_output)  # [B, K, proto_dim]

        # Compute prototype activation scores via cosine similarity
        proto_features_norm = F.normalize(proto_features, dim=-1)
        proto_bank_norm = F.normalize(self.prototypes.unsqueeze(0).expand(B, -1, -1), dim=-1)
        proto_activations = (proto_features_norm * proto_bank_norm).sum(dim=-1) / self.temperature
        proto_activations = torch.sigmoid(proto_activations)  # [B, K]

        # Compute spatial attention maps per prototype
        # Average attention weights across heads: [B, K, N]
        spatial_attn = attn_weights.mean(dim=1)  # [B, K, N]
        proto_spatial_maps = spatial_attn.reshape(B, self.num_prototypes, self.grid_size, self.grid_size)

        # Compute attribute scores (max activation within each bank)
        attr_scores = self._compute_attribute_scores(proto_activations)  # [B, 6]

        return proto_activations, proto_spatial_maps, attr_scores, proto_features

    def _compute_attribute_scores(self, proto_activations: torch.Tensor) -> torch.Tensor:
        """Compute per-attribute scores as max activation within each bank."""
        B = proto_activations.shape[0]
        attr_scores = torch.zeros(B, NUM_ATTRIBUTES, device=proto_activations.device)

        for i, (attr_name, (start, end)) in enumerate(self.attribute_banks.items()):
            bank_activations = proto_activations[:, start:end]  # [B, bank_size]
            attr_scores[:, i] = bank_activations.max(dim=-1).values

        return attr_scores

    def prototype_diversity_loss(self) -> torch.Tensor:
        """
        Encourage prototype diversity within each attribute bank.
        Minimizes cosine similarity between prototypes in the same bank.
        """
        loss = torch.tensor(0.0, device=self.prototypes.device)
        count = 0

        for attr_name, (start, end) in self.attribute_banks.items():
            bank_protos = F.normalize(self.prototypes[start:end], dim=-1)  # [bank_size, proto_dim]
            sim_matrix = torch.matmul(bank_protos, bank_protos.T)  # [bank_size, bank_size]

            # Exclude diagonal
            mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool, device=sim_matrix.device)
            loss = loss + sim_matrix[mask].mean()
            count += 1

        return loss / count

    def prototype_compactness_loss(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Ensure at least one prototype is close to each training sample.
        Minimizes the distance from each sample to its nearest prototype.
        """
        patches = self.patch_projection(patch_tokens)  # [B, N, proto_dim]
        cls_features = patches.mean(dim=1)  # [B, proto_dim] - global average

        cls_norm = F.normalize(cls_features, dim=-1)  # [B, proto_dim]
        proto_norm = F.normalize(self.prototypes, dim=-1)  # [K, proto_dim]

        similarities = torch.matmul(cls_norm, proto_norm.T)  # [B, K]
        max_sim = similarities.max(dim=-1).values  # [B]

        return (1.0 - max_sim).mean()

    def get_top_prototypes(self, proto_activations: torch.Tensor, top_k: int = 5) -> Dict:
        """Get the top-K activated prototypes with their attribute labels."""
        values, indices = proto_activations.topk(top_k, dim=-1)

        results = []
        for b in range(proto_activations.shape[0]):
            sample_results = []
            for k in range(top_k):
                idx = indices[b, k].item()
                val = values[b, k].item()
                # Find which attribute bank this prototype belongs to
                attr_name = "unknown"
                for name, (start, end) in self.attribute_banks.items():
                    if start <= idx < end:
                        attr_name = name
                        break
                sample_results.append({
                    "prototype_id": idx,
                    "activation": val,
                    "attribute": attr_name,
                })
            results.append(sample_results)

        return results

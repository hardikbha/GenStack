"""
ViT Adapter Blocks for XGenDet v10.

Lightweight bottleneck adapters inserted AFTER each of the last N CLIP
transformer blocks. Each adapter is a residual module:

    x → LayerNorm → Linear(1024 → bottleneck_dim) → GELU
      → Linear(bottleneck_dim → 1024) → scale × out → x + out

Key properties:
  - Fully residual: initialized to near-zero output (scale=0.01)
    so training starts from the same point as the frozen model
  - No catastrophic forgetting: adapters add a learned delta on top of
    frozen representations; the frozen path is always preserved
  - Tiny: 64-dim bottleneck × 2 layers × 24 blocks = ~3.2M params total
    (use last 12 blocks = ~1.6M; last 6 = ~800K)
"""

import torch
import torch.nn as nn


class BottleneckAdapter(nn.Module):
    """
    Residual bottleneck adapter for a single transformer block output.
    Applied after the block's output (post-LN position).
    """
    def __init__(self, hidden_dim: int = 1024, bottleneck_dim: int = 64):
        super().__init__()
        self.ln    = nn.LayerNorm(hidden_dim)
        self.down  = nn.Linear(hidden_dim, bottleneck_dim)
        self.act   = nn.GELU()
        self.up    = nn.Linear(bottleneck_dim, hidden_dim)
        self.scale = nn.Parameter(torch.ones(1) * 0.01)

        # Init: near-zero output so adapter starts as identity
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq_len, batch, hidden_dim]  (CLIP's native layout)
        h = self.ln(x)
        h = self.up(self.act(self.down(h)))
        return x + self.scale * h


class CLIPViTWithAdapters(nn.Module):
    """
    Wraps a CLIP visual transformer, inserting BottleneckAdapters after
    the specified transformer blocks.

    Usage:
        wrapped = CLIPViTWithAdapters(clip_visual, adapter_blocks=list(range(12, 24)))
        # Then call wrapped.forward(x) instead of clip_visual.transformer(x)
    """
    def __init__(
        self,
        transformer,
        adapter_blocks: list,
        hidden_dim: int = 1024,
        bottleneck_dim: int = 64,
    ):
        super().__init__()
        self.transformer = transformer
        self.adapter_blocks = set(adapter_blocks)
        self.hidden_dim = hidden_dim

        # One adapter per specified block
        self.adapters = nn.ModuleDict({
            str(i): BottleneckAdapter(hidden_dim, bottleneck_dim)
            for i in adapter_blocks
        })

    def forward(self, x: torch.Tensor):
        """
        Mirrors CLIP transformer forward: iterates resblocks, applies adapters.
        Returns the final hidden state tensor (same as original transformer).
        """
        for i, block in enumerate(self.transformer.resblocks):
            x = block(x)
            if i in self.adapter_blocks:
                x = self.adapters[str(i)](x)
        return x

    def count_adapter_params(self) -> int:
        return sum(p.numel() for p in self.adapters.parameters())

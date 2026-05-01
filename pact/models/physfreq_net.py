"""
PhysFreqNet: Tri-branch physics + frequency deepfake detector.

Branches (all configurable — turn on/off for ablations):
    - 'fft'     : SpectralBranch          (FFT magnitude -> CNN -> 256d)
    - 'retinex' : RetinexBranch           (Retinex I = R x L decomposition -> 256d)
    - 'ca'      : ChromaticAberrationBranch (Chromatic Aberration map -> 128d)

Fusion: concat -> Linear(D_total, 256) -> GELU -> Dropout -> Linear(256, 64)
        -> GELU -> Dropout -> Linear(64, 1)

Output: dict with 'logit' [B,1] and 'prob' [B,1].
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PhysFreqNet(nn.Module):
    """
    Tri-branch physics + frequency deepfake detector.

    Args:
        branches : tuple of branch names to enable. Subset of ('fft', 'retinex', 'ca').
                   At least one branch is required.
        dropout  : dropout probability in the fusion head.
    """

    def __init__(self, branches=('fft', 'retinex', 'ca'), dropout: float = 0.1):
        super().__init__()
        assert len(branches) > 0, "Need at least one branch"
        self.branch_names = tuple(branches)
        self.branches = nn.ModuleDict()
        d_total = 0

        if 'fft' in branches:
            from .spectral_branch import SpectralBranch
            self.branches['fft'] = SpectralBranch(output_dim=256)
            d_total += 256
        if 'retinex' in branches:
            from .retinex_branch import RetinexBranch
            self.branches['retinex'] = RetinexBranch(output_dim=256)
            d_total += 256
        if 'ca' in branches:
            from .chromatic_aberration_branch import ChromaticAberrationBranch
            self.branches['ca'] = ChromaticAberrationBranch(output_dim=128)
            d_total += 128

        self.fusion_head = nn.Sequential(
            nn.Linear(d_total, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.d_total = d_total
        self._log_params()

    def _log_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[PhysFreqNet] branches={self.branch_names}  "
            f"total={total:,}  trainable={trainable:,}  fusion_in={self.d_total}"
        )

    def forward(self, x):
        """
        Args:
            x : [B, 3, 224, 224] image tensor.

        Returns:
            dict with:
              'logit': [B, 1] raw logits
              'prob' : [B, 1] sigmoid(logit)
        """
        feats = []
        for name in self.branch_names:
            feats.append(self.branches[name](x))
        fused = torch.cat(feats, dim=1)   # [B, D_total]
        logit = self.fusion_head(fused)   # [B, 1]
        return {'logit': logit, 'prob': torch.sigmoid(logit)}

    def get_trainable_params(self):
        """
        Returns list of param-group dicts pre-formatted for AdamW.

        Branches get lr=1e-4, fusion head gets lr=5e-4 (warmer because it's
        shallow and randomly initialized). weight_decay=1e-4 on all groups
        (optimizer-level wd will override if set via optimizer kwargs).
        """
        groups = []
        for name in self.branch_names:
            groups.append({
                'params': list(self.branches[name].parameters()),
                'lr': 1e-4,
                'weight_decay': 1e-4,
                'name': f'branch_{name}',
            })
        groups.append({
            'params': list(self.fusion_head.parameters()),
            'lr': 5e-4,
            'weight_decay': 1e-4,
            'name': 'fusion_head',
        })
        return groups

    def save_checkpoint(self, path):
        torch.save({
            'state_dict': self.state_dict(),
            'branches': self.branch_names,
        }, path)
        print(f"[PhysFreqNet] Saved to {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.load_state_dict(ckpt['state_dict'], strict=False)
        print(f"[PhysFreqNet] Loaded from {path}, branches={ckpt.get('branches', '?')}")


if __name__ == "__main__":
    # Smoke test — prints param counts and runs a random forward pass for each
    # ablation configuration.
    import itertools

    all_branches = ('fft', 'retinex', 'ca')
    configs = []
    for r in range(1, len(all_branches) + 1):
        for combo in itertools.combinations(all_branches, r):
            configs.append(combo)

    x = torch.randn(2, 3, 224, 224)
    for cfg in configs:
        try:
            net = PhysFreqNet(branches=cfg)
            with torch.no_grad():
                out = net(x)
            print(f"  cfg={cfg}  logit={tuple(out['logit'].shape)}  prob={tuple(out['prob'].shape)}")
        except Exception as e:
            print(f"  cfg={cfg}  ERROR: {e}")

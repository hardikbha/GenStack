"""
XGenDet Loss Functions for Stage 1 training.

Combined loss:
L_total = L_cls + w_family * L_family + w_proto * L_proto + w_heatmap * L_heatmap
        + w_attr * L_attr + w_calib * L_calib
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class XGenDetLoss(nn.Module):
    def __init__(
        self,
        w_family: float = 0.5,
        w_proto_div: float = 0.3,
        w_proto_compact: float = 0.2,
        w_heatmap: float = 0.3,
        w_attr: float = 0.1,
        w_calib: float = 0.2,
        attr_margin: float = 0.3,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.w_family = w_family
        self.w_proto_div = w_proto_div
        self.w_proto_compact = w_proto_compact
        self.w_heatmap = w_heatmap
        self.w_attr = w_attr
        self.w_calib = w_calib
        self.attr_margin = attr_margin
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def classification_loss(self, binary_logit: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy loss for real/fake classification."""
        return self.bce_loss(binary_logit.squeeze(-1), labels.float())

    def family_loss(self, family_logit: torch.Tensor, family_labels: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss for generator family classification."""
        return self.ce_loss(family_logit, family_labels)

    def heatmap_loss(self, heatmap: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Self-supervised heatmap loss.
        - Fake images: heatmap should be high everywhere (entire image is AI-generated)
        - Real images: heatmap should be low everywhere
        """
        if heatmap is None:
            return torch.tensor(0.0, device=labels.device)

        B = heatmap.shape[0]
        heatmap_flat = heatmap.reshape(B, -1)  # [B, H*W]

        # Create target: 1 for fake (high everywhere), 0 for real (low everywhere)
        target = labels.float().unsqueeze(-1).expand_as(heatmap_flat)

        return F.binary_cross_entropy(heatmap_flat, target)

    def attribute_loss(self, attr_scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Attribute regularization loss.
        - Real images: all attribute scores should be low
        - Fake images: at least one attribute should be high
        """
        margin = self.attr_margin

        # Real images: penalize high attribute scores
        real_mask = (labels == 0).float()
        real_loss = (F.relu(attr_scores - margin) ** 2).mean(dim=-1)
        real_loss = (real_loss * real_mask).sum() / (real_mask.sum() + 1e-8)

        # Fake images: penalize if max attribute score is too low
        fake_mask = (labels == 1).float()
        max_attr = attr_scores.max(dim=-1).values
        fake_loss = (F.relu(margin - max_attr) ** 2)
        fake_loss = (fake_loss * fake_mask).sum() / (fake_mask.sum() + 1e-8)

        return real_loss + fake_loss

    def focal_calibration_loss(
        self,
        confidence: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Focal loss variant for better calibration."""
        p_t = confidence.squeeze(-1)
        targets = labels.float()

        # p_t for correct class
        p_correct = p_t * targets + (1 - p_t) * (1 - targets)

        focal_weight = (1 - p_correct) ** self.focal_gamma
        loss = -self.focal_alpha * focal_weight * torch.log(p_correct + 1e-8)

        return loss.mean()

    def forward(
        self,
        outputs: dict,
        labels: torch.Tensor,
        family_labels: torch.Tensor,
        prototype_module=None,
    ) -> dict:
        """
        Compute total loss.

        Args:
            outputs: dict from XGenDet.forward()
            labels: [B] binary labels (0=real, 1=fake)
            family_labels: [B] generator family labels (0=Real, 1=GAN, 2=Diffusion, 3=AR)
            prototype_module: PrototypeModule instance for diversity/compactness losses

        Returns:
            dict with individual and total losses
        """
        losses = {}

        # 1. Binary classification loss
        losses["cls"] = self.classification_loss(outputs["binary_logit"], labels)

        # 2. Generator family loss
        losses["family"] = self.family_loss(outputs["family_logit"], family_labels)

        # 3. Prototype losses
        if prototype_module is not None:
            losses["proto_div"] = prototype_module.prototype_diversity_loss()
            losses["proto_compact"] = prototype_module.prototype_compactness_loss(
                outputs["patches_orig"]
            )
        else:
            losses["proto_div"] = torch.tensor(0.0, device=labels.device)
            losses["proto_compact"] = torch.tensor(0.0, device=labels.device)

        # 4. Heatmap loss
        losses["heatmap"] = self.heatmap_loss(outputs["heatmap"], labels)

        # 5. Attribute loss
        losses["attr"] = self.attribute_loss(outputs["attr_scores"], labels)

        # 6. Calibration loss
        losses["calib"] = self.focal_calibration_loss(outputs["confidence"], labels)

        # Total loss
        losses["total"] = (
            losses["cls"]
            + self.w_family * losses["family"]
            + self.w_proto_div * losses["proto_div"]
            + self.w_proto_compact * losses["proto_compact"]
            + self.w_heatmap * losses["heatmap"]
            + self.w_attr * losses["attr"]
            + self.w_calib * losses["calib"]
        )

        return losses

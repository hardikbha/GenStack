"""
XGenDet Confidence Calibration via Temperature Scaling.

Post-hoc calibration of the model's confidence scores using
a held-out calibration set.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader


class TemperatureScaling:
    """Post-hoc temperature scaling for confidence calibration."""

    def __init__(self):
        self.temperature = 1.0

    def fit(self, model, val_loader, device, max_iter=100, lr=0.01):
        """
        Find optimal temperature on validation data.

        Args:
            model: XGenDet model
            val_loader: validation DataLoader
            device: torch device
        """
        model.eval()

        # Collect all logits and labels
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels, families in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs, return_heatmap=False)
                all_logits.append(outputs["binary_logit"].cpu())
                all_labels.append(labels)

        logits = torch.cat(all_logits, dim=0).squeeze(-1)  # [N]
        labels = torch.cat(all_labels, dim=0).float()       # [N]

        # Optimize temperature
        temperature = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
        criterion = nn.BCEWithLogitsLoss()

        def closure():
            optimizer.zero_grad()
            loss = criterion(logits / temperature, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        self.temperature = temperature.item()
        print(f"Optimal temperature: {self.temperature:.4f}")
        return self.temperature

    def calibrate(self, logits):
        """Apply temperature scaling to logits."""
        return torch.sigmoid(logits / self.temperature)


def compute_ece(confidences, accuracies, n_bins=15):
    """
    Compute Expected Calibration Error.

    Args:
        confidences: numpy array of confidence scores
        accuracies: numpy array of binary correctness (0 or 1)
        n_bins: number of calibration bins

    Returns:
        ece: Expected Calibration Error
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue

        bin_confidence = confidences[mask].mean()
        bin_accuracy = accuracies[mask].mean()
        bin_weight = mask.sum() / len(confidences)

        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return ece

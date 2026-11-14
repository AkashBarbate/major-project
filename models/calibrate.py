"""
Calibration script. Needs val_logits.npy and val_labels.npy
saved during training.
"""

import numpy as np
import torch
import torch.nn as nn
import json
import os

SAVE_PATH = 'models/improved_model'

CATEGORIES = [
    'Politics', 'Science and Technology', 'Business and Economy',
    'International Affairs', 'Education', 'Health', 'Sports', 'Entertainment'
]


def calibrate():
    # Load val logits and labels saved during training
    val_logits = np.load(os.path.join(SAVE_PATH, 'val_logits.npy'))
    val_labels = np.load(os.path.join(SAVE_PATH, 'val_labels.npy'))

    logits_tensor = torch.tensor(val_logits, dtype=torch.float32)
    labels_tensor = torch.tensor(val_labels, dtype=torch.float32)

    # Temperature parameter
    temperature = nn.Parameter(torch.tensor([1.5]))
    optimizer   = torch.optim.LBFGS([temperature], lr=0.01, max_iter=200)
    criterion   = nn.BCEWithLogitsLoss()

    def eval_step():
        optimizer.zero_grad()
        scaled_logits = logits_tensor / temperature
        loss = criterion(scaled_logits, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(eval_step)

    T = float(temperature.item())
    print(f"Optimal temperature: {T:.4f}")

    # Evaluate calibration quality
    # Before calibration
    probs_uncal = torch.sigmoid(logits_tensor).detach().numpy()
    # After calibration
    probs_cal   = torch.sigmoid(logits_tensor / temperature).detach().numpy()

    # ECE (Expected Calibration Error)
    def compute_ece(probs, labels, n_bins=10):
        bins   = np.linspace(0, 1, n_bins + 1)
        ece    = 0.0
        n      = probs.shape[0] * probs.shape[1]
        probs_flat  = probs.flatten()
        labels_flat = labels.flatten()
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (probs_flat >= lo) & (probs_flat < hi)
            if mask.sum() == 0:
                continue
            acc  = labels_flat[mask].mean()
            conf = probs_flat[mask].mean()
            ece += mask.sum() / n * abs(acc - conf)
        return ece

    ece_before = compute_ece(probs_uncal, val_labels)
    ece_after  = compute_ece(probs_cal,   val_labels)

    print(f"ECE before calibration: {ece_before:.4f}")
    print(f"ECE after calibration:  {ece_after:.4f}")
    print(f"Improvement: {(ece_before - ece_after)/ece_before*100:.1f}%")

    # Save
    out = {
        'temperature':   T,
        'ece_before':    float(ece_before),
        'ece_after':     float(ece_after),
    }
    out_path = os.path.join(SAVE_PATH, 'temperature.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved -> {out_path}")
    print(f"Use temperature={T:.4f} in predict.py and app.py")
    return T


if __name__ == '__main__':
    calibrate()
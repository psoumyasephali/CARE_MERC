"""Render the GitHub social-preview card (1280x640) used when the repo link is
shared. Upload the output at Settings -> General -> Social preview.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "docs" / "figures" / "social-preview.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

NAVY = "#0A2A66"
NAVY_2 = "#0A3E91"
WHITE = "#FFFFFF"
MUTED = "#AdC2E6"
GREEN = "#7BE0A0"
CHIP = "#163a7a"


def chip(ax, x, y, label):
    w = 0.20
    ax.add_patch(FancyBboxPatch((x, y), w, 0.085,
                 boxstyle="round,pad=0.006,rounding_size=0.04",
                 linewidth=0, facecolor=CHIP, transform=ax.transAxes))
    ax.text(x + w / 2, y + 0.043, label, transform=ax.transAxes,
            ha="center", va="center", color=WHITE, fontsize=15, weight="bold")


fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=NAVY, zorder=0))
# accent bar
ax.add_patch(plt.Rectangle((0, 0), 0.018, 1, facecolor=GREEN, zorder=1))

ax.text(0.07, 0.80, "CARE-MERC", color=WHITE, fontsize=62, weight="bold", va="center")
ax.text(0.072, 0.665, "Multimodal Emotion Recognition on MELD",
        color=MUTED, fontsize=27, va="center")

# Key result (well-spaced so big numbers and labels never collide)
ax.text(0.072, 0.50, "0.639", color=GREEN, fontsize=42, weight="bold", va="center")
ax.text(0.072 + 0.150, 0.50, "weighted-F1", color=WHITE, fontsize=20, va="center")
ax.text(0.072 + 0.370, 0.50, "0.649", color=GREEN, fontsize=42, weight="bold", va="center")
ax.text(0.072 + 0.520, 0.50, "accuracy", color=WHITE, fontsize=20, va="center")
ax.text(0.072 + 0.700, 0.50, "3-seed mean", color=MUTED, fontsize=17, va="center")

# One-line finding
ax.text(0.072, 0.35,
        "A fine-tuned text encoder does the work; frozen audio and visual add nothing.",
        color=WHITE, fontsize=21, va="center")

# Modality chips
chip(ax, 0.072, 0.18, "Text")
chip(ax, 0.072 + 0.23, 0.18, "Audio")
chip(ax, 0.072 + 0.46, 0.18, "Visual")

# Footer
ax.text(0.072, 0.075,
        "Soumya Sephali Pradhan   |   PyTorch  -  RoBERTa / WavLM / ViT  -  trained on an M1 Pro laptop",
        color=MUTED, fontsize=16, va="center")

fig.savefig(OUT, dpi=100)
plt.close(fig)
print(f"saved {OUT}")

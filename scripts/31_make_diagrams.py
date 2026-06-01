"""Render the architecture and pipeline diagrams used in the README.

Pure matplotlib (no graphviz dependency). Writes PNG + SVG to docs/figures/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGDIR = REPO_ROOT / "docs" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

INK = "#0A3E91"
FILL = "#EAF1FC"
ACCENT = "#2E7D32"
ACCENT_FILL = "#EAF3EB"
GREY = "#5b6470"


def box(ax, x, y, w, h, text, fill=FILL, edge=INK, fontsize=11, weight="normal"):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.012,rounding_size=0.02",
                       linewidth=1.4, edgecolor=edge, facecolor=fill)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color="#10202e", weight=weight, zorder=5)


def arrow(ax, p0, p1, color=GREY):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.3, color=color, shrinkA=2, shrinkB=2))


def architecture():
    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis("off")
    ax.set_title("CARE-MERC architecture", fontsize=14, color=INK, pad=8)

    # Inputs
    box(ax, 2.0, 10.2, 2.4, 0.95, "Text\nRoBERTa-large (1024)")
    box(ax, 5.0, 10.2, 2.4, 0.95, "Audio\nWavLM-large (1024)")
    box(ax, 8.0, 10.2, 2.4, 0.95, "Visual\nViT face crop (768)")

    # Concat + speaker, projection
    box(ax, 5.0, 8.4, 7.6, 0.9,
        "Concatenate with speaker embedding  +  input projection\n"
        "(Linear 512, LayerNorm, ReLU, dropout)", fontsize=10.5)
    for sx in (2.0, 5.0, 8.0):
        arrow(ax, (sx, 9.72), (sx, 8.85))

    # Side input: speaker id feeds the concat
    box(ax, 5.0, 9.45, 2.2, 0.55, "Speaker id  ->  embedding", fill="#fbf5e6",
        edge="#b8902a", fontsize=9.5)

    # Two parallel branches
    box(ax, 2.7, 6.4, 3.6, 1.0,
        "Contextual memory\n(+ position embedding, BiGRU -> 256)", fontsize=10)
    box(ax, 7.3, 6.4, 3.6, 1.0,
        "Modality gate\n(softmax over T/A/V -> weighted sum, 256)", fontsize=10)
    arrow(ax, (4.2, 7.95), (2.9, 6.95))
    arrow(ax, (5.8, 7.95), (7.1, 6.95))
    box(ax, 9.1, 7.45, 1.5, 0.5, "Position", fill="#fbf5e6", edge="#b8902a", fontsize=9.5)

    # Sum
    box(ax, 5.0, 4.7, 3.0, 0.8, "Sum the two branches", fontsize=10.5)
    arrow(ax, (2.7, 5.9), (4.2, 5.1))
    arrow(ax, (7.3, 5.9), (5.8, 5.1))

    # Emotion-transition add
    box(ax, 5.0, 3.2, 4.6, 0.8,
        "+ 0.5 x emotion-transition (previous emotion)", fontsize=10)
    arrow(ax, (5.0, 4.3), (5.0, 3.6))
    box(ax, 8.6, 3.2, 1.7, 0.5, "Prev. emotion", fill="#fbf5e6", edge="#b8902a", fontsize=9.5)

    # Head
    box(ax, 5.0, 1.6, 3.4, 0.85, "Emotion head -> 7 logits",
        fill=ACCENT_FILL, edge=ACCENT, fontsize=11, weight="bold")
    arrow(ax, (5.0, 2.8), (5.0, 2.03), color=ACCENT)

    fig.text(0.5, 0.015,
             "Audio and visual encoders are frozen; the text encoder is fine-tuned end to end.",
             ha="center", fontsize=9, color=GREY)
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    out = FIGDIR / "architecture.png"
    plt.savefig(out, dpi=170)
    print(f"saved {out}")
    plt.close(fig)


def pipeline():
    fig, ax = plt.subplots(figsize=(12.5, 3.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Data and training pipeline", fontsize=13, color=INK, pad=6)

    stages = [
        (1.2, 2.0, "00-01\nAudit +\npreprocess"),
        (3.4, 2.0, "02-04\nExtract text,\naudio, visual"),
        (5.6, 2.0, "05\nConsolidate\nfeatures"),
        (7.9, 2.0, "22\nFine-tune\ntext encoder"),
        (10.2, 2.0, "24\nTrain CARE-MERC\n(3 seeds)"),
        (12.6, 2.0, "30\nFigures +\nmetrics"),
    ]
    w, h = 1.9, 1.3
    for i, (x, y, t) in enumerate(stages):
        last = i == len(stages) - 1
        box(ax, x, y, w, h, t,
            fill=ACCENT_FILL if last else FILL,
            edge=ACCENT if last else INK, fontsize=9.5)
        if i > 0:
            arrow(ax, (stages[i - 1][0] + w / 2, y), (x - w / 2, y))

    fig.text(0.5, 0.04,
             "Feature extraction needs the MELD download; training uses only the extracted features.",
             ha="center", fontsize=9, color=GREY)
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    out = FIGDIR / "pipeline.png"
    plt.savefig(out, dpi=170)
    print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    pipeline()

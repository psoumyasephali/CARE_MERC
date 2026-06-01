"""Generate the result figures from the committed per-seed JSON files.

Reads results/*.json only (no training, no model loading). Produces four PNGs
under docs/figures/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
FIGDIR = REPO_ROOT / "docs" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# Best MELD weighted-F1 we are aware of as of 2025 (Sync-TVA, arXiv:2507.21395).
PUBLISHED_BEST_WF1 = 0.674


def load_run(name: str) -> dict:
    with open(RESULTS / f"{name}.json") as fh:
        return json.load(fh)


def aggregate_3seed(prefix: str) -> dict:
    runs = [load_run(f"{prefix}_s{s}") for s in (42, 43, 44)]
    out = {}
    for k in ("wf1", "mf1", "acc"):
        v = [r["test"][k] for r in runs]
        out[k] = (float(np.mean(v)), float(np.std(v, ddof=1)))
    per = {}
    for c in EMOTIONS:
        v = [r["test"]["per_class_f1"][c] for r in runs]
        per[c] = (float(np.mean(v)), float(np.std(v, ddof=1)))
    out["per_class"] = per
    return out


# --- Confusion matrix (final model, seed 42) ----------------------------------

def fig_confusion_matrix():
    run = load_run("caremerc_ftlarge_s42")
    cm = np.array(run["test"]["confusion_matrix"], dtype=np.int64)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.maximum(row_sums, 1)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(7))
    ax.set_yticks(range(7))
    ax.set_xticklabels(EMOTIONS, rotation=35, ha="right")
    ax.set_yticklabels(EMOTIONS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix: fine-tuned RoBERTa-large, MELD test (seed 42)")
    for i in range(7):
        for j in range(7):
            count = int(cm[i, j])
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{count}", ha="center", va="center", color=color, fontsize=10)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Row-normalized frequency")
    plt.tight_layout()
    out = FIGDIR / "confusion_matrix.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


# --- Per-class F1 grouped bars ------------------------------------------------

def fig_per_class():
    frozen = aggregate_3seed("caremerc_frozen")
    ftbase = aggregate_3seed("caremerc_ftbase")
    ftlarge = aggregate_3seed("caremerc_ftlarge")

    def col(agg):
        return ([agg["per_class"][c][0] for c in EMOTIONS],
                [agg["per_class"][c][1] for c in EMOTIONS])

    m_fr, s_fr = col(frozen)
    m_fb, s_fb = col(ftbase)
    m_fl, s_fl = col(ftlarge)

    x = np.arange(len(EMOTIONS))
    width = 0.27

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width, m_fr, width, yerr=s_fr, label="Frozen RoBERTa-large",
           color="#A6C8FF", edgecolor="black", linewidth=0.5, capsize=3)
    ax.bar(x, m_fb, width, yerr=s_fb, label="Fine-tuned RoBERTa-base",
           color="#3D7DE0", edgecolor="black", linewidth=0.5, capsize=3)
    ax.bar(x + width, m_fl, width, yerr=s_fl, label="Fine-tuned RoBERTa-large (final)",
           color="#0A3E91", edgecolor="black", linewidth=0.5, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(EMOTIONS)
    ax.set_ylabel("Per-class F1 (test, 3-seed mean and std)")
    ax.set_ylim(0, 0.9)
    ax.set_title("Per-class F1 by text encoder")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", frameon=False)
    plt.tight_layout()
    out = FIGDIR / "per_class_f1.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


# --- Weighted-F1 by text encoder ----------------------------------------------

def fig_trajectory():
    frozen = aggregate_3seed("caremerc_frozen")
    ftbase = aggregate_3seed("caremerc_ftbase")
    ftlarge = aggregate_3seed("caremerc_ftlarge")
    text_only_large = load_run("roberta_large_finetune")["test"]["wf1"]

    labels = ["Frozen\nRoBERTa-large", "Fine-tuned\nRoBERTa-base", "Fine-tuned\nRoBERTa-large"]
    means = [frozen["wf1"][0], ftbase["wf1"][0], ftlarge["wf1"][0]]
    stds = [frozen["wf1"][1], ftbase["wf1"][1], ftlarge["wf1"][1]]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(x, means, yerr=stds, color="#0A3E91", linewidth=2.0, marker="o",
                markersize=9, markerfacecolor="white", markeredgecolor="#0A3E91",
                markeredgewidth=2.0, capsize=4)
    for xi, m in zip(x, means):
        ax.annotate(f"{m:.3f}", (xi, m), textcoords="offset points",
                    xytext=(0, 14), ha="center", fontsize=10)

    ax.axhline(text_only_large, linestyle=":", color="gray", linewidth=1.2,
               label=f"Text-only fine-tuned RoBERTa-large ({text_only_large:.3f})")
    ax.axhline(PUBLISHED_BEST_WF1, linestyle="--", color="green", linewidth=1.2,
               label=f"Best published MELD result ({PUBLISHED_BEST_WF1:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test weighted-F1 (3-seed mean and std)")
    ax.set_ylim(0.45, 0.70)
    ax.set_title("Fine-tuning the text encoder, not fusion, drives the gain")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    plt.tight_layout()
    out = FIGDIR / "wf1_by_encoder.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


# --- Text-only vs multimodal --------------------------------------------------

def fig_text_only_comparison():
    ft_base = load_run("roberta_finetune")["test"]
    ft_large = load_run("roberta_large_finetune")["test"]
    ftbase = aggregate_3seed("caremerc_ftbase")
    ftlarge = aggregate_3seed("caremerc_ftlarge")

    metrics = ["wF1", "mF1"]
    base_text = [ft_base["wf1"], ft_base["mf1"]]
    base_mm = [ftbase["wf1"][0], ftbase["mf1"][0]]
    base_mm_std = [ftbase["wf1"][1], ftbase["mf1"][1]]
    large_text = [ft_large["wf1"], ft_large["mf1"]]
    large_mm = [ftlarge["wf1"][0], ftlarge["mf1"][0]]
    large_mm_std = [ftlarge["wf1"][1], ftlarge["mf1"][1]]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(2)
    width = 0.18
    centers = {
        "base_text": x - 1.5 * width,
        "base_mm":   x - 0.5 * width,
        "large_text": x + 0.5 * width,
        "large_mm":   x + 1.5 * width,
    }
    series = {
        "base_text": (base_text, None, "RoBERTa-base: text only", "#A6C8FF"),
        "base_mm": (base_mm, base_mm_std, "RoBERTa-base: with multimodal stack (3-seed)", "#3D7DE0"),
        "large_text": (large_text, None, "RoBERTa-large: text only", "#A2C7A4"),
        "large_mm": (large_mm, large_mm_std, "RoBERTa-large: with multimodal stack (3-seed)", "#2E7D32"),
    }
    for key, (vals, errs, label, color) in series.items():
        ax.bar(centers[key], vals, width, yerr=errs, label=label,
               color=color, edgecolor="black", linewidth=0.5,
               capsize=(3 if errs else 0))
        for px, v in zip(centers[key], vals):
            ax.annotate(f"{v:.3f}", (px, v), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Metric value (test)")
    ax.set_ylim(0, 0.75)
    ax.set_title("Text only vs. multimodal: fusion adds no measurable gain")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    plt.tight_layout()
    out = FIGDIR / "text_only_vs_multimodal.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    fig_confusion_matrix()
    fig_per_class()
    fig_trajectory()
    fig_text_only_comparison()


if __name__ == "__main__":
    main()

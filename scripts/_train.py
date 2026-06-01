"""Shared training loop for CARE-MERC.

The three numbered training entry points (13 frozen features, 23 fine-tuned
RoBERTa-base, 24 fine-tuned RoBERTa-large) differ only in three defaults
(tag, npz file, text dim), so the logic lives here and they are thin wrappers
around build_argparser() + run().
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    confusion_matrix,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.care_merc import CareMerc  # noqa: E402
from utils.dataset import CareMercDataset, collate  # noqa: E402


EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def worker_init(worker_id: int) -> None:
    # No-op at the default num_workers=0; takes effect only with worker processes.
    base = torch.initial_seed() % 2**32
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def class_weights_inverse_sqrt(labels: np.ndarray, num_classes: int = 7) -> torch.Tensor:
    """Normalized inverse-sqrt class weights, giving roughly
    [0.82, 1.65, 1.66, 0.65, 0.40, 1.04, 0.78] (mean = 1) on MELD.

    Formula: w_i = K * (1/sqrt(n_i)) / sum_j(1/sqrt(n_j)). The unnormalized
    form `(N / (K*n)) ** 0.5` has the same shape but ~1.39x the magnitude,
    which over-weights the minority classes; the normalized form is used here.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    inv_sqrt = 1.0 / np.sqrt(counts)
    w = num_classes * inv_sqrt / inv_sqrt.sum()
    return torch.tensor(w, dtype=torch.float32)


def build_scheduler(optimizer: torch.optim.Optimizer, warmup_epochs: int, max_epochs: int):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        # cosine 1 -> 0 over the remaining epochs
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            out = model(
                text=batch["text"],
                audio=batch["audio"],
                visual=batch["visual"],
                speaker_id=batch["speaker_id"],
                position=batch["position"],
                prev_emotion=batch["prev_emotion"],
            )
            all_logits.append(out["emotion_logits"].cpu())
            all_labels.append(batch["emotion_label"].cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = logits.argmax(axis=1)
    return {
        "acc": float(accuracy_score(labels, preds)),
        "wf1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "mf1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "preds": preds,
        "labels": labels,
        "logits": logits,
    }


def build_argparser(tag: str, npz_name: str, text_dim: int) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--tag", type=str, default=tag)
    ap.add_argument("--audio_drop", type=float, default=0.15)
    ap.add_argument("--visual_drop", type=float, default=0.15)
    ap.add_argument("--npz_name", type=str, default=npz_name)
    ap.add_argument("--text_dim", type=int, default=text_dim)
    ap.add_argument("--features_dir", type=str, default=str(REPO_ROOT / "features"))
    ap.add_argument("--ckpt_dir", type=str, default=str(REPO_ROOT / "checkpoints"))
    ap.add_argument("--results_dir", type=str, default=str(REPO_ROOT / "results"))
    ap.add_argument("--num_workers", type=int, default=0)  # MPS-safe default
    return ap


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = pick_device()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    run_name = f"caremerc_{args.tag}_s{args.seed}"
    ckpt_path = Path(args.ckpt_dir) / f"{run_name}.pt"
    results_path = Path(args.results_dir) / f"{run_name}.json"
    log_path = Path(args.results_dir) / f"{run_name}_log.txt"

    log_fh = open(log_path, "w")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_fh.write(msg + "\n")
        log_fh.flush()

    log(f"[run] {run_name}")
    log(f"[run] device={device} torch={torch.__version__}")
    log(f"[run] args={vars(args)}")

    # Datasets / loaders
    train_ds = CareMercDataset(args.features_dir, "train", npz_name=args.npz_name)
    dev_ds = CareMercDataset(args.features_dir, "dev", npz_name=args.npz_name)
    test_ds = CareMercDataset(args.features_dir, "test", npz_name=args.npz_name)
    log(f"[data] using npz: {args.npz_name}  text_dim={train_ds.text.shape[1]}")
    log(f"[data] train={len(train_ds)} dev={len(dev_ds)} test={len(test_ds)}")

    gen = torch.Generator()
    gen.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=worker_init,
        generator=gen,
        collate_fn=collate,
        drop_last=False,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=worker_init,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=worker_init,
        collate_fn=collate,
    )

    # Class weights from train labels
    train_labels_np = train_ds.labels.numpy()
    cw = class_weights_inverse_sqrt(train_labels_np).to(device)
    log(f"[loss] class weights: {[round(float(x), 3) for x in cw.tolist()]}")

    # Size the speaker embedding from the data (MELD has 304 speakers)
    max_speaker = int(max(
        train_ds.speaker.max().item(),
        dev_ds.speaker.max().item(),
        test_ds.speaker.max().item(),
    ))  # already +1-shifted in the dataset
    num_speakers = max_speaker  # embedding will be sized num_speakers+1 internally
    log(f"[model] num_speakers (post-shift index max): {num_speakers}")

    model = CareMerc(num_speakers=num_speakers, text_dim=args.text_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[model] params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.warmup, args.epochs)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    history = []
    best_dev_wf1 = -1.0
    best_epoch = -1
    bad_epochs = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_seen = 0
        for step, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            audio_in = batch["audio"]
            visual_in = batch["visual"]
            if torch.rand(1).item() < args.audio_drop:
                audio_in = torch.zeros_like(audio_in)
            if torch.rand(1).item() < args.visual_drop:
                visual_in = torch.zeros_like(visual_in)
            out = model(
                text=batch["text"],
                audio=audio_in,
                visual=visual_in,
                speaker_id=batch["speaker_id"],
                position=batch["position"],
                prev_emotion=batch["prev_emotion"],
            )
            loss = criterion(out["emotion_logits"], batch["emotion_label"])
            lv = float(loss.item())
            if not math.isfinite(lv) or lv > 50:
                log(
                    f"[ep {epoch:02d} step {step:03d}] BAD loss={lv}  "
                    f"logit_abs_max={float(out['emotion_logits'].abs().max().item()):.3e}  "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += lv * batch["emotion_label"].size(0)
            n_seen += batch["emotion_label"].size(0)
        scheduler.step()
        train_loss = epoch_loss / max(1, n_seen)

        dev = evaluate(model, dev_loader, device)
        lr_now = optimizer.param_groups[0]["lr"]
        log(
            f"[ep {epoch:02d}] lr={lr_now:.2e} train_loss={train_loss:.4f}  "
            f"dev acc={dev['acc']:.4f} wF1={dev['wf1']:.4f} mF1={dev['mf1']:.4f}"
        )
        history.append({
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_loss,
            "dev_acc": dev["acc"],
            "dev_wf1": dev["wf1"],
            "dev_mf1": dev["mf1"],
        })

        if dev["wf1"] > best_dev_wf1:
            best_dev_wf1 = dev["wf1"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "dev_wf1": dev["wf1"],
                "dev_mf1": dev["mf1"],
                "args": vars(args),
            }, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                log(f"[early-stop] no dev_wF1 improvement for {bad_epochs} epochs (best ep={best_epoch})")
                break

    train_time = time.time() - t0
    log(f"[train] done in {train_time/60:.1f} min  best dev wF1={best_dev_wf1:.4f} (ep {best_epoch})")

    # Reload best checkpoint and evaluate on test
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    test = evaluate(model, test_loader, device)
    dev_final = evaluate(model, dev_loader, device)

    per_class_f1 = f1_score(
        test["labels"], test["preds"], labels=list(range(7)), average=None, zero_division=0
    ).tolist()
    cm = confusion_matrix(test["labels"], test["preds"], labels=list(range(7))).tolist()
    report = classification_report(
        test["labels"], test["preds"],
        labels=list(range(7)), target_names=EMOTIONS, digits=4, zero_division=0,
    )

    log("[test] at best-dev checkpoint:")
    log(f"  acc={test['acc']:.4f}  wF1={test['wf1']:.4f}  mF1={test['mf1']:.4f}")
    log("[test] per-class F1: " + ", ".join(
        f"{n}={v:.4f}" for n, v in zip(EMOTIONS, per_class_f1)
    ))
    log("\n" + report)

    results = {
        "run_name": run_name,
        "args": vars(args),
        "best_epoch": best_epoch,
        "best_dev_wf1": best_dev_wf1,
        "train_time_min": train_time / 60.0,
        "dev_final": {"acc": dev_final["acc"], "wf1": dev_final["wf1"], "mf1": dev_final["mf1"]},
        "test": {
            "acc": test["acc"],
            "wf1": test["wf1"],
            "mf1": test["mf1"],
            "per_class_f1": dict(zip(EMOTIONS, per_class_f1)),
            "confusion_matrix": cm,
        },
        "history": history,
        "class_weights": [float(x) for x in cw.cpu().tolist()],
    }
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)
    log(f"[save] {results_path}")
    log(f"[save] {ckpt_path}")
    log_fh.close()

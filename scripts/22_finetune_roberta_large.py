"""Fine-tune RoBERTa-large on MELD train for emotion classification, then
extract [CLS] features for train/dev/test.

Same setup as the RoBERTa-base fine-tune (scripts/20) except: model=roberta-large,
batch_size=8, epochs=3.

Outputs:
  features/{split}/text_finetuned_large.npy   (N, 1024) float32
  features/{split}/consolidated_v3.npz        (text swapped, A/V/labels/etc. unchanged)
  checkpoints/roberta_large_meld_ft.pt
  results/roberta_large_finetune.json
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
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizerFast, RobertaModel
from sklearn.metrics import f1_score, accuracy_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

EMOTION_ORDER = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
EMOTION_TO_ID = {e: i for i, e in enumerate(EMOTION_ORDER)}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


class MELDTextDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer, max_len: int = 128):
        df = pd.read_csv(csv_path)
        self.texts = df["Utterance"].astype(str).tolist()
        self.labels = df["emotion_label"].astype(np.int64).to_numpy()
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tok(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


class RobertaForEmotion(nn.Module):
    def __init__(self, name: str = "roberta-base", num_labels: int = 7, dropout: float = 0.1):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(name)
        hidden = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # [CLS] = first token of last hidden state
        cls = out.last_hidden_state[:, 0, :]
        logits = self.classifier(self.drop(cls))
        return logits, cls  # also return cls features


def class_weights_inverse_sqrt(labels: np.ndarray, K: int = 7) -> torch.Tensor:
    counts = np.bincount(labels, minlength=K).astype(np.float64)
    inv = 1.0 / np.sqrt(counts)
    w = K * inv / inv.sum()
    return torch.tensor(w, dtype=torch.float32)


def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for b in loader:
            ids = b["input_ids"].to(device)
            am = b["attention_mask"].to(device)
            lbl = b["label"].to(device)
            logits, _ = model(ids, am)
            all_logits.append(logits.cpu())
            all_labels.append(lbl.cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = logits.argmax(axis=1)
    return {
        "acc": float(accuracy_score(labels, preds)),
        "wf1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "mf1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def extract_features(model, loader, device) -> np.ndarray:
    model.eval()
    feats = []
    with torch.no_grad():
        for b in loader:
            ids = b["input_ids"].to(device)
            am = b["attention_mask"].to(device)
            _, cls = model(ids, am)
            feats.append(cls.cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--model_name", type=str, default="roberta-large")
    ap.add_argument("--features_dir", type=str, default=str(REPO_ROOT / "features"))
    ap.add_argument("--ckpt_dir", type=str, default=str(REPO_ROOT / "checkpoints"))
    ap.add_argument("--results_dir", type=str, default=str(REPO_ROOT / "results"))
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"[run] device={device} model={args.model_name}", flush=True)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print("[load] tokenizer", flush=True)
    tok = RobertaTokenizerFast.from_pretrained(args.model_name)

    train_csv = REPO_ROOT / "data/processed/train_processed.csv"
    dev_csv = REPO_ROOT / "data/processed/dev_processed.csv"
    test_csv = REPO_ROOT / "data/processed/test_processed.csv"

    train_ds = MELDTextDataset(str(train_csv), tok, args.max_len)
    dev_ds = MELDTextDataset(str(dev_csv), tok, args.max_len)
    test_ds = MELDTextDataset(str(test_csv), tok, args.max_len)
    print(f"[data] train={len(train_ds)} dev={len(dev_ds)} test={len(test_ds)}", flush=True)

    gen = torch.Generator(); gen.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, generator=gen)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    # In-order loaders for feature extraction (must match consolidated.npz row order)
    train_extract = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dev_extract = dev_loader
    test_extract = test_loader

    cw = class_weights_inverse_sqrt(train_ds.labels).to(device)
    print(f"[loss] class weights: {[round(float(x), 3) for x in cw.tolist()]}", flush=True)

    print("[load] model", flush=True)
    model = RobertaForEmotion(args.model_name).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    ckpt_path = Path(args.ckpt_dir) / "roberta_large_meld_ft.pt"
    history = []
    best_dev_wf1 = -1.0
    best_epoch = -1
    bad_epochs = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        n_seen = 0
        for step, b in enumerate(train_loader):
            ids = b["input_ids"].to(device)
            am = b["attention_mask"].to(device)
            lbl = b["label"].to(device)
            logits, _ = model(ids, am)
            loss = criterion(logits, lbl)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.item()) * lbl.size(0)
            n_seen += lbl.size(0)
        train_loss = running / max(1, n_seen)
        dev = evaluate(model, dev_loader, device)
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"[ep {epoch}] lr={lr_now:.2e} train_loss={train_loss:.4f}  "
            f"dev acc={dev['acc']:.4f} wF1={dev['wf1']:.4f} mF1={dev['mf1']:.4f}",
            flush=True,
        )
        history.append({"epoch": epoch, "lr": lr_now, "train_loss": train_loss, **{f"dev_{k}": v for k, v in dev.items()}})
        if dev["wf1"] > best_dev_wf1:
            best_dev_wf1 = dev["wf1"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save({"state_dict": model.state_dict(), "args": vars(args), "epoch": epoch, "dev": dev}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[early-stop] best ep={best_epoch} dev_wF1={best_dev_wf1:.4f}", flush=True)
                break

    train_time = time.time() - t0
    print(f"[train] done in {train_time/60:.1f} min  best dev wF1={best_dev_wf1:.4f} (ep {best_epoch})", flush=True)

    # Load best checkpoint
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    test = evaluate(model, test_loader, device)
    print(f"[test] acc={test['acc']:.4f}  wF1={test['wf1']:.4f}  mF1={test['mf1']:.4f}", flush=True)

    print("[extract] computing features per split", flush=True)
    feats = {}
    for split, loader_ in [("train", train_extract), ("dev", dev_extract), ("test", test_extract)]:
        f = extract_features(model, loader_, device)
        out = Path(args.features_dir) / split / "text_finetuned_large.npy"
        np.save(out, f)
        print(f"  saved {out}  shape={f.shape}  dtype={f.dtype}", flush=True)
        feats[split] = f

    # Build consolidated_v3.npz for each split with text swapped
    print("[merge] building consolidated_v3.npz", flush=True)
    for split in ["train", "dev", "test"]:
        old = np.load(Path(args.features_dir) / split / "consolidated.npz", allow_pickle=True)
        assert feats[split].shape[0] == old["labels"].shape[0], "row count mismatch"
        out = Path(args.features_dir) / split / "consolidated_v3.npz"
        np.savez(
            out,
            text=feats[split].astype(np.float32),
            audio=old["audio"],
            visual=old["visual"],
            speaker=old["speaker"],
            labels=old["labels"],
            dialogue=old["dialogue"],
            utterance=old["utterance"],
        )
        print(f"  saved {out}", flush=True)

    results = {
        "args": vars(args),
        "best_epoch": best_epoch,
        "best_dev_wf1": best_dev_wf1,
        "train_time_min": train_time / 60.0,
        "test": test,
        "history": history,
        "n_params": n_params,
    }
    with open(Path(args.results_dir) / "roberta_large_finetune.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[save] {Path(args.results_dir) / 'roberta_large_finetune.json'}", flush=True)


if __name__ == "__main__":
    main()

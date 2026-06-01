"""Linear-probe sanity check: text-only vs. concat(text+audio+visual).

Fits sklearn LogisticRegression probes on the consolidated.npz features to
measure how much class-discriminative signal each modality carries before any
fusion model is trained.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATS = REPO_ROOT / "features"
RESULTS = REPO_ROOT / "results"
EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


def load_split(split: str):
    d = np.load(FEATS / split / "consolidated.npz", allow_pickle=True)
    return {
        "text": d["text"].astype(np.float32),
        "audio": d["audio"].astype(np.float32),
        "visual": d["visual"].astype(np.float32),
        "labels": d["labels"].astype(np.int64),
    }


def run_probe(name: str, X_train, y_train, X_test, y_test, class_weight=None):
    t0 = time.time()
    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        random_state=42,
        class_weight=class_weight,
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    wf1 = f1_score(y_test, preds, average="weighted", zero_division=0)
    mf1 = f1_score(y_test, preds, average="macro", zero_division=0)
    acc = accuracy_score(y_test, preds)
    per = f1_score(y_test, preds, labels=list(range(7)), average=None, zero_division=0).tolist()
    report = classification_report(
        y_test, preds, labels=list(range(7)),
        target_names=EMOTIONS, digits=4, zero_division=0,
    )
    elapsed = time.time() - t0
    print(f"\n=== {name} ({elapsed:.1f}s) ===")
    print(f"acc={acc:.4f}  wF1={wf1:.4f}  mF1={mf1:.4f}")
    for n, v in zip(EMOTIONS, per):
        print(f"  {n:9s}: F1={v:.4f}")
    print(report)
    return {
        "name": name,
        "acc": float(acc),
        "wf1": float(wf1),
        "mf1": float(mf1),
        "per_class_f1": dict(zip(EMOTIONS, per)),
        "elapsed_s": elapsed,
    }


def main() -> None:
    print("Loading features ...")
    train = load_split("train")
    test = load_split("test")
    print(f"  train: text={train['text'].shape}  audio={train['audio'].shape}  visual={train['visual'].shape}")
    print(f"  test:  text={test['text'].shape}   audio={test['audio'].shape}   visual={test['visual'].shape}")

    results = []
    # 1. Text only, unweighted
    results.append(run_probe(
        "text_only_unweighted",
        train["text"], train["labels"], test["text"], test["labels"],
        class_weight=None,
    ))
    # 2. Text only, balanced class weights
    results.append(run_probe(
        "text_only_balanced",
        train["text"], train["labels"], test["text"], test["labels"],
        class_weight="balanced",
    ))
    # 3. Concat(T,A,V) unweighted
    X_train_concat = np.concatenate([train["text"], train["audio"], train["visual"]], axis=1)
    X_test_concat = np.concatenate([test["text"], test["audio"], test["visual"]], axis=1)
    print(f"\n[concat] shape: train={X_train_concat.shape} test={X_test_concat.shape}")
    results.append(run_probe(
        "concat_TAV_unweighted",
        X_train_concat, train["labels"], X_test_concat, test["labels"],
        class_weight=None,
    ))

    # Audit expected vs. computed
    expected = {
        "text_only_unweighted":   {"wf1": 0.580, "mf1": 0.405},
        "text_only_balanced":     {"wf1": 0.476, "mf1": 0.326},
        "concat_TAV_unweighted":  {"wf1": 0.510, "mf1": None},
    }
    print("\n=== Summary vs. audit expectations ===")
    print(f"{'probe':<30} {'wF1':>8} {'audit':>8} {'Δ':>8}  {'mF1':>8} {'audit':>8} {'Δ':>8}")
    for r in results:
        e = expected[r["name"]]
        dw = r["wf1"] - e["wf1"] if e["wf1"] is not None else None
        dm = r["mf1"] - e["mf1"] if e["mf1"] is not None else None
        dw_s = f"{dw:+.4f}" if dw is not None else "  n/a "
        dm_s = f"{dm:+.4f}" if dm is not None else "  n/a "
        e_w = f"{e['wf1']:.3f}" if e["wf1"] is not None else "  n/a"
        e_m = f"{e['mf1']:.3f}" if e["mf1"] is not None else "  n/a"
        print(f"{r['name']:<30} {r['wf1']:>8.4f} {e_w:>8} {dw_s:>8}  {r['mf1']:>8.4f} {e_m:>8} {dm_s:>8}")

    out_path = RESULTS / "text_lr_probe.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "expected": expected}, f, indent=2)
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()

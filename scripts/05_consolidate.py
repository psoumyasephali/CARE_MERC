"""Combine per-modality feature dicts into a single .npz file per split."""
import numpy as np, pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC = PROJECT_ROOT / "data" / "processed"
FEAT = PROJECT_ROOT / "features"

for split in ["train", "dev", "test"]:
    df = pd.read_csv(PROC / f"{split}_processed.csv")
    fdir = FEAT / split
    audio = np.load(fdir / "audio_features_wavlm_large.npy", allow_pickle=True).item()
    text  = np.load(fdir / "text_features_roberta_large.npy",  allow_pickle=True).item()
    visual = np.load(fdir / "visual_features_face_vit.npy",    allow_pickle=True).item()
    A, T, V, S, L, D, U = [], [], [], [], [], [], []
    for _, row in df.iterrows():
        uid = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}"
        A.append(audio.get(uid, np.zeros(1024, dtype=np.float32)))
        T.append(text.get(uid,  np.zeros(1024, dtype=np.float32)))
        V.append(visual.get(uid, np.zeros(768, dtype=np.float32)))
        S.append(row["speaker_id"]); L.append(row["emotion_label"])
        D.append(row["Dialogue_ID"]); U.append(row["Utterance_ID"])
    out = fdir / "consolidated.npz"
    np.savez(out,
             audio=np.array(A, dtype=np.float32),
             text=np.array(T, dtype=np.float32),
             visual=np.array(V, dtype=np.float32),
             speaker=np.array(S, dtype=np.int64),
             labels=np.array(L, dtype=np.int64),
             dialogue=np.array(D, dtype=np.int64),
             utterance=np.array(U, dtype=np.int64))
    print(f"{split}: saved {out}")

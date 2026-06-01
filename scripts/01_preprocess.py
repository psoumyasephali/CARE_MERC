"""Preprocess MELD: encode emotion + speaker labels, attach video paths."""
import os, json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT = json.load(open(PROJECT_ROOT / "data" / "processed" / "audit_report.json"))
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

csv_paths = AUDIT["csv_paths"]
video_folders = AUDIT["video_folders"]

print("Loading CSVs...")
dfs = {split: pd.read_csv(csv_paths[f"{split}_sent_emo.csv"])
       for split in ["train", "dev", "test"]}
for split, df in dfs.items():
    print(f"  {split}: {len(df)} utterances")

# Fit encoders on union of all splits
all_emotions = pd.concat([dfs[s]["Emotion"] for s in dfs])
all_speakers = pd.concat([dfs[s]["Speaker"] for s in dfs])
emo_enc = LabelEncoder().fit(all_emotions)
spk_enc = LabelEncoder().fit(all_speakers)
print(f"\nEmotion classes ({len(emo_enc.classes_)}): {list(emo_enc.classes_)}")
print(f"Speaker count: {len(spk_enc.classes_)}")

# Process each split
for split in ["train", "dev", "test"]:
    df = dfs[split].copy()
    df["emotion_label"] = emo_enc.transform(df["Emotion"])
    df["speaker_id"]    = spk_enc.transform(df["Speaker"])
    folder = Path(video_folders[split]["path"])
    abs_paths = df.apply(
        lambda r: folder / f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}.mp4",
        axis=1)
    # Store repo-relative paths so the CSV stays portable across machines.
    df["video_path"] = abs_paths.apply(lambda p: os.path.relpath(p, PROJECT_ROOT))
    # Drop rows whose video file is missing
    n_before = len(df)
    df["video_exists"] = abs_paths.apply(lambda p: Path(p).exists())
    df = df[df["video_exists"]].drop(columns=["video_exists"]).reset_index(drop=True)
    n_after = len(df)
    if n_before != n_after:
        print(f"  {split}: dropped {n_before - n_after} rows with missing videos")
    # Sort by dialogue, utterance for context modeling
    df = df.sort_values(["Dialogue_ID", "Utterance_ID"]).reset_index(drop=True)
    out_csv = OUT_DIR / f"{split}_processed.csv"
    df.to_csv(out_csv, index=False)
    print(f"  saved {out_csv}  rows={len(df)}")

# Persist encoders + metadata
np.save(OUT_DIR / "emotion_classes.npy", emo_enc.classes_)
np.save(OUT_DIR / "speaker_classes.npy", spk_enc.classes_)
metadata = {
    "num_emotions": len(emo_enc.classes_),
    "num_speakers": len(spk_enc.classes_),
    "emotion_labels": emo_enc.classes_.tolist(),
    "speaker_labels": spk_enc.classes_.tolist(),
    "train_size": len(dfs["train"]),
    "dev_size":   len(dfs["dev"]),
    "test_size":  len(dfs["test"]),
}
with open(OUT_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print(f"\nMetadata: {metadata}")

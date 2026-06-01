"""Pre-flight audit for the CARE-MERC project.
Validates: Mac environment, archive presence, archive contents, CSV columns,
video paths, label distribution, and disk space.
Writes: data/processed/audit_report.json
Halts: if any critical check fails."""
from __future__ import annotations
import os, sys, json, shutil, zipfile, platform, subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_PATH = Path(os.environ.get("MELD_ARCHIVE",
                    PROJECT_ROOT / "archive.zip"))
RAW_DIR  = PROJECT_ROOT / "data" / "raw"
OUT_DIR  = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

report: dict = {"phase": 0, "checks": [], "fatal": [], "warnings": []}

def check(name, ok, detail=""):
    report["checks"].append({"name": name, "ok": bool(ok), "detail": str(detail)})
    flag = "✓" if ok else "✗"
    print(f"  {flag} {name}: {detail}")
    return ok

def fatal(msg):
    report["fatal"].append(msg)
    print(f"\n[FATAL] {msg}")

# ---- 1. Mac environment ----
print("\n=== 1. Mac environment ===")
chip = platform.processor() or platform.machine()
is_apple_silicon = "arm" in chip.lower() or "apple" in chip.lower()
check("CPU arch", True, chip)

try:
    out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    check("CPU model", True, out)
    is_m1_or_newer = "Apple M" in out
except Exception as e:
    is_m1_or_newer = is_apple_silicon
    check("CPU model", False, str(e))

mem_gb = float("nan")
try:
    mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip())
    mem_gb = mem_bytes / (1024**3)
    check("RAM ≥ 14 GB", mem_gb >= 14, f"{mem_gb:.1f} GB")
    if mem_gb < 14:
        report["warnings"].append("Less than 16 GB RAM - reduce BATCH to 16 in training script")
except Exception as e:
    check("RAM check", False, str(e))

# ---- 2. Python & PyTorch ----
print("\n=== 2. Python & PyTorch ===")
py_ver = sys.version.split()[0]
check("Python ≥ 3.10", tuple(map(int, py_ver.split(".")[:2])) >= (3, 10), py_ver)

try:
    import torch
    check("PyTorch installed", True, torch.__version__)
    mps_ok = torch.backends.mps.is_available()
    check("MPS available", mps_ok, "Apple Silicon GPU acceleration")
    if not mps_ok and is_apple_silicon:
        report["warnings"].append("MPS not available on Apple Silicon - reinstall torch with `pip install torch torchvision torchaudio`")
except ImportError:
    fatal("PyTorch is not installed. Run `pip install -r requirements.txt` first.")

# ---- 3. Disk space ----
print("\n=== 3. Disk space ===")
free_gb = shutil.disk_usage(PROJECT_ROOT).free / (1024**3)
check("Free disk ≥ 30 GB", free_gb >= 30, f"{free_gb:.1f} GB free")
if free_gb < 30:
    report["warnings"].append("Less than 30 GB free disk - feature files alone need ~5 GB; archive extraction needs ~10 GB")

# ---- 4. Archive presence ----
# Two supported flows: (a) a Kaggle archive.zip we extract here, or (b) raw data
# already extracted into data/raw/ (the official-release flow in the README).
# A missing archive is only fatal if the raw data is not already present.
print("\n=== 4. Kaggle MELD archive ===")
already_extracted = any(RAW_DIR.glob("**/train_sent_emo.csv"))
if not check("archive.zip exists", ARCHIVE_PATH.exists(), str(ARCHIVE_PATH)):
    if already_extracted:
        check("Raw data already extracted", True, "archive not needed")
    else:
        fatal(f"archive.zip not found at {ARCHIVE_PATH} and no extracted raw data "
              f"in {RAW_DIR}. Set MELD_ARCHIVE, or extract MELD.Raw into data/raw/.")
else:
    size_mb = ARCHIVE_PATH.stat().st_size / (1024**2)
    check("Archive size sensible", size_mb > 1000, f"{size_mb:.1f} MB (expected ~2500 MB)")

# ---- 5. Archive contents (without extracting yet) ----
print("\n=== 5. Archive contents ===")
csv_names_seen = []
mp4_count = 0
roots = set()
if ARCHIVE_PATH.exists():
    try:
        with zipfile.ZipFile(ARCHIVE_PATH) as zf:
            names = zf.namelist()
            check("Archive opens", True, f"{len(names)} entries")
            for n in names:
                if n.endswith(".csv"):
                    csv_names_seen.append(n)
                if n.endswith(".mp4"):
                    mp4_count += 1
                    roots.add(n.split("/")[0])
            check("CSV files present", len(csv_names_seen) >= 3, f"found {len(csv_names_seen)}: {csv_names_seen[:5]}")
            check("MP4 files present", mp4_count > 10000, f"{mp4_count} mp4 files")
            check("Top-level root", len(roots) > 0, f"roots: {list(roots)[:3]}")
    except Exception as e:
        fatal(f"Could not read archive: {e}")

# ---- 6. Extract if not already ----
print("\n=== 6. Extraction ===")
RAW_DIR.mkdir(parents=True, exist_ok=True)
if already_extracted:
    check("Already extracted", True, "skipping extraction")
else:
    if ARCHIVE_PATH.exists():
        print(f"  Extracting {ARCHIVE_PATH} to {RAW_DIR} (this takes 5-10 min)...")
        try:
            with zipfile.ZipFile(ARCHIVE_PATH) as zf:
                zf.extractall(RAW_DIR)
            check("Extracted", True, str(RAW_DIR))
        except Exception as e:
            fatal(f"Extraction failed: {e}")

# ---- 7. Locate CSVs (Kaggle layout varies) ----
print("\n=== 7. CSV discovery ===")
expected_csvs = ["train_sent_emo.csv", "dev_sent_emo.csv", "test_sent_emo.csv"]
csv_paths: dict = {}
for csv_name in expected_csvs:
    found = list(RAW_DIR.rglob(csv_name))
    if found:
        csv_paths[csv_name] = str(found[0])
        check(csv_name, True, str(found[0]))
    else:
        check(csv_name, False, "NOT FOUND - Kaggle MELD layout may differ")
        fatal(f"Missing CSV: {csv_name}")

# ---- 8. Video folder discovery ----
print("\n=== 8. Video folder discovery ===")
# Common Kaggle/MELD folder names
candidate_folders = {
    "train": ["train_splits", "train_splits_complete"],
    "dev":   ["dev_splits_complete", "dev_splits"],
    "test":  ["output_repeated_splits_test", "test_splits_complete", "test_splits"],
}
video_folders: dict = {}
for split, options in candidate_folders.items():
    found = None
    for opt in options:
        hits = list(RAW_DIR.rglob(opt))
        if hits and any(hits[0].rglob("*.mp4")):
            found = hits[0]
            break
    if found:
        n_mp4 = len(list(found.glob("*.mp4")))
        video_folders[split] = {"path": str(found), "count": n_mp4}
        check(f"{split} videos", n_mp4 > 100, f"{n_mp4} mp4 files in {found.name}")
    else:
        fatal(f"Could not find video folder for {split} split. Looked for: {options}")

# ---- 9. CSV column sanity ----
print("\n=== 9. CSV column sanity ===")
try:
    import pandas as pd
    for csv_name, csv_path in csv_paths.items():
        df = pd.read_csv(csv_path)
        required_cols = {"Utterance", "Speaker", "Emotion", "Dialogue_ID", "Utterance_ID"}
        missing = required_cols - set(df.columns)
        check(f"{csv_name} columns", not missing,
              f"shape={df.shape}, missing={missing if missing else 'none'}")
        if missing:
            fatal(f"{csv_name} is missing columns: {missing}")
        # Emotion label distribution
        if csv_name == "train_sent_emo.csv":
            emo_counts = df["Emotion"].value_counts().to_dict()
            report["train_emotion_counts"] = emo_counts
            print(f"    train emotion counts: {emo_counts}")
except Exception as e:
    fatal(f"CSV parsing failed: {e}")

# ---- 10. Sample video readability ----
print("\n=== 10. Sample video readability ===")
try:
    import cv2, torchaudio
    sample_split = "train"
    sample_csv = pd.read_csv(csv_paths["train_sent_emo.csv"]).head(3)
    folder = Path(video_folders[sample_split]["path"])
    for _, row in sample_csv.iterrows():
        vp = folder / f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
        if not vp.exists():
            check("video exists", False, str(vp))
            continue
        cap = cv2.VideoCapture(str(vp))
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        wav, sr = torchaudio.load(str(vp))
        check(f"{vp.name} readable", nframes > 0 and wav.shape[1] > 0,
              f"frames={nframes}, audio={wav.shape}, sr={sr}")
        break
except Exception as e:
    fatal(f"Video/audio readability failed: {e}")

# ---- 11. Persist report ----
report["csv_paths"] = csv_paths
report["video_folders"] = video_folders
report["mac"] = {"chip": chip, "ram_gb": round(mem_gb, 1)}
out_path = OUT_DIR / "audit_report.json"
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nAudit report written to {out_path}")

# ---- 12. Final verdict ----
print("\n" + "=" * 60)
if report["fatal"]:
    print("AUDIT FAILED - fix issues above before proceeding.")
    for m in report["fatal"]:
        print(f"  · {m}")
    sys.exit(1)
else:
    print("AUDIT PASSED")
    if report["warnings"]:
        print("Warnings (non-fatal):")
        for w in report["warnings"]:
            print(f"  · {w}")
    print("=" * 60)

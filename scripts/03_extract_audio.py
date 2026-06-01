"""Extract WavLM-large features (audio decoded from .mp4 via ffmpeg)."""
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import shutil, subprocess
import torch, torchaudio, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import WavLMModel, AutoFeatureExtractor

FFMPEG = shutil.which("ffmpeg")


def _decode_via_ffmpeg(video_path: str, sr: int = 16000):
    """Fallback decoder for mp4s torchaudio can't open (e.g. 6ch unknown layout)."""
    if FFMPEG is None:
        raise RuntimeError("ffmpeg not found on PATH; install it (e.g. `brew install ffmpeg`).")
    cmd = [FFMPEG, "-v", "error", "-i", str(video_path),
           "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    pcm = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    return torch.from_numpy(pcm).unsqueeze(0), sr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC = PROJECT_ROOT / "data" / "processed"
FEAT = PROJECT_ROOT / "features"

def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AudioEnc:
    def __init__(self):
        self.device = pick_device()
        print(f"Loading WavLM-large on {self.device}...")
        self.fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
        # fp32 on MPS: WavLM's parametrized conv layers don't cast cleanly with
        # torch_dtype=fp16 (mixed param dtypes after load → MPS conv1d errors).
        self.model = WavLMModel.from_pretrained(
            "microsoft/wavlm-large").to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.max_len = 160000  # 10 s @ 16 kHz

    @torch.no_grad()
    def encode(self, video_path: str):
        try:
            try:
                wav, sr = torchaudio.load(video_path)
            except Exception:
                # 6ch / unknown channel_layout clips → ffmpeg subprocess fallback
                wav, sr = _decode_via_ffmpeg(video_path, sr=16000)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != 16000:
                wav = torchaudio.transforms.Resample(sr, 16000)(wav)
            if wav.shape[1] > self.max_len:
                wav = wav[:, :self.max_len]
            elif wav.shape[1] < self.max_len:
                wav = torch.nn.functional.pad(wav, (0, self.max_len - wav.shape[1]))
            inp = self.fe(wav.squeeze().numpy(), sampling_rate=16000,
                          return_tensors="pt").to(self.device)
            out = self.model(**inp).last_hidden_state    # (1, T, 1024)
            attn = torch.softmax(out.norm(dim=-1), dim=-1)
            feat = (out * attn.unsqueeze(-1)).sum(dim=1).squeeze()
            return feat.float().cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"  audio err {video_path}: {e}")
            return np.zeros(1024, dtype=np.float32)

def main():
    enc = AudioEnc()
    for split in ["train", "dev", "test"]:
        df = pd.read_csv(PROC / f"{split}_processed.csv")
        out = FEAT / split
        out.mkdir(parents=True, exist_ok=True)
        feats: dict = {}
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Audio {split}"):
            uid = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}"
            vp = row["video_path"]
            vp = vp if os.path.isabs(vp) else str(PROJECT_ROOT / vp)
            feats[uid] = enc.encode(vp)
        np.save(out / "audio_features_wavlm_large.npy", feats)
        print(f"  saved {out/'audio_features_wavlm_large.npy'}  n={len(feats)}")

if __name__ == "__main__":
    main()

"""Extract RoBERTa-large [CLS] features."""
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC = PROJECT_ROOT / "data" / "processed"
FEAT = PROJECT_ROOT / "features"

def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextEnc:
    def __init__(self):
        self.device = pick_device()
        print(f"Loading RoBERTa-large on {self.device}...")
        self.tok = RobertaTokenizer.from_pretrained("roberta-large")
        dtype = torch.float16 if self.device.type != "cpu" else torch.float32
        self.model = RobertaModel.from_pretrained(
            "roberta-large", torch_dtype=dtype).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, text: str, max_len: int = 128):
        try:
            inp = self.tok(str(text), padding="max_length", truncation=True,
                           max_length=max_len, return_tensors="pt").to(self.device)
            out = self.model(**inp)
            cls = out.last_hidden_state[:, 0, :].squeeze()
            return cls.float().cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"  text err: {e}")
            return np.zeros(1024, dtype=np.float32)

def main():
    enc = TextEnc()
    for split in ["train", "dev", "test"]:
        df = pd.read_csv(PROC / f"{split}_processed.csv")
        out = FEAT / split
        out.mkdir(parents=True, exist_ok=True)
        feats: dict = {}
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Text {split}"):
            uid = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}"
            feats[uid] = enc.encode(row["Utterance"])
        np.save(out / "text_features_roberta_large.npy", feats)
        print(f"  saved {out/'text_features_roberta_large.npy'}  n={len(feats)}")

if __name__ == "__main__":
    main()

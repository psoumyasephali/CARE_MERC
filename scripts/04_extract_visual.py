"""Sample 8 frames per video → detect largest face (MTCNN) → ViT-base."""
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2, torch, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN
from transformers import ViTModel, ViTImageProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC = PROJECT_ROOT / "data" / "processed"
FEAT = PROJECT_ROOT / "features"

def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VisualEnc:
    def __init__(self, num_frames: int = 8):
        self.device = pick_device()
        self.num_frames = num_frames
        print(f"Loading MTCNN + ViT-base on {self.device}...")
        # MTCNN itself often runs better on CPU on Mac; safe fallback
        self.mtcnn = MTCNN(image_size=224, margin=20, post_process=False,
                           device="cpu", select_largest=True)
        self.proc = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        dtype = torch.float16 if self.device.type != "cpu" else torch.float32
        self.vit = ViTModel.from_pretrained(
            "google/vit-base-patch16-224", torch_dtype=dtype).to(self.device).eval()
        for p in self.vit.parameters():
            p.requires_grad = False
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    @torch.no_grad()
    def encode(self, video_path: str):
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            return np.zeros(768, dtype=np.float32)
        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        crops = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame)
            face = self.mtcnn(pil)
            if face is None:
                # fallback: center-crop
                h, w, _ = frame.shape
                s = min(h, w)
                cx, cy = w // 2, h // 2
                fc = frame[cy - s//2:cy + s//2, cx - s//2:cx + s//2]
                fc = cv2.resize(fc, (224, 224))
                inp = self.proc(images=Image.fromarray(fc), return_tensors="pt")
                face = inp["pixel_values"].squeeze(0)
            else:
                face = face.float() / 255.0
                face = (face - self._mean) / self._std
            crops.append(face)
        cap.release()
        if not crops:
            return np.zeros(768, dtype=np.float32)
        batch = torch.stack(crops).to(self.device)
        if self.vit.dtype == torch.float16:
            batch = batch.half()
        out = self.vit(pixel_values=batch).last_hidden_state[:, 0, :]   # (N, 768)
        attn = torch.softmax(out.norm(dim=-1), dim=0)
        feat = (out * attn.unsqueeze(-1)).sum(dim=0)
        return feat.float().cpu().numpy().astype(np.float32)

def main():
    enc = VisualEnc(num_frames=8)
    for split in ["train", "dev", "test"]:
        df = pd.read_csv(PROC / f"{split}_processed.csv")
        out = FEAT / split
        out.mkdir(parents=True, exist_ok=True)
        feats: dict = {}
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Visual {split}"):
            uid = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}"
            vp = row["video_path"]
            vp = vp if os.path.isabs(vp) else str(PROJECT_ROOT / vp)
            feats[uid] = enc.encode(vp)
        np.save(out / "visual_features_face_vit.npy", feats)
        print(f"  saved {out/'visual_features_face_vit.npy'}  n={len(feats)}")

if __name__ == "__main__":
    main()

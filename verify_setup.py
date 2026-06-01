"""Verify the Mac M1 Pro environment for CARE-MERC."""
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch, transformers, sklearn, cv2

print("=" * 60)
print("CARE-MERC: Mac M1 Pro Environment Verification")
print("=" * 60)
print(f"PyTorch         : {torch.__version__}")
print(f"MPS available   : {torch.backends.mps.is_available()}")
print(f"MPS built       : {torch.backends.mps.is_built()}")
print(f"Transformers    : {transformers.__version__}")
print(f"scikit-learn    : {sklearn.__version__}")
print(f"OpenCV          : {cv2.__version__}")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Selected device : {device}")
x = torch.randn(64, 64, device=device)
y = (x @ x.T).sum()
print(f"Test matmul     : OK  (sum={y.item():.2f})")
print("=" * 60)

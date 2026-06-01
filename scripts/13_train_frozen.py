"""Train CARE-MERC on the frozen RoBERTa-large features with modality dropout.

During training, the audio vector is zeroed for the whole batch with p=0.15 and
the visual vector independently with p=0.15 (text is never dropped). Evaluation
uses all three modalities. Shared training logic lives in _train.py.
"""
from _train import build_argparser, run

if __name__ == "__main__":
    parser = build_argparser(tag="frozen", npz_name="consolidated.npz", text_dim=1024)
    run(parser.parse_args())

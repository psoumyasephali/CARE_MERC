"""Train CARE-MERC with modality dropout on the fine-tuned RoBERTa-base text
features (consolidated_v2.npz). text_dim is 768; audio/visual/speaker/labels
are unchanged from the frozen-feature run. Shared logic lives in _train.py.
"""
from _train import build_argparser, run

if __name__ == "__main__":
    parser = build_argparser(tag="ftbase", npz_name="consolidated_v2.npz", text_dim=768)
    run(parser.parse_args())

"""Train CARE-MERC with modality dropout on the fine-tuned RoBERTa-large text
features (consolidated_v3.npz). text_dim=1024; audio/visual/speaker/labels are
unchanged from the other runs. This is the final model in the project. Shared
logic lives in _train.py.
"""
from _train import build_argparser, run

if __name__ == "__main__":
    parser = build_argparser(tag="ftlarge", npz_name="consolidated_v3.npz", text_dim=1024)
    run(parser.parse_args())

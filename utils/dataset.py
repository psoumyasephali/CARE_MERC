"""Dataset wrapper for CARE-MERC.

Reads a consolidated .npz of per-utterance features and exposes, for each
utterance, its position in the dialogue and the previous utterance's emotion
(ground-truth at both train and eval time, following the usual ERC
convention).
"""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


_NO_PREV_TOKEN = 7  # 8-way embedding: classes 0..6 plus this sentinel.


class CareMercDataset(Dataset):
    """Per-utterance dataset for one MELD split.

    Expects ``features_dir/split/npz_name`` with arrays: text, audio, visual,
    speaker, labels, dialogue, utterance. Speaker ids are shifted by +1 so
    index 0 is free for an unknown speaker. ``prev_emotion`` is derived from
    the preceding utterance in the same dialogue, and ``position`` is the
    utterance index. Sentiment labels are loaded only if present (the optional
    auxiliary head is off by default).
    """

    def __init__(self, features_dir: str, split: str, npz_name: str = "consolidated.npz",
                 max_position: int = 50):
        npz_path = os.path.join(features_dir, split, npz_name)
        data = np.load(npz_path, allow_pickle=True)

        self.text = torch.from_numpy(data["text"]).float()
        self.audio = torch.from_numpy(data["audio"]).float()
        self.visual = torch.from_numpy(data["visual"]).float()
        self.speaker = torch.from_numpy(data["speaker"]).long() + 1  # reserve 0 for unknown
        self.labels = torch.from_numpy(data["labels"]).long()
        self.dialogue = data["dialogue"].astype(np.int64)
        self.utterance = data["utterance"].astype(np.int64)

        # Optional: only needed when training with the auxiliary sentiment head.
        sent_path = os.path.join(features_dir, split, "sentiment_labels.npy")
        if os.path.exists(sent_path):
            sentiment = np.load(sent_path).astype(np.int64)
            assert sentiment.shape[0] == self.labels.shape[0], (
                f"sentiment len {sentiment.shape[0]} != labels len {self.labels.shape[0]}"
            )
            self.sentiment = torch.from_numpy(sentiment).long()
        else:
            self.sentiment = torch.zeros(self.labels.shape[0], dtype=torch.long)

        # Build dialogue → indices ordered by utterance position
        by_dialogue: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self.labels)):
            by_dialogue[int(self.dialogue[i])].append(i)
        for did in by_dialogue:
            by_dialogue[did].sort(key=lambda i: int(self.utterance[i]))
        self.by_dialogue = dict(by_dialogue)

        # For each row, prev_emotion = label of preceding utterance in the same
        # dialogue (in available-utterance order). _NO_PREV_TOKEN for first.
        prev = np.full(len(self.labels), _NO_PREV_TOKEN, dtype=np.int64)
        for did, idx_list in self.by_dialogue.items():
            for rank, idx in enumerate(idx_list):
                if rank > 0:
                    prev[idx] = int(self.labels[idx_list[rank - 1]])
        self.prev_emotion = torch.from_numpy(prev).long()

        # Position-in-dialogue from the `utterance` field directly. Observed
        # range on MELD is 0..32 (test); the model's max_position=50 covers it.
        self.position = torch.from_numpy(self.utterance).long()
        assert int(self.position.max()) < max_position, (
            f"utterance position {int(self.position.max())} >= max_position "
            f"{max_position}; raise CareMerc(max_position=...)"
        )

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "text": self.text[idx],
            "audio": self.audio[idx],
            "visual": self.visual[idx],
            "speaker_id": self.speaker[idx],
            "position": self.position[idx],
            "prev_emotion": self.prev_emotion[idx],
            "emotion_label": self.labels[idx],
            "sentiment_label": self.sentiment[idx],
            "dialogue_id": torch.tensor(int(self.dialogue[idx]), dtype=torch.long),
        }


def collate(batch: list[dict]) -> dict:
    out: dict[str, torch.Tensor] = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch])
    return out

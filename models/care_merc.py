"""CARE-MERC: a compact utterance-level model for emotion recognition in
conversation.

Each utterance is described by three feature vectors (text from RoBERTa,
audio from WavLM, visual from a face-cropped ViT) plus a speaker id, a
position-in-dialogue index, and the previous utterance's emotion. The
forward pass projects and fuses the modalities, adds a small contextual and
emotion-transition signal, and predicts one of seven emotions.

See docs/results.md for the architecture description and ablations.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CareMerc(nn.Module):
    def __init__(
        self,
        num_speakers: int = 304,
        num_emotions: int = 7,
        text_dim: int = 1024,
        audio_dim: int = 1024,
        visual_dim: int = 768,
        speaker_emb_dim: int = 64,
        proj_dim: int = 512,
        hidden_dim: int = 256,
        max_position: int = 50,
        eta_emb_dim: int = 64,
        dropout: float = 0.5,
        use_sentiment_head: bool = False,
        num_sentiments: int = 3,
    ):
        """Build the model.

        Args:
            num_speakers: number of distinct speakers; the embedding reserves
                one extra slot (index 0) for unknown speakers.
            num_emotions: number of emotion classes (7 for MELD).
            text_dim / audio_dim / visual_dim: feature widths (1024 / 1024 / 768).
            proj_dim / hidden_dim: internal projection and hidden widths.
            max_position: largest utterance position the model can embed.
            dropout: dropout used in the input block and emotion head.
            use_sentiment_head: optional auxiliary 3-way sentiment head, used in
                an ablation (off in the final model; see docs/results.md).
        """
        super().__init__()
        self.num_emotions = num_emotions
        self.use_sentiment_head = use_sentiment_head

        self.speaker_emb = nn.Embedding(num_speakers + 1, speaker_emb_dim)

        concat_dim = audio_dim + text_dim + visual_dim + speaker_emb_dim
        self.input_proj = nn.Linear(concat_dim, proj_dim)
        self.input_ln = nn.LayerNorm(proj_dim)
        self.input_drop = nn.Dropout(dropout)

        self.position_emb = nn.Embedding(max_position, proj_dim)
        self.cem_gru = nn.GRU(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.cem_proj = nn.Linear(2 * hidden_dim, hidden_dim)

        self.audio_proj = nn.Linear(audio_dim, proj_dim)
        self.text_proj = nn.Linear(text_dim, proj_dim)
        self.visual_proj = nn.Linear(visual_dim, proj_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(3 * proj_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.amg_out = nn.Linear(proj_dim, hidden_dim)

        # 7 emotions + 1 "no-prev" sentinel = 8
        self.eta_emb = nn.Embedding(num_emotions + 1, eta_emb_dim)
        self.eta_proj = nn.Linear(eta_emb_dim, hidden_dim)

        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_emotions),
        )

        if use_sentiment_head:
            self.sentiment_head = nn.Sequential(
                nn.Linear(hidden_dim, eta_emb_dim),
                nn.ReLU(),
                nn.Linear(eta_emb_dim, num_sentiments),
            )

    def _fuse_modalities(self, audio: torch.Tensor, text: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        """Project each modality, learn a softmax weight per modality, and
        return their weighted sum. Lets the model down-weight uninformative
        streams on a per-example basis."""
        a = self.audio_proj(audio)
        t = self.text_proj(text)
        v = self.visual_proj(visual)
        gate_in = torch.cat([a, t, v], dim=-1)
        gates = F.softmax(self.gate_net(gate_in), dim=-1)
        fused = gates[:, 0:1] * a + gates[:, 1:2] * t + gates[:, 2:3] * v
        return self.amg_out(fused)

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        visual: torch.Tensor,
        speaker_id: torch.Tensor,
        position: torch.Tensor,
        prev_emotion: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run a forward pass.

        Inputs are all shape (B, ...): text/audio/visual feature vectors,
        speaker_id, position (utterance index in the dialogue), and
        prev_emotion (label of the previous utterance, or the no-prev
        sentinel for the first utterance).

        Returns a dict with "emotion_logits" (B, num_emotions), "rep" (the
        pre-head representation), and "sentiment_logits" when the optional
        sentiment head is enabled.
        """
        # Step 1+2: speaker emb + concat + projection
        spk = self.speaker_emb(speaker_id)
        x = torch.cat([audio, text, visual, spk], dim=-1)
        x = self.input_proj(x)
        x = self.input_ln(x)
        x = F.relu(x)
        x = self.input_drop(x)  # (B, 512)

        # Step 3: CEM: additive position embed, then BiGRU on length-1 seq, then project
        pos = self.position_emb(position)  # (B, 512)
        x_seq = (x + pos).unsqueeze(1)  # (B, 1, 512)
        gru_out, _ = self.cem_gru(x_seq)  # (B, 1, 512)
        cem = self.cem_proj(gru_out.squeeze(1))  # (B, 256)

        # Step 4: AMG fusion
        amg = self._fuse_modalities(audio, text, visual)  # (B, 256)

        # Step 5: residual combine
        combined = cem + amg  # (B, 256)

        # Step 6: ETA: additive prev-emotion contribution
        eta = self.eta_proj(self.eta_emb(prev_emotion))  # (B, 256)
        combined_with_eta = combined + 0.5 * eta  # (B, 256)

        # Step 7: emotion head
        emotion_logits = self.emotion_head(combined_with_eta)

        out = {"emotion_logits": emotion_logits, "rep": combined_with_eta}
        if self.use_sentiment_head:
            out["sentiment_logits"] = self.sentiment_head(combined_with_eta)
        return out

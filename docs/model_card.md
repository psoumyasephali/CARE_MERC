# Model Card: CARE-MERC

## Model details

- **Architecture:** CARE-MERC, a compact utterance-level classifier for emotion
  recognition in conversation. Speaker-aware embedding, an input projection,
  contextual memory (positional embedding + BiGRU), adaptive modality gating, an
  emotion-transition term, and a linear emotion head. See
  [models/care_merc.py](../models/care_merc.py).
- **Parameters:** about 4.86M trainable.
- **Inputs:** per utterance, three feature vectors (text 1024 from RoBERTa-large,
  audio 1024 from WavLM-large, visual 768 from a ViT-base on MTCNN face crops),
  plus speaker id, position in dialogue, and the previous utterance's emotion.
- **Output:** logits over 7 emotions (anger, disgust, fear, joy, neutral,
  sadness, surprise).
- **Encoders:** text encoder fine-tuned end to end on MELD; audio and visual
  encoders frozen.

## Intended use

Research and educational use for studying multimodal emotion recognition in
conversation, and as a reference for how a fine-tuned text-only control changes
the interpretation of multimodal gains on MELD. Not intended for real-world
affect inference about individuals.

## Training data

MELD (Poria et al., 2019): about 13.7K utterances from the TV series *Friends*,
labelled with 7 emotions, with a 17:1 class imbalance and roughly 48% neutral.
Splits used: 9,989 train / 1,108 dev / 2,610 test (usable after feature
extraction).

## Evaluation

MELD test, 3-seed mean and std (seeds 42, 43, 44):

| Metric | Score |
|---|---|
| Accuracy | 0.6485 ± 0.0061 |
| Weighted F1 | 0.6394 ± 0.0019 |
| Macro F1 | 0.4811 ± 0.0016 |

Per-class F1 and the full analysis are in [results.md](results.md).

## Limitations

- Frozen audio and visual encoders; the text-only equivalence finding is partly
  conditioned on that.
- Trained and evaluated on MELD only (scripted English-language TV dialogue from
  *Friends*); it will not transfer cleanly to other domains, languages, or
  spontaneous speech.
- Utterance-level granularity only.
- Weakest on the rare classes (disgust, fear) despite inverse-frequency class
  weighting.

## Ethical considerations

MELD is derived from a copyrighted TV show; use is governed by the dataset's
license. Emotion labels are annotator judgments of scripted acting, not ground
truth about real affect. Models like this should not be used to make decisions
about real people.

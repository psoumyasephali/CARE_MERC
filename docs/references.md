# References

### Foundation models

- Liu et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* arXiv:1907.11692.
- Chen et al. (2022). *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing.* IEEE JSTSP.
- Dosovitskiy et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR.
- Zhang et al. (2016). *Joint Face Detection and Alignment Using Multi-task Cascaded Convolutional Networks (MTCNN).* IEEE Signal Processing Letters.

### Training techniques considered

- Cao et al. (2019). *Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss (LDAM-DRW).* NeurIPS.
- Khosla et al. (2020). *Supervised Contrastive Learning.* NeurIPS.
- Ganin & Lempitsky (2015). *Unsupervised Domain Adaptation by Backpropagation (gradient reversal).* ICML.
- Zhang et al. (2018). *mixup: Beyond Empirical Risk Minimization.* ICLR.

### Dataset

- Poria et al. (2019). *MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations.* ACL.

### Baselines used in the comparison table

- Majumder et al. (2019). *DialogueRNN: An Attentive RNN for Emotion Detection in Conversations.* AAAI. (Text-only and multimodal DialogueRNN are the MELD-paper baselines.)
- Chudasama et al. (2022). *M2FNet: Multi-modal Fusion Network for Emotion Recognition in Conversation.* CVPR Workshops. (MELD test weighted-F1 ~0.665; the comparison table rounds to 0.665.)
- *Sync-TVA: A Graph-Attention Framework for Multimodal Emotion Recognition with Cross-Modal Fusion* (2025). arXiv:2507.21395 (unrefereed preprint). The 0.674 wF1 / 0.683 acc are the MELD test figures from the paper's main results table; the strongest MELD weighted-F1 we are aware of as of 2025.

### Related work (not directly compared)

- Hu et al. (2022). *MM-DFN: Multimodal Dynamic Fusion Network for Emotion Recognition in Conversations.* ICASSP.
- Ma et al. (2024). *SDT: A Self-distillation Transformer for Multimodal ERC.* IEEE Transactions on Multimedia.
- Ai et al. (2025). *DER-GCN: Dialogue and Event Relation-aware GCN for Multimodal ERC.* IEEE TNNLS.

"""
src/model.py
Stage 3-4: fusion + classifier head.

Fuses frozen CLIP image/text embeddings via concatenation + elementwise
product (following HateCLIPer's fine-grained fusion approach from the
literature survey), then feeds the fused vector through a small MLP.
Only this module is ever trained -- CLIP itself is never touched here.
"""

import torch
import torch.nn as nn


class FusionClassifierHead(nn.Module):
    """
    Input: image_embeds [B, 512], text_embeds [B, 512] (already L2-normalized CLIP embeddings)
    Output: logits [B]  (apply sigmoid externally for probabilities)
    """

    def __init__(self, embed_dim=512, hidden_dim=256, dropout=0.3):
        super().__init__()
        # concat(image, text) -> 2*embed_dim, plus elementwise product -> embed_dim
        fused_dim = embed_dim * 2 + embed_dim

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def fuse(self, image_embeds, text_embeds):
        concat = torch.cat([image_embeds, text_embeds], dim=-1)   # [B, 1024] -- each modality's independent signal
        product = image_embeds * text_embeds                       # [B, 512]  -- cross-modal interaction signal
        return torch.cat([concat, product], dim=-1)                # [B, 1536]

    def forward(self, image_embeds, text_embeds):
        fused = self.fuse(image_embeds, text_embeds)
        logits = self.classifier(fused).squeeze(-1)
        return logits

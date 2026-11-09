import torch
import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, in_features=256, hidden_features=128, dropout=0.25):
        super().__init__()
        # Note: LN is typically applied on the final Transformer output (handled in TransformerEncoder out_ln)
        # So here we can just do the linear layers. But we add an extra LN as per spec:
        # "Head: LN, Linear 256->128, GELU, Dropout, Linear 128->1 -> logit [B, 1]"
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, 1)
        )

    def forward(self, x):
        return self.head(x)

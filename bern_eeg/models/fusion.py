import torch
import torch.nn as nn

class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, ffn_dim=512, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1_q = nn.LayerNorm(embed_dim)
        self.ln1_kv = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, q, kv):
        # q: [B, N_q, D]
        # kv: [B, N_kv, D]
        q_norm = self.ln1_q(q)
        kv_norm = self.ln1_kv(kv)
        
        attn_out, _ = self.mha(q_norm, kv_norm, kv_norm)
        q = q + attn_out
        
        q = q + self.ffn(self.ln2(q))
        return q

class CrossAttentionFusion(nn.Module):
    def __init__(self, num_layers=2, embed_dim=256, num_heads=4, ffn_dim=512, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'raw': CrossAttentionLayer(embed_dim, num_heads, ffn_dim, dropout),
                'short': CrossAttentionLayer(embed_dim, num_heads, ffn_dim, dropout),
                'long': CrossAttentionLayer(embed_dim, num_heads, ffn_dim, dropout),
                'grp': CrossAttentionLayer(embed_dim, num_heads, ffn_dim, dropout),
                'hilbert': CrossAttentionLayer(embed_dim, num_heads, ffn_dim, dropout)
            }) for _ in range(num_layers)
        ])

    def forward(self, tokens_dict):
        # tokens_dict: dict of modality -> [B, N_m, D]
        # We iteratively apply cross attention
        keys = list(tokens_dict.keys())
        
        out_dict = {k: v for k, v in tokens_dict.items()}
        
        for layer in self.layers:
            new_out_dict = {}
            for m in keys:
                # Context is all other modalities concatenated
                ctx_list = [out_dict[k] for k in keys if k != m]
                if ctx_list:
                    ctx = torch.cat(ctx_list, dim=1)
                    if m in layer:
                        new_out_dict[m] = layer[m](out_dict[m], ctx)
                    else:
                        new_out_dict[m] = out_dict[m]
                else:
                    new_out_dict[m] = out_dict[m]
            out_dict = new_out_dict
            
        return out_dict

class TransformerEncoderLayerPreLN(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, ffn_dim=512, dropout=0.25, attn_dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_norm = self.ln1(x)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers=2, embed_dim=256, num_heads=4, ffn_dim=512, dropout=0.25, attn_dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayerPreLN(embed_dim, num_heads, ffn_dim, dropout, attn_dropout)
            for _ in range(num_layers)
        ])
        self.ln_out = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.ln_out(x)

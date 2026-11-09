import torch
import torch.nn as nn
from .raw import RawEncoder
from .stft import STFTTransform, STFTEncoder
from .grp import GRPTransform, GRPEncoder
from .hilbert import HilbertTransform, HilbertEncoder
from .fusion import CrossAttentionFusion, TransformerEncoder
from .classifier import ClassifierHead

class PositionalEncoding(nn.Module):
    def __init__(self, num_tokens, embed_dim):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, embed_dim) * 0.02)
        
    def forward(self, x):
        return x + self.pos_emb

class GatedFusion(nn.Module):
    # Fallback/Ablation baseline fusion
    def __init__(self, embed_dim, num_modalities):
        super().__init__()
        self.attn_pools = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.Tanh(),
                nn.Linear(embed_dim // 2, 1)
            ) for _ in range(num_modalities)
        ])
        
        self.gating = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, num_modalities),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, tokens_list):
        pooled = []
        for i, tokens in enumerate(tokens_list):
            attn = torch.softmax(self.attn_pools[i](tokens), dim=1) # [B, N, 1]
            p = torch.sum(tokens * attn, dim=1) # [B, D]
            pooled.append(p)
            
        concat = torch.cat(pooled, dim=1) # [B, num_modalities * D]
        gates = self.gating(concat) # [B, num_modalities]
        
        # Weighted sum of pooled representations
        fused = torch.zeros_like(pooled[0])
        for i, p in enumerate(pooled):
            fused += p * gates[:, i:i+1]
            
        return fused

class MultiModalEEGModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        embed_dim = cfg.model.embed_dim
        
        self.use_raw = cfg.model.use_raw
        self.use_stft = cfg.model.use_stft
        self.use_grp = cfg.model.use_grp
        self.use_hilbert = cfg.model.use_hilbert
        
        self.modalities = []
        
        if self.use_raw:
            self.raw_encoder = RawEncoder(cfg)
            self.raw_pos = PositionalEncoding(40, embed_dim)
            self.modalities.append('raw')
            
        if self.use_stft:
            # Short STFT
            self.stft_s_trans = STFTTransform(n_fft=cfg.stft.short['win'], hop_length=cfg.stft.short['hop'], fmax=cfg.stft.fmax, input_bn=cfg.stft.input_bn)
            self.stft_s_enc = STFTEncoder(in_channels=2, target_freq_bins=8)
            self.stft_s_pos = PositionalEncoding(80, embed_dim) # 8 x 10
            
            # Long STFT
            self.stft_l_trans = STFTTransform(n_fft=cfg.stft.long['win'], hop_length=cfg.stft.long['hop'], fmax=cfg.stft.fmax, input_bn=cfg.stft.input_bn)
            self.stft_l_enc = STFTEncoder(in_channels=2, target_freq_bins=8)
            self.stft_l_pos = PositionalEncoding(40, embed_dim) # 8 x 5
            
            self.modalities.extend(['stft_s', 'stft_l'])
            
        if self.use_grp:
            self.grp_trans = GRPTransform(decimate=cfg.grp.decimate, m=cfg.grp.m, tau=cfg.grp.tau, percentile=cfg.grp.percentile, image_size=cfg.grp.image_size)
            self.grp_enc = GRPEncoder(in_channels=2)
            self.grp_pos = PositionalEncoding(64, embed_dim) # 8 x 8
            self.modalities.append('grp')
            
        if self.use_hilbert:
            self.hilbert_trans = HilbertTransform(fmax=cfg.hilbert.fmax, n_bands=cfg.hilbert.n_bands, time_bins=cfg.hilbert.time_bins)
            self.hilbert_enc = HilbertEncoder(in_channels=2)
            self.hilbert_pos = PositionalEncoding(40, embed_dim) # 2 x 20
            self.modalities.append('hilbert')

        # Modality embeddings
        self.modality_emb = nn.ParameterDict({
            m: nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            for m in self.modalities
        })

        self.fusion_type = cfg.model.fusion
        
        if self.fusion_type == 'cross_attention':
            self.cross_attn = CrossAttentionFusion(
                num_layers=cfg.model.ca_layers,
                embed_dim=embed_dim,
                num_heads=cfg.model.heads,
                ffn_dim=cfg.model.ffn_dim,
                dropout=cfg.model.attn_dropout
            )
            
        if cfg.model.use_transformer:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.transformer = TransformerEncoder(
                num_layers=cfg.model.tf_layers,
                embed_dim=embed_dim,
                num_heads=cfg.model.heads,
                ffn_dim=cfg.model.ffn_dim,
                dropout=cfg.model.dropout,
                attn_dropout=cfg.model.attn_dropout
            )
        elif self.fusion_type == 'gated':
            self.gated_fusion = GatedFusion(embed_dim, len(self.modalities))
            
        self.classifier = ClassifierHead(
            in_features=embed_dim,
            hidden_features=cfg.model.head_hidden,
            dropout=cfg.model.dropout
        )

    def forward(self, x):
        # x is [B, 2, 10240]
        tokens_dict = {}
        
        if self.use_raw:
            raw_t = self.raw_encoder(x)
            raw_t = self.raw_pos(raw_t) + self.modality_emb['raw']
            tokens_dict['raw'] = raw_t
            
        if self.use_stft:
            stft_s_img = self.stft_s_trans(x)
            stft_s_t = self.stft_s_enc(stft_s_img)
            stft_s_t = self.stft_s_pos(stft_s_t) + self.modality_emb['stft_s']
            tokens_dict['stft_s'] = stft_s_t
            
            stft_l_img = self.stft_l_trans(x)
            stft_l_t = self.stft_l_enc(stft_l_img)
            stft_l_t = self.stft_l_pos(stft_l_t) + self.modality_emb['stft_l']
            tokens_dict['stft_l'] = stft_l_t
            
        if self.use_grp:
            grp_img = self.grp_trans(x)
            grp_t = self.grp_enc(grp_img)
            grp_t = self.grp_pos(grp_t) + self.modality_emb['grp']
            tokens_dict['grp'] = grp_t
            
        if self.use_hilbert:
            hilbert_img = self.hilbert_trans(x)
            hilbert_t = self.hilbert_enc(hilbert_img)
            hilbert_t = self.hilbert_pos(hilbert_t) + self.modality_emb['hilbert']
            tokens_dict['hilbert'] = hilbert_t

        if self.fusion_type == 'cross_attention':
            tokens_dict = self.cross_attn(tokens_dict)

        if self.cfg.model.use_transformer:
            # Concatenate CLS token and all modalities
            B = x.size(0)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            
            # Keep a fixed order
            concat_list = [cls_tokens]
            for m in self.modalities:
                concat_list.append(tokens_dict[m])
                
            all_tokens = torch.cat(concat_list, dim=1)
            
            # Transformer
            tf_out = self.transformer(all_tokens)
            
            # Extract CLS token
            z = tf_out[:, 0, :]
            
        elif self.fusion_type == 'gated':
            # Gated fusion on tokens
            tokens_list = [tokens_dict[m] for m in self.modalities]
            z = self.gated_fusion(tokens_list)
            
        else:
            # Simple global average pooling fallback
            pooled = [torch.mean(tokens_dict[m], dim=1) for m in self.modalities]
            z = torch.stack(pooled, dim=1).mean(dim=1)
            
        # Classifier
        logits = self.classifier(z)
        
        return logits

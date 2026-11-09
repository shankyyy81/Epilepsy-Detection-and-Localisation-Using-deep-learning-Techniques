import torch
import torch.nn as nn
import torch.nn.functional as F
from .cbam import CBAM
from .stft import ResNetCBAMBlock

class GRPTransform(nn.Module):
    def __init__(self, decimate=8, m=3, tau=2, percentile=10.0, image_size=128):
        super().__init__()
        self.decimate = decimate
        self.m = m
        self.tau = tau
        self.percentile = percentile
        self.image_size = image_size
        
        # Simple anti-aliasing filter before decimation (moving average or low-pass)
        # Using a simple moving average for simplicity here, though a proper lowpass is better.
        self.register_buffer('filter_b', torch.ones(decimate) / decimate)
        self.register_buffer('filter_a', torch.tensor([1.0] + [0.0]*(decimate-1)))

    def forward(self, x):
        B, C, T = x.size()
        
        # Low-pass filter (anti-alias)
        # using F.conv1d as a moving average
        # x is [B, C, T]
        B, C, T = x.size()
        x_reshaped = x.view(B*C, 1, T)
        x_filtered = F.conv1d(x_reshaped, self.filter_b.view(1, 1, -1), padding=self.decimate//2)
        x_filtered = x_filtered[:, :, :T]
        x_filtered = x_filtered.view(B, C, T)
        
        # Decimate
        x_dec = x_filtered[:, :, ::self.decimate] # [B, C, T // 8] = [B, 2, 1280]
        
        # Delay embedding
        # L = T_dec - (m-1)*tau
        B, C, T_dec = x_dec.size()
        L = T_dec - (self.m - 1) * self.tau
        
        # Create embedding [B, C, L, m]
        embeds = []
        for i in range(self.m):
            start = i * self.tau
            end = start + L
            embeds.append(x_dec[:, :, start:end].unsqueeze(-1))
        
        embed_tensor = torch.cat(embeds, dim=-1) # [B, C, L, m]
        
        # Compute pairwise distance matrix (cdist in fp32)
        embed_tensor_fp32 = embed_tensor.float()
        
        # Flatten B and C for cdist: [B*C, L, m]
        flat_embed = embed_tensor_fp32.view(B*C, L, self.m)
        dist_matrix = torch.cdist(flat_embed, flat_embed, p=2.0) # [B*C, L, L]
        
        # Threshold at per-sample percentile
        # We need to compute the threshold for each sample in B*C
        # Reshape to [B*C, L*L] to find percentile
        dist_flat = dist_matrix.view(B*C, -1)
        # torch.quantile is slow and memory intensive, but works for B*C
        epsilon = torch.quantile(dist_flat, self.percentile / 100.0, dim=-1, keepdim=True).unsqueeze(-1)
        
        # Recurrence matrix: 1 if dist <= eps, 0 otherwise
        recurrence = (dist_matrix <= epsilon).float() # [B*C, L, L]
        recurrence = recurrence.view(B, C, L, L)
        
        # Adaptive avg pool to image_size x image_size
        if recurrence.device.type == 'mps':
            out = F.adaptive_avg_pool2d(recurrence.cpu(), (self.image_size, self.image_size)).to(recurrence.device)
        else:
            out = F.adaptive_avg_pool2d(recurrence, (self.image_size, self.image_size))
        
        return out

class GRPEncoder(nn.Module):
    def __init__(self, in_channels=2, widths=[32, 64, 128], blocks=[1, 1, 1]):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, widths[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0])
        self.act1 = nn.GELU()
        
        # Layers
        layers = []
        in_c = widths[0]
        # First block stride 2, subsequent blocks stride 2 to reduce size
        # 128x128 -> conv1 -> 64x64
        # block1 -> 32x32
        # block2 -> 16x16
        # block3 -> 8x8
        strides = [2, 2, 2]
        
        for w, num_blocks, stride in zip(widths, blocks, strides):
            layers.append(ResNetCBAMBlock(in_c, w, stride=stride))
            for _ in range(1, num_blocks):
                layers.append(ResNetCBAMBlock(w, w, stride=1))
            in_c = w
            
        self.encoder = nn.Sequential(*layers)
        self.proj = nn.Conv2d(widths[-1], 256, 1) # 1x1 projection to 256
        
    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.encoder(x)
        x = self.proj(x)
        
        # Flatten H x W -> tokens
        x = x.flatten(2) # [B, 256, 64]
        return x.transpose(1, 2) # [B, 64, 256]

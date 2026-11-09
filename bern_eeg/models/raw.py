import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiKernelConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernels=(7, 15, 31), stride=1):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, k, stride=stride, padding=k//2)
            for k in kernels
        ])
        self.proj = nn.Conv1d(out_channels * len(kernels), out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        
    def forward(self, x):
        outs = [conv(x) for conv in self.convs]
        out = torch.cat(outs, dim=1)
        return self.act(self.bn(self.proj(out)))

class DilatedResBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=(1, 2, 4, 8), stride=1):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, 3, stride=stride, 
                      padding=d, dilation=d)
            for i, d in enumerate(dilations)
        ])
        self.proj = nn.Conv1d(out_channels * len(dilations), out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        res = self.shortcut(x)
        # Note: if stride != 1, only the first conv should stride. 
        # But for parallel dilated convs, strided dilated convs are a bit weird.
        # Let's adjust: apply a strided projection first if stride > 1, then parallel dilated.
        # Actually, let's keep it simple: parallel convs, all with stride=1, followed by a pool, or just the first conv strided.
        # To avoid shape mismatch, let's just do parallel convs with padding=d*dilation, but stride applied to all.
        outs = [conv(x) for conv in self.convs]
        out = torch.cat(outs, dim=1)
        out = self.bn(self.proj(out))
        return self.act(out + res)

class SEBlock1d(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.shortcut(x))

class RawEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Input: [B, 2, 10240]
        # Stem: depthwise Conv1d(k=15, s=2, x8), pointwise (16->32)
        self.stem = nn.Sequential(
            nn.Conv1d(2, 16, 15, stride=2, padding=7, groups=2),
            nn.Conv1d(16, 32, 1),
            nn.BatchNorm1d(32),
            nn.GELU()
        ) # [B, 32, 5120]
        
        # Stage 1: multi-kernel residual, stride 4
        self.stage1 = nn.Sequential(
            MultiKernelConv1d(32, 64, kernels=(7, 15, 31), stride=4),
            ResBlock1d(64, 64, stride=1)
        ) # [B, 64, 1280]
        
        # Stage 2: dilated residual, stride 4
        self.stage2 = nn.Sequential(
            DilatedResBlock1d(64, 128, dilations=(1, 2, 4, 8), stride=4),
            ResBlock1d(128, 128, stride=1)
        ) # [B, 128, 320]
        
        # Stage 3: dilated residual, stride 4
        self.stage3 = nn.Sequential(
            DilatedResBlock1d(128, 256, dilations=(1, 2, 4, 8), stride=4),
            ResBlock1d(256, 256, stride=1)
        ) # [B, 256, 80]
        
        # Stage 4: residual, stride 2, SE attention
        self.stage4 = nn.Sequential(
            ResBlock1d(256, 256, stride=2),
            SEBlock1d(256)
        ) # [B, 256, 40]

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        # Transpose to [B, 40, 256]
        return x.transpose(1, 2)

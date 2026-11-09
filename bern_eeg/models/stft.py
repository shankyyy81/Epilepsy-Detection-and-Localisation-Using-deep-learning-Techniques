import torch
import torch.nn as nn
from .cbam import CBAM

class ResNetCBAMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.cbam = CBAM(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.act2 = nn.GELU()

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out += self.shortcut(x)
        return self.act2(out)

class STFTEncoder(nn.Module):
    def __init__(self, in_channels=2, widths=[32, 64, 128, 256], blocks=[2, 2, 2, 2], strides=[1, 2, 2, 2], target_freq_bins=8):
        super().__init__()
        self.target_freq_bins = target_freq_bins
        
        self.conv1 = nn.Conv2d(in_channels, widths[0], kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0])
        self.act1 = nn.GELU()
        
        layers = []
        in_c = widths[0]
        for w, num_blocks, stride in zip(widths, blocks, strides):
            layers.append(ResNetCBAMBlock(in_c, w, stride=stride))
            for _ in range(1, num_blocks):
                layers.append(ResNetCBAMBlock(w, w, stride=1))
            in_c = w
            
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.encoder(x)
        
        # Adaptive pool over frequency dimension (H) to target_freq_bins, while keeping time dimension (W)
        B, C, H, W = x.size()
        if x.device.type == 'mps':
            x = torch.nn.functional.adaptive_avg_pool2d(x.cpu(), (self.target_freq_bins, W)).to(x.device)
        else:
            x = torch.nn.functional.adaptive_avg_pool2d(x, (self.target_freq_bins, W))
        
        # Flatten freq x time -> tokens
        x = x.flatten(2) # [B, C, freq_bins * time_bins]
        return x.transpose(1, 2) # [B, tokens, C]

class STFTTransform(nn.Module):
    def __init__(self, n_fft, hop_length, fmax, fs=512, input_bn=True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmax = fmax
        self.fs = fs
        self.register_buffer('window', torch.hann_window(n_fft))
        
        # We need to compute max frequency bin
        freqs = torch.fft.rfftfreq(n_fft, 1/fs)
        self.max_bin = torch.where(freqs <= fmax)[0][-1].item() + 1
        
        self.input_bn = None
        if input_bn:
            self.input_bn = nn.BatchNorm2d(2)

    def forward(self, x):
        B, C, T = x.size()
        x = x.view(B*C, T)
        
        # Returns [B*C, F, T_out] (complex)
        stft_out = torch.stft(
            x, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            window=self.window, 
            center=False,
            return_complex=True
        )
        
        # Magnitude/Power
        power = stft_out.abs().pow(2)
        log_power = torch.log1p(power)
        
        # Crop to fmax
        log_power = log_power[:, :self.max_bin, :]
        
        # Reshape back to [B, C, F, T_out]
        _, F, T_out = log_power.size()
        out = log_power.view(B, C, F, T_out)
        
        if self.input_bn is not None:
            out = self.input_bn(out)
            
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class HilbertTransform(nn.Module):
    def __init__(self, fs=512, fmin=0.5, fmax=128.0, n_bands=16, time_bins=80):
        super().__init__()
        self.fs = fs
        self.n_bands = n_bands
        self.time_bins = time_bins
        
        # Log-spaced frequencies
        freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_bands + 1)
        
        # Create simple IIR bandpass filters for each band
        # In a real scenario, use scipy.signal.butter or similar to generate coefficients
        # Here we mock this with a learnable filter bank for simplicity/differentiability,
        # or we could use FFT-based bandpass. Since it says "FFT-based analytic signal",
        # let's do it in frequency domain.
        
        self.register_buffer('freq_bands', torch.tensor(freqs))

    def forward(self, x):
        B, C, T = x.size()
        
        # FFT
        X = torch.fft.rfft(x) # [B, C, F]
        freqs = torch.fft.rfftfreq(T, 1/self.fs).to(x.device)
        
        envelopes = []
        for i in range(self.n_bands):
            f_low = self.freq_bands[i]
            f_high = self.freq_bands[i+1]
            
            # Mask
            mask = (freqs >= f_low) & (freqs < f_high)
            X_band = X * mask
            
            # IFFT to get analytic signal (multiply by 2 for positive freqs except DC and Nyquist)
            # Actually, torch.fft.rfft only has positive freqs. 
            # Analytic signal via IFFT of positive freqs:
            # We take irfft but need to double the non-DC components to get the real part of analytic,
            # or use complex ifft to get full analytic. 
            # A simpler way: time-domain signal = irfft(X_band). Hilbert transform = imag part.
            # Analytic signal = x + i*H(x).
            # Envelope = abs(Analytic).
            # We can use torch.fft.irfft to get real part, and shift phase by pi/2 for imaginary.
            # Or use full FFT: 
            X_full = torch.fft.fft(x)
            full_freqs = torch.fft.fftfreq(T, 1/self.fs).to(x.device)
            full_mask = (full_freqs >= f_low) & (full_freqs < f_high)
            X_band_full = X_full * full_mask
            X_band_full[:, :, full_freqs > 0] *= 2.0
            
            analytic = torch.fft.ifft(X_band_full)
            env = analytic.abs() # [B, C, T]
            envelopes.append(env)
            
        env_tensor = torch.stack(envelopes, dim=2) # [B, C, n_bands, T]
        
        # Log envelope
        log_env = torch.log1p(env_tensor)
        
        # Pool to time_bins
        if log_env.device.type == 'mps':
            out = F.adaptive_avg_pool2d(log_env.cpu(), (self.n_bands, self.time_bins)).to(log_env.device)
        else:
            out = F.adaptive_avg_pool2d(log_env, (self.n_bands, self.time_bins))
        
        return out

class HilbertEncoder(nn.Module):
    def __init__(self, in_channels=2, widths=[32, 64, 128]):
        super().__init__()
        # Input: [B, 2, 16, 80]
        self.conv1 = nn.Conv2d(in_channels, widths[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0])
        self.act1 = nn.GELU()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(widths[0], widths[1], 3, stride=2, padding=1, bias=False), # [B, 64, 8, 40]
            nn.BatchNorm2d(widths[1]),
            nn.GELU()
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(widths[1], widths[2], 3, stride=2, padding=1, bias=False), # [B, 128, 4, 20]
            nn.BatchNorm2d(widths[2]),
            nn.GELU()
        )
        
        self.proj = nn.Conv2d(widths[2], 256, 1)
        
    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.proj(x)
        
        # Output [B, 256, 4, 20]
        if x.device.type == 'mps':
            x = F.adaptive_avg_pool2d(x.cpu(), (2, 20)).to(x.device) # Pool to 2 freq bins to get 40 tokens total
        else:
            x = F.adaptive_avg_pool2d(x, (2, 20)) # Pool to 2 freq bins to get 40 tokens total
        x = x.flatten(2) # [B, 256, 40]
        return x.transpose(1, 2) # [B, 40, 256]

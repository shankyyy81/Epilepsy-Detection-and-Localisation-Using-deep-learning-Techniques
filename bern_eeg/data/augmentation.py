"""
Data augmentation module for EEG signals and spectrogram representations.
All operations are vectorized and support batched PyTorch tensors on CPU/CUDA/MPS.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EEGAugmentation(nn.Module):
    """
    Augmentation engine applying raw-signal perturbations and SpecAugment.
    x is expected to have shape [B, C, T] where C is channels and T is sequence length.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.aug_cfg = cfg.augment if hasattr(cfg, "augment") else {}

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.aug_cfg.get("enabled", False) or not self.training:
            return (x, y) if y is not None else x

        B, C, T = x.shape
        device = x.device

        # 1. Additive Gaussian noise (scaled relative to per-sample std)
        p_noise = self.aug_cfg.get("p_noise", 0.0)
        noise_std = self.aug_cfg.get("noise_std", 0.05)
        if p_noise > 0:
            mask = (torch.rand(B, 1, 1, device=device) < p_noise).float()
            std = x.std(dim=-1, keepdim=True) + 1e-6
            noise = torch.randn_like(x) * (noise_std * std)
            x = x + mask * noise

        # 2. Amplitude scaling
        p_scale = self.aug_cfg.get("p_scale", 0.0)
        scale_range = self.aug_cfg.get("scale_range", [0.8, 1.2])
        if p_scale > 0:
            mask = (torch.rand(B, 1, 1, device=device) < p_scale).float()
            scale = torch.empty(B, 1, 1, device=device).uniform_(scale_range[0], scale_range[1])
            x = (1 - mask) * x + mask * (x * scale)

        # 3. Random time shift
        p_shift = self.aug_cfg.get("p_shift", 0.0)
        shift_max_s = self.aug_cfg.get("shift_max_s", 0.25)
        fs = self.cfg.data.get("fs", 512)
        max_shift = int(shift_max_s * fs)
        if p_shift > 0 and max_shift > 0:
            for b in range(B):
                if torch.rand(1).item() < p_shift:
                    shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
                    x[b] = torch.roll(x[b], shifts=shift, dims=-1)

        # 4. Time masking (zero out consecutive blocks)
        p_time_mask = self.aug_cfg.get("p_time_mask", 0.0)
        time_mask_frac = self.aug_cfg.get("time_mask_frac", 0.05)
        if p_time_mask > 0:
            mask_len = int(time_mask_frac * T)
            if mask_len > 0:
                for b in range(B):
                    if torch.rand(1).item() < p_time_mask:
                        start = int(torch.randint(0, max(1, T - mask_len), (1,)).item())
                        x[b, :, start:start + mask_len] = 0.0

        # 5. Channel dropout
        p_ch_drop = self.aug_cfg.get("p_channel_dropout", 0.0)
        if p_ch_drop > 0 and C > 1:
            for b in range(B):
                if torch.rand(1).item() < p_ch_drop:
                    drop_idx = int(torch.randint(0, C, (1,)).item())
                    x[b, drop_idx, :] = 0.0

        # 6. Polarity inversion
        p_polarity = self.aug_cfg.get("p_polarity", 0.0)
        if p_polarity > 0:
            mask = (torch.rand(B, 1, 1, device=device) < p_polarity).float()
            x = (1 - 2 * mask) * x

        # 7. Mixup (if enabled and y provided)
        p_mixup = self.aug_cfg.get("p_mixup", 0.0)
        mixup_alpha = self.aug_cfg.get("mixup_alpha", 0.2)
        if p_mixup > 0 and y is not None and B > 1 and torch.rand(1).item() < p_mixup:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            idx = torch.randperm(B, device=device)
            x = lam * x + (1.0 - lam) * x[idx]
            y = lam * y + (1.0 - lam) * y[idx]

        return (x, y) if y is not None else x


def apply_specaugment(spec: torch.Tensor, freq_mask_param: int = 8, time_mask_param: int = 10) -> torch.Tensor:
    """
    SpecAugment for 2D/3D/4D spectrograms [B, C, F, T].
    Zeros out random frequency channels and time slices.
    """
    if spec.dim() == 4:
        B, C, F, T = spec.shape
        for b in range(B):
            # Frequency masking
            if freq_mask_param > 0 and F > freq_mask_param:
                f = int(torch.randint(0, freq_mask_param, (1,)).item())
                f0 = int(torch.randint(0, F - f, (1,)).item())
                spec[b, :, f0:f0 + f, :] = 0.0
            # Time masking
            if time_mask_param > 0 and T > time_mask_param:
                t = int(torch.randint(0, time_mask_param, (1,)).item())
                t0 = int(torch.randint(0, T - t, (1,)).item())
                spec[b, :, :, t0:t0 + t] = 0.0
    return spec

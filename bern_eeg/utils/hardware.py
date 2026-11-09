"""Device selection, AMP settings, automatic batch-size probing."""
from __future__ import annotations

import gc
import platform
from contextlib import nullcontext
from typing import Callable, Optional, Sequence

import torch


def get_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_enabled(device: torch.device, requested: bool) -> bool:
    # fp16 autocast + GradScaler only on CUDA (RTX 2080 = Turing, fp16 tensor cores, no bf16).
    return bool(requested and device.type == "cuda")


def autocast_ctx(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def resolve_num_workers(value, device: torch.device) -> int:
    if value != "auto":
        return int(value)
    # Data are held in RAM and all transforms run on the GPU, so workers only shuttle
    # tensors. On macOS (spawn) workers would pickle the whole array -> use 0.
    if platform.system() == "Darwin" or device.type != "cuda":
        return 0
    return 2


def empty_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def auto_batch_size(build_model: Callable[[], torch.nn.Module], sample_shape: Sequence[int],
                    device: torch.device, candidates=(64, 32, 16, 8, 4), amp: bool = True,
                    logger=None) -> Optional[int]:
    """Largest batch from `candidates` for which forward+backward fits in GPU memory (CUDA)."""
    if device.type != "cuda":
        return None
    for bs in candidates:
        model = None
        try:
            model = build_model().to(device)
            model.train()
            x = torch.randn(bs, *sample_shape, device=device)
            with autocast_ctx(device, amp):
                out = model(x)
                loss = out.float().mean()
            loss.backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            if peak < 0.85 * total:
                if logger:
                    logger.info(f"auto_batch_size: batch {bs} fits (peak {peak:.2f}/{total:.2f} GB)")
                return bs
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
        finally:
            del model
            empty_cache(device)
            torch.cuda.reset_peak_memory_stats()
    return candidates[-1]

"""Logging, TensorBoard wrapper and environment banner."""
from __future__ import annotations

import logging
import os
import platform
import sys
from typing import Optional

import torch


def get_logger(name: str = "bern_eeg", log_file: Optional[str] = None,
               level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file:
        log_file = os.path.abspath(log_file)
        exists = any(isinstance(h, logging.FileHandler) and h.baseFilename == log_file
                     for h in logger.handlers)
        if not exists:
            # remove previous file handlers (new experiment)
            for h in [h for h in logger.handlers if isinstance(h, logging.FileHandler)]:
                logger.removeHandler(h)
                h.close()
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


class TBWriter:
    """TensorBoard SummaryWriter if available, otherwise a no-op."""

    def __init__(self, log_dir: str, enabled: bool = True):
        self.writer = None
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=log_dir)
            except Exception:  # tensorboard not installed
                self.writer = None

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


def environment_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "gpu_name": None,
        "gpu_total_memory_gb": None,
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu_name"] = p.name
        info["gpu_total_memory_gb"] = round(p.total_memory / 1024 ** 3, 2)
    try:
        import numpy, scipy, sklearn
        info.update(numpy=numpy.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__)
    except Exception:
        pass
    return info


def log_environment(logger: logging.Logger) -> dict:
    info = environment_info()
    logger.info("=" * 70)
    logger.info(f"torch.cuda.is_available(): {info['cuda_available']}")
    logger.info(f"PyTorch {info['torch']} | CUDA {info['cuda_version']} | cuDNN {info['cudnn_version']}")
    if info["cuda_available"]:
        logger.info(f"GPU: {info['gpu_name']} ({info['gpu_total_memory_gb']} GB)")
        logger.info(gpu_memory_string())
    elif info["mps_available"]:
        logger.info("GPU: Apple MPS backend (AMP disabled on MPS)")
    else:
        logger.info("GPU: none detected - running on CPU")
    logger.info(f"Python {info['python']} | {info['platform']}")
    logger.info("=" * 70)
    return info


def gpu_memory_string() -> str:
    if not torch.cuda.is_available():
        return "GPU memory: n/a"
    a = torch.cuda.memory_allocated() / 1024 ** 2
    r = torch.cuda.memory_reserved() / 1024 ** 2
    m = torch.cuda.max_memory_allocated() / 1024 ** 2
    return f"GPU memory allocated {a:.0f} MB | reserved {r:.0f} MB | peak {m:.0f} MB"
